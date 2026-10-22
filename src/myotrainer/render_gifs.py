"""
GIF с успешными имитациями движений обученной MyoTrainer policy.

1. Отбор: для каждого движения eval-сета по кинематике референса
   выбирается самое активное окно run.clip_seconds и считаются признаки
   (ходьба, поворот, высота таза, руки, ноги). Клипы выбираются
   farthest-point по этим признакам — максимально непохожие друг на друга.
2. Проверка: policy (deterministic) прогоняется headless с начала
   движения до конца окна; клип засчитывается, если эпизод не оборвался.
3. Рендер: offscreen mujoco.Renderer — модель (мышцы по активации) +
   полупрозрачный призрак референса. Клипы идут друг за другом,
   GIF зациклен.

Использование:
    uv run python -m myotrainer.render_gifs \
        device=mps \
        run=render_gifs \
        run.checkpoint=/path/to/model.pth \
        '+run/experts@run.experts.myo=myohuman' \
        learning.policy=lattice

Параметры (run=render_gifs):
    run.checkpoint:    путь к .pth (обязательный)
    run.num_clips:     сколько движений в GIF
    run.clip_seconds:  длина каждого клипа, сек
    run.output:        путь к .gif
    run.frame_stride:  шаг по кадрам (30 Hz control / stride = fps)
    run.ghost_alpha:   прозрачность референса
"""

import copy
import hydra
import torch
import mujoco
import logging
import numpy as np

from pathlib import Path
from omegaconf import DictConfig
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

from myotrainer.action_parser import ActionParser
from myotrainer.validate_policy import _load_policy
from myotrainer.observation_parser import ObservationParser


logger = logging.getLogger(__name__)

FEATURES = ("walk", "turn", "height", "arms", "legs")
ARM_JOINTS = ("sternoclavicular2_r", "pro_sup_l")  # первый / последний
LEG_JOINTS = ("hip_flexion_r", "knee_angle_beta_rotation1_l")


########################################
#            Clip selection            #
########################################


@dataclass
class Window:
    motion_id: int
    start: float  # сек от начала движения
    features: np.ndarray  # (len(FEATURES),)


def joint_qpos_slice(model: mujoco.MjModel, names: tuple) -> slice:
    """qpos-диапазон между первым и последним суставом группы."""
    first, last = (model.jnt_qposadr[model.joint(n).id] for n in names)
    return slice(int(first), int(last) + 1)


def root_yaw(qpos: np.ndarray) -> np.ndarray:
    """Yaw корня из кватерниона (w, x, y, z) в qpos[3:7]."""
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)))


def best_window(
    ref: np.ndarray,
    motion_id: int,
    dt: float,
    run: DictConfig,
    arms: slice,
    legs: slice,
) -> Window:
    """Окно с максимальной суммарной активностью (признаки / clip_seconds)."""
    length = int(run.clip_seconds / dt)
    step = int(run.window_step / dt)
    yaw = root_yaw(ref)
    joint_speed = np.abs(np.diff(ref, axis=0)) / dt

    windows = []
    for s in range(0, len(ref) - length + 1, step):
        e = s + length
        features = np.array(
            [
                np.linalg.norm(np.diff(ref[s:e, :2], axis=0), axis=1).sum(),
                np.abs(np.diff(yaw[s:e])).sum(),
                np.ptp(ref[s:e, 2]),
                joint_speed[s : e - 1, arms].mean(),
                joint_speed[s : e - 1, legs].mean(),
            ],
        )
        windows.append(Window(motion_id, s * dt, features / run.clip_seconds))
    return max(windows, key=lambda w: w.features.sum())


def rank_windows(env, run: DictConfig) -> list[Window]:
    """
    Окна-кандидаты в порядке farthest-point по нормированным признакам:
    первым — самое активное, дальше — самое непохожее на уже выбранные.
    """
    arms = joint_qpos_slice(env.mj_model, ARM_JOINTS)
    legs = joint_qpos_slice(env.mj_model, LEG_JOINTS)

    # ── 1. Лучшее окно каждого длинного движения ──
    windows = []
    for motion_id in env._all_motion_ids:
        if env.get_motion_length(int(motion_id)) < run.clip_seconds:
            continue
        frames = env.initial_pos_data[motion_id]
        ref = np.stack([frames[t] for t in sorted(frames)])
        windows.append(
            best_window(ref, int(motion_id), env.dt, run, arms, legs),
        )

    # ── 2. Нормировка + активная половина ─────────
    feats = np.stack([w.features for w in windows])
    feats = feats / feats.std(axis=0)
    activity = feats.sum(axis=1)
    keep = activity >= np.median(activity)
    windows = [w for w, k in zip(windows, keep, strict=True) if k]
    feats, activity = feats[keep], activity[keep]

    # ── 3. Farthest-point порядок ─────────────────
    order = [int(activity.argmax())]
    min_dist = np.linalg.norm(feats - feats[order[0]], axis=1)
    while len(order) < len(windows):
        nxt = int(min_dist.argmax())
        order.append(nxt)
        min_dist = np.minimum(
            min_dist,
            np.linalg.norm(feats - feats[nxt], axis=1),
        )
    return [windows[i] for i in order]


########################################
#               Rollouts               #
########################################


@dataclass
class Clip:
    window: Window
    qpos: np.ndarray  # (T, nq) симуляция
    act: np.ndarray  # (T, na) активации мышц
    ref_qpos: np.ndarray  # (T, nq_ref) референс


def rollout_window(
    cfg: DictConfig,
    expert,
    policy,
    parser: ObservationParser,
    action_parser: ActionParser,
    window: Window,
) -> Clip | None:
    """Эпизод с начала движения до конца окна; None — если упала раньше."""
    env = expert.env
    device = torch.device(cfg.device)
    expert_name = next(iter(cfg.run.experts.values())).type
    window_end = window.start + cfg.run.clip_seconds

    # ── Выбор движения (как в env.forward_motions) ─
    env._current_eval_motion_id = window.motion_id
    env._active_motion_ids = np.array([window.motion_id])
    obs, _ = expert.reset()
    parser.reset(obs)

    qpos, act, ref_qpos = [], [], []
    while True:
        # ── 1. Action ─────────────────────────────
        obs_ts, obs_sigs = parser.get_observation(device)
        with torch.no_grad():
            action, _, _, _ = policy.get_action(
                obs_ts,
                obs_sigs,
                action_parser.action_signatures,
                expert_name=expert_name,
                deterministic=True,
                return_std=not cfg.run.fast_eval,
                return_value=not cfg.run.fast_eval,
            )
        action_np = action.squeeze(0).cpu().numpy()

        # ── 2. Step ───────────────────────────────
        obs, _, terminated, truncated, _ = expert.step(action_np)
        parser.update(obs)
        sim_time = env.cur_t * env.dt + env._motion_start_time

        # ── 3. Record внутри окна ─────────────────
        if sim_time >= window.start:
            ref = env.get_state_from_motionlib_cache(sim_time)
            qpos.append(env.mj_data.qpos.copy())
            act.append(env.mj_data.act.copy())
            ref_qpos.append(ref["qpos"].copy())

        if terminated:
            return None
        if truncated or sim_time >= window_end:
            break

    return Clip(
        window=window,
        qpos=np.stack(qpos),
        act=np.stack(act),
        ref_qpos=np.stack(ref_qpos),
    )


def collect_clips(
    cfg: DictConfig,
) -> tuple[list[Clip], mujoco.MjModel, float]:
    """Прогон кандидатов в farthest-point порядке до num_clips успешных."""
    from myotrainer.trainer import create_expert_wrapper

    experts_list = list(cfg.run.experts.values())
    if len(experts_list) != 1:
        raise ValueError("render_gifs supports single expert only")

    # ── Env + policy ──────────────────────────────
    expert = create_expert_wrapper(
        experts_list[0],
        mode=cfg.run.mode,
        overrides=["run.headless=true"],
    )
    env = expert.env
    parser = ObservationParser.from_env(
        env,
        history_len=cfg.learning.history_len,
    )
    action_parser = ActionParser.from_env(env)
    policy = _load_policy(
        cfg,
        str(cfg.run.checkpoint),
        torch.device(cfg.device),
        expert,
        parser,
    )

    # ── Кандидаты → rollouts ──────────────────────
    candidates = rank_windows(env, cfg.run)
    clips: list[Clip] = []
    env.start_eval(im_eval=True)
    try:
        for tried, window in enumerate(candidates, start=1):
            clip = rollout_window(
                cfg,
                expert,
                policy,
                parser,
                action_parser,
                window,
            )
            stats = ", ".join(
                f"{n}={v:.2f}"
                for n, v in zip(FEATURES, window.features, strict=True)
            )
            logger.info(
                f"[{tried}/{len(candidates)}] motion {window.motion_id} "
                f"@{window.start:.0f}s ({stats}): "
                f"{'success' if clip is not None else 'fail'}",
            )
            if clip is not None:
                clips.append(clip)
            if len(clips) == cfg.run.num_clips:
                break
    finally:
        env.end_eval()

    if len(clips) < cfg.run.num_clips:
        raise RuntimeError(
            f"Only {len(clips)} successful clips found, "
            f"need run.num_clips={cfg.run.num_clips}",
        )
    return clips, env.mj_model, env.dt


########################################
#              Rendering               #
########################################


class ClipRenderer:
    """
    Offscreen рендер: модель + полупрозрачный призрак референса на чистой
    тёмной сцене (без комнаты, приглушённый пол без отражений).
    """

    def __init__(self, env_model: mujoco.MjModel, run: DictConfig):
        self.run = run
        self.model = copy.copy(env_model)
        self.clean_scene()
        self.renderer = mujoco.Renderer(
            self.model,
            height=run.height,
            width=run.width,
        )
        self.sim_data = mujoco.MjData(self.model)
        self.ref_data = mujoco.MjData(self.model)
        self.font = ImageFont.load_default(size=22)

        # ── Мышцы по активации, без сайтов ────────
        self.sim_option = mujoco.MjvOption()
        self.sim_option.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
        self.sim_option.sitegroup[:] = 0

        # ── Призрак: только тело, без мышц ────────
        self.ghost_option = mujoco.MjvOption()
        self.ghost_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
        self.ghost_option.sitegroup[:] = 0
        self.ghost_rgba = np.array([0.3, 0.7, 1.0, run.ghost_alpha])

        self.camera = mujoco.MjvCamera()
        self.camera.distance = run.cam_distance
        self.camera.azimuth = run.cam_azimuth
        self.camera.elevation = run.cam_elevation

    def clean_scene(self) -> None:
        """Прячет геомы мира кроме пола; пол — приглушённая шахматка."""
        model = self.model
        floor = model.geom("floor").id
        plane = model.material("MatPlane").id

        # ── Offscreen буфер + сглаживание ─────────
        model.vis.global_.offwidth = self.run.width
        model.vis.global_.offheight = self.run.height
        model.vis.quality.offsamples = 8

        # ── Комната и логотип ─────────────────────
        for g in np.flatnonzero(model.geom_bodyid == 0):
            if g != floor:
                model.geom_rgba[g, 3] = 0
                model.geom_matid[g] = -1

        # ── Пол ───────────────────────────────────
        model.geom_matid[floor] = plane
        model.mat_texrepeat[plane] = [4, 4]
        model.mat_rgba[plane] = [0.55, 0.55, 0.6, 1]
        model.mat_reflectance[plane] = 0

    def render_clip(self, clip: Clip) -> list[Image.Image]:
        """Кадры клипа (palette); камера плавно следит за тазом."""
        frames = []
        lookat = clip.qpos[0, :3].copy()
        label = f"motion {clip.window.motion_id}"
        for t in range(0, len(clip.qpos), self.run.frame_stride):
            # ── 1. Позы симуляции и референса ───────
            self.sim_data.qpos[:] = clip.qpos[t]
            self.sim_data.act[:] = clip.act[t]
            self.ref_data.qpos[: clip.ref_qpos.shape[1]] = clip.ref_qpos[t]
            mujoco.mj_forward(self.model, self.sim_data)
            mujoco.mj_kinematics(self.model, self.ref_data)

            # ── 2. Камера (EMA по тазу, высота фикс.) ─
            lookat = 0.9 * lookat + 0.1 * clip.qpos[t, :3]
            self.camera.lookat[:2] = lookat[:2]
            self.camera.lookat[2] = 0.9

            # ── 3. Сцена: модель + призрак ──────────
            self.renderer.update_scene(
                self.sim_data,
                camera=self.camera,
                scene_option=self.sim_option,
            )
            scene = self.renderer.scene
            first_ghost = scene.ngeom
            mujoco.mjv_addGeoms(
                self.model,
                self.ref_data,
                self.ghost_option,
                mujoco.MjvPerturb(),
                mujoco.mjtCatBit.mjCAT_DYNAMIC,
                scene,
            )
            for i in range(first_ghost, scene.ngeom):
                scene.geoms[i].rgba = self.ghost_rgba

            # ── 4. Подпись + palette без dithering ──
            image = Image.fromarray(self.renderer.render())
            ImageDraw.Draw(image).text(
                (14, 10),
                label,
                fill=(230, 230, 230),
                font=self.font,
            )
            frames.append(
                image.quantize(
                    colors=256,
                    method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.NONE,
                ),
            )
        return frames


########################################
#             Entry point              #
########################################


def render_gifs(cfg: DictConfig) -> Path:
    run = cfg.run
    if not str(run.checkpoint or "").strip():
        raise ValueError(
            "run.checkpoint is required: "
            "python -m myotrainer.render_gifs run=render_gifs "
            "run.checkpoint=path/to/model.pth",
        )

    clips, model, dt = collect_clips(cfg)
    logger.info(
        f"Rendering {len(clips)} clips: {[c.window.motion_id for c in clips]}",
    )

    # ── Клипы друг за другом ──────────────────────
    clip_renderer = ClipRenderer(model, run)
    frames = []
    for clip in clips:
        frames.extend(clip_renderer.render_clip(clip))

    # ── GIF (loop=0 — бесконечный повтор) ─────────
    output = Path(run.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / (dt * run.frame_stride)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    size_mb = output.stat().st_size / 2**20
    logger.info(
        f"Saved {output} ({len(frames)} frames, {fps:.0f} fps, "
        f"{size_mb:.1f} MB)",
    )
    return output


@hydra.main(config_path="../../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    render_gifs(cfg)


if __name__ == "__main__":
    main()
