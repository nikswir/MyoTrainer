# MyoTrainer — a PPO harness for musculoskeletal muscle control

[![CI](https://github.com/nikswir/MyoTrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/nikswir/MyoTrainer/actions/workflows/ci.yml)

MyoTrainer is a reinforcement-learning **training harness** for muscle-actuated
musculoskeletal control. It provides a **PPO** training loop, **On-Policy
Behavior Cloning (OBC)** distillation, and a set of **policy architectures**,
wired to run out of the box against two environments: **Kinesis** (MyoLegs,
lower body) and **[MyoHuman](https://github.com/nikswir/Myohuman)** (full body,
338 muscles).

> **MyoTrainer is the *trainer*.** The body model, task and motion data come
> from **[MyoHuman](https://github.com/nikswir/Myohuman)** — this repo is the
> algorithms and network architectures that learn to act in it.

> The muscle transformer is adapted from **Arnold** (Chiappa et al.,
> [arXiv:2508.18066](https://arxiv.org/abs/2508.18066)).

<p align="center">
  <img src="docs/assets/imitation.gif" width="640"
       alt="Trained MyoTrainer policy imitating motion clips in MyoHuman">
  <br>
  <em>A trained policy (MLP + LATTICE) tracking held-out KIT motions in
  MyoHuman: walking, turning and arm movements. Muscles are coloured by
  activation; the translucent blue skeleton is the reference motion.</em>
</p>

## What it provides

A config-driven ([Hydra](https://hydra.cc)) harness around a PPO training loop:

- **Environments** — pluggable through expert wrappers: **Kinesis** (MyoLegs,
  lower-body) and **[MyoHuman](https://github.com/nikswir/Myohuman)**
  (full-body, 338 muscles).
- **Algorithms** — **PPO** (with GAE) for on-policy RL and **On-Policy Behavior
  Cloning (OBC)** for policy distillation; the two are mixed per expert through
  loss weights.
- **Policy architectures**
  - **MLP** actor–critic with **LATTICE** latent time-correlated exploration;
  - a **muscle transformer** — an encoder–decoder that tokenizes observations
    and muscles through a *sensorimotor vocabulary*, with configurable
    observation/action **granulation** and a hierarchical group-to-group /
    within-group decoder;
  - a **mixture-of-experts gate** that routes between frozen expert policies.
- **Tooling** — hard-negative mining, policy validation & metrics, Weights &
  Biases logging, a profiler, and multi-GPU (DataParallel) training.

## Setup

Requires **Git**, **[Git LFS](https://git-lfs.com/)** and
**[uv](https://docs.astral.sh/uv/)** (installs Python 3.12+ itself).

```bash
git lfs install                                             # once per machine
git clone --recurse-submodules <repo-url> MyoTrainer        # + Kinesis, MyoHuman
cd MyoTrainer
git lfs pull                                                # heavy assets under downloads/
./scripts/install.sh                                        # uv sync into .venv
./scripts/setup_experts.sh                                  # patch submodules, stage assets, fetch Kinesis model
```

`setup_experts.sh` initializes submodules, pulls their LFS files (including
MyoHuman), patches and stages the Kinesis assets into
`src/myotrainer/experts/Kinesis/data/`, and downloads the `kinesis-moe-imitation`
expert from Hugging Face.

Already cloned without submodules?
`git submodule update --init --recursive && git lfs pull`.

## Usage

Training is configured with [Hydra](https://hydra.cc) and driven through one
entry point; environments are added as expert config groups.

```bash
# OBC distillation from the Kinesis expert
uv run python -m myotrainer.train_myotrainer \
    '+run/experts@run.experts.kin=kinesis' \
    exp_name=obc_kinesis

# PPO in MyoHuman
uv run python -m myotrainer.train_myotrainer \
    '+run/experts@run.experts.myo=myohuman' \
    run.experts.myo.learning.ppo_weight=1.0 \
    run.experts.myo.learning.obc_imitation_weight=0 \
    run.num_threads=15 \
    exp_name=ppo_myohuman

# Multi-GPU (DataParallel)
uv run python -m myotrainer.train_myotrainer \
    device_ids=[0,1] '+run/experts@run.experts.kin=kinesis' exp_name=obc_2gpu
```

Validate a trained policy and mine hard negatives:

```bash
uv run python -m myotrainer.validate_policy   # roll out & score imitation metrics
uv run python -m myotrainer.mine_negatives    # select worst-imitated clips
```

Render a GIF of successful imitations (diverse clips picked by reference
kinematics, policy rollout + translucent reference ghost):

```bash
uv run python -m myotrainer.render_gifs device=mps run=render_gifs \
    run.checkpoint=/path/to/model.pth \
    '+run/experts@run.experts.myo=myohuman' learning.policy=lattice
```

## Repo layout

```text
src/myotrainer/
├── train_myotrainer.py   entry point — launches a PPO / OBC run
├── trainer.py            training loop: rollouts, PPO update, OBC, checkpoints
├── torch_model/          policy architectures — transformer, MLP+LATTICE, MoE gate, tokenizers
├── experts/              environment wrappers — Kinesis (MyoLegs), MyoHuman
├── observation_parser.py, action_parser.py   observation / action token layout
├── mine_negatives.py, validate_policy.py      hard-negative mining, evaluation
├── render_gifs.py        GIF of successful imitations (docs/assets/imitation.gif)
└── logger.py, wandb_logger.py, profiler.py, memory.py   logging & utilities
cfg/                      Hydra config (config.yaml, learning/, run/, run/experts/)
downloads/Kinesis_assets/ SMPL, motion dicts, initial poses (Git LFS)
scripts/                  install.sh, setup_experts.sh
tests/                    characterization tests (+ stage-2 device gate)
```

Submodules: **Kinesis** and **MyoHuman**. More on the environment wrappers:
[src/myotrainer/experts/README.md](src/myotrainer/experts/README.md).

## How it relates to MyoHuman

| | [MyoHuman](https://github.com/nikswir/Myohuman) | MyoTrainer (this repo) |
| --- | --- | --- |
| Role | the **environment** — body, task, data | the **trainer** — policies & algorithms |
| Provides | `MyoHumanIm`, MJCF model, retargeted motions | PPO / OBC harness, muscle transformer, MoE gate |

## References

- **Arnold** — Chiappa et al., *A Generalist Muscle Transformer Policy*, 2025,
  [arXiv:2508.18066](https://arxiv.org/abs/2508.18066).
- **Kinesis** — Simos et al., *Reinforcement Learning-Based Motion Imitation
  for Physiologically Plausible Musculoskeletal Motor Control*, 2025,
  [arXiv:2503.14637](https://arxiv.org/abs/2503.14637).
- **LATTICE** — Chiappa et al., 2023 — latent time-correlated exploration.
- **PPO** — Schulman et al., 2017. **DeepMimic** — Peng et al., 2018.
- **MyoSuite** — Caggiano et al., 2022.

## Development

The engineering workflow — toolchain (uv), the pre-commit gate, two-stage
tests, CI — is documented in [AGENTS.md](AGENTS.md).

```bash
uv run pre-commit run --all-files   # lint, format, types, style checks
uv run pytest                       # stage-1 (fast, CPU) tests
RUN_STAGE2=1 uv run pytest          # + device-parity tests
```
