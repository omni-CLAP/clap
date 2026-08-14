<div align="center">
  <h1>👏 CLAP: Cross-Embodiment Action-Conditioned Video World Models are Zero-Shot Physical Simulators</h1>
</div>

<p align="center">
  <img alt="Paper" src="https://img.shields.io/badge/Paper-coming%20soon-lightgrey?logo=arxiv">
  <a href="https://omni-clap.github.io"><img alt="Website" src="https://img.shields.io/badge/%F0%9F%8C%90%20Website-Live-d15619"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue?logo=opensourceinitiative"></a>
  <a href="https://huggingface.co/omni-CLAP/CLAP"><img alt="HF Video Models" src="https://img.shields.io/badge/HF-Video%20Models-yellow?logo=huggingface"></a>
</p>


CLAP is a cross-embodiment, action-conditioned video generation framework that unifies disparate human and robot action spaces via end-effector poses, language, and latent actions.

- **Two-Stage Curriculum:** *Learns unsupervised physical priors from unlabeled video, then grounds them in end-effector space for zero-shot deployment.*

- **Sample-Efficient Adaptation:** *Facilitates few-shot transfer to target robots, matching or beating single-embodiment baselines like DROID (without post-training) and Bridge (with post-training).*

- **Zero-Shot Real-World Generalization:** *Generalizes out-of-the-box to power real-world inference planning and RL finetuning with robot policies like π₀.₅ and MolmoAct-2.*


This package covers the full lifecycle: cross-embodiment video modeling, post-training, novel-embodiment
adaptation, offline evaluation, and real-robot policy-in-the-loop deployment. New to the
terminology? See the [Glossary](#-glossary) for EE, LAM, OXE, DROID, and other terms used
throughout.

## Contents

- [📢 Updates](#-updates)
- [✨ Features](#-features)
- [📋 Requirements](#-requirements)
- [📦 Installation](#-installation)
  - [1. Core package](#1-core-package)
  - [2. Policy backends (optional)](#2-policy-backends-optional)
  - [3. Environment](#3-environment)
- [🏁 Getting started](#-getting-started)
  - [📥 Download a checkpoint (optional)](#-download-a-checkpoint-optional)
  - [🔁 Replay demonstration trajectories](#-replay-demonstration-trajectories)
  - [🎮 Live Teleop (DROID, Bridge, Bimanual YAM, G1 Humanoid)](#-live-teleop-for-droid-bridge-taco-play-bimanual-yam-g1-humanoid)
  - [🚀 Deploy (policy-in-the-loop)](#-deploy-policy-in-the-loop-on-droid-and-bimanual-yam-with-live-preview)
- [🧩 Sample-efficient adaptation to new embodiments](#-sample-efficient-adaptation-to-new-embodiments)
- [🗂️ Checkpoints](#-checkpoints)
  - [🌐 Cross-embodiment models](#-cross-embodiment-models)
  - [🎯 Post-trained on a single robot platform](#-post-trained-on-a-single-robot-platform)
  - [🆕 Novel-embodiment adaptation](#-novel-embodiment-adaptation)
- [📁 Project layout](#-project-layout)
- [📚 Reference](#-reference) (configuration, entrypoints, dataset registry, testing — [`docs/REFERENCE.md`](docs/REFERENCE.md))
- [🛠️ Tips and Troubleshooting](#-tips-and-troubleshooting)
- [🙏 Acknowledgment](#-acknowledgment)
- [📄 BibTeX](#-bibtex)
- [📖 Glossary](#-glossary)
- [⚖️ License](#-license)

## 📢 Updates
- **[2026/08/12]** 🚀 CLAP is live.


## ✨ Features

CLAP delivers the most comprehensive suite of action-conditioned video
world models to date — spanning diverse action-conditioning spaces (end-effector, language, and latent)
and robot morphologies, across DROID, Bridge, bimanual YAM robots, and
G1 humanoids.

| `conditioning` | Action representation | Typical use |
|---|---|---|
| 🦾 `"ee"` | 7-dim end-effector cartesian pose + gripper (or raw joint angles for joint-space embodiments) | Cross-embodiment, post-training, adaptation |
| 🧬 `"lam"` | 32-dim latent-action-model embedding (learned, not physical) | Curriculum cross-embodiment training, including egocentric human video |
| 💬 `"language"` | Per-frame CLIP-encoded text captions of the action | Language-conditioned, post-training, adaptation |

## 📋 Requirements

- **Python** ≥ 3.9 (use 3.11 for in-process openpi policies).
- **A CUDA GPU**: A single ≥12GB GPU (e.g., RTX 3060) comfortably runs eval/replay/teleop/deploy; in-process
  openpi deployment needs ≥32GB (e.g., RTX 5090) — server mode instead runs openpi on its own
  machine/GPU, so `clap-rollout-deploy` itself only needs CLAP's own ~10GB. Inference VRAM usage and timing statistics per nominal prediction (11 total frames, 25 denoising steps, mean ± std with n = 20 trials):

  | | A100-40GB | A100-80GB | H200 |
  |---|---|---|---|
  | CLAP alone, idle | 9.6GB | 9.6GB | 9.7GB |
  | + openpi in-process, idle | 17.6GB | 17.6GB | 17.7GB |
  | + openpi in-process, inference peak | **24.2GB** | **24.2GB** | **25.6GB** |
  | `predict_chunk` (steady-state) | 3.24s ± 0.02s | 2.88s ± 0.00s | 1.49s ± 0.00s |

  Steady-state excludes each process's first `predict_chunk` call, which pays a one-time
  cuDNN kernel-autotuning cost. How much that first call costs depends on whether anything
  else already touched the GPU first: for `clap-rollout-deploy --in-process` (the table above),
  openpi loads before CLAP's first prediction and absorbs part of that cost, so the first call
  only takes ~3s; for `clap-rollout-replay`/`clap-teleop`/`clap-eval`, which never load openpi,
  the very first call is genuinely cold and can take ~15s instead. Either way, it's a one-time
  cost — every call after the first lands at the steady-state numbers above.

- Full cross-embodiment modeling runs (`examples/slurm/*.slurm`) are sized for 8-GPU nodes
  (80GB-class GPUs, ~180-450GB system RAM) — post-training/adaptation runs on a single dataset
  need much less.
- **CUDA 12.8**-compatible driver — `torch`/`torchvision` are pinned to the
  `cu128` wheel index (see Installation below).
- Real training/eval against the full OXE mix needs the underlying video
  datasets on disk (not included — see [Dataset registry](docs/REFERENCE.md#dataset-registry)); the
  [Getting started](#-getting-started) walkthrough below needs none of that.

## 📦 Installation

### 1. Core package

This package is developed and tested with [`uv`](https://github.com/astral-sh/uv)
— recommended:

```bash
git clone <this-repo> clap && cd clap
VENV_HOME=${UV_ENV_DIR:-".venv"}
uv venv ${VENV_HOME}/clap --python 3.11   # or point --python at any interpreter >= 3.9
source ${VENV_HOME}/clap/bin/activate
uv pip install -e .
```

Or, in one step with `uv sync` (creates the venv and installs the exact locked versions from
`uv.lock` together — more reproducible, since `uv pip install -e .` above re-resolves versions
each time instead of pinning to the committed lockfile):

```bash
git clone <this-repo> clap && cd clap
VENV_HOME=${UV_ENV_DIR:-".venv"}
UV_PROJECT_ENVIRONMENT=${VENV_HOME}/clap uv sync --python 3.11
source ${VENV_HOME}/clap/bin/activate
```

### 2. Policy backends (optional)

Real-robot/policy-in-the-loop deployment (`clap-rollout-deploy`) additionally
needs a trained policy — openpi or MolmoAct2, matching `deploy_config.policy_type`.
Both are only imported inside `clap.rollout.policies.{openpi_policy,molmoact_policy}`
when you actually construct one; skip this step for training/eval/replay/teleop.

#### 🤖 openpi (pi0 / pi05 / pi0fast)

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git optional_dependencies/openpi
cd optional_dependencies/openpi
# Or, if you already cloned without --recurse-submodules:
#   git submodule update --init --recursive

# For separate server-based policy calls
#   for in-process (comment the next line)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Full details (other install methods): [Physical-Intelligence/openpi#installation](https://github.com/Physical-Intelligence/openpi#installation).
Checkpoints, and in-process vs. server tradeoffs, are covered under [Tips and Troubleshooting](#-tips-and-troubleshooting).

#### 🤖 MolmoAct2

```bash
git clone https://github.com/allenai/molmoact2.git optional_dependencies/molmoact2
cd optional_dependencies/molmoact2
# CLAP supports only server-based policy calls for now
uv sync
```

Full details (other install methods): [allenai/molmoact2#4-start-a-server](https://github.com/allenai/molmoact2#4-start-a-server).
Checkpoints and bimanual_yam setup are covered under [Tips and Troubleshooting](#-tips-and-troubleshooting).

Server warmup crashing with `predict_action() got an unexpected keyword argument 'action_mode'`?
See [Tips and Troubleshooting](#-tips-and-troubleshooting).

### 3. Environment

All cluster/site-specific paths are read from `CLAP_*` environment variables
(never hardcoded) — see `setup/env.example.sh` for the full list with explanations.
At minimum:

```bash
cp setupe/env.example.sh setup/env.sh
# edit env.sh: set CLAP_OXE_BASE_PATH, and any per-dataset overrides your data needs
source setup/env.sh
```

`CLAP_OXE_BASE_PATH` is the only strictly required variable (`clap.config.paths.PathConfig`
raises immediately if it's unset). Everything else has a generic default —
`dataset_meta_info/` (per-dataset `stat.json` normalization bounds) ships
pre-populated in this repo for every registered dataset, so most setups don't
need `CLAP_META_INFO_ROOT` at all.

## 🏁 Getting started

No dataset or account setup needed for this section — everything below runs
against `sample_data/oxe/` (~232MB, real full-length val episodes for `droid`/
`bridge`/`taco_play` and the two novel-embodiment adaptation targets
`bimanual_yam`/`g1_humanoid`, shipped in this repo) and downloads one
checkpoint from HF on first use. To get started, activate your virtual environment:
```bash
source ${UV_ENV_DIR:-.venv}/clap/bin/activate   # activate the venv created in Installation ▸ 1. Core package
```

### 📥 Download a checkpoint (optional)

Checkpoints are automatically downloaded on demand to `model_ckpt/<CKPT_NAME>/` — set `CKPT_NAME` to pick the default; see the [checkpoints table](#-checkpoints). To pre-fetch one manually instead:

```bash
hf download omni-CLAP/CLAP --include "cross_embodiment_oxe_ee/checkpoint-100000.pt" --local-dir model_ckpt
```

### 🔁 Replay demonstration trajectories

Roll a checkpoint out against a sample episode and write a side-by-side
GT/prediction video + PSNR/SSIM/LPIPS:

```bash
bash examples/getting_started/replay.sh                                 # droid, clap-curr
DATASET=bridge bash examples/getting_started/replay.sh                  # or: droid | bridge | taco_play
CKPT_NAME=clap-lam DATASET=bridge bash examples/getting_started/replay.sh  # or other models: clap-lam, clap-lang, clap-ee-droid, clap-curr-bridge, ...
CKPT_NAME=clap-ee DATASET=droid MAX_EPISODES=3 bash examples/getting_started/replay.sh  # multiple episodes

# Novel-embodiment adaptation targets (14-/26-dim joint-space actions, not 7-dim EE
# cartesian) -- pair with their matching adapt_* checkpoint; MAX_CHUNKS caps their
# much longer sample episodes (1000+ frames) for a quick demo
CKPT_NAME=adapt-yam DATASET=bimanual_yam MAX_EPISODES=3 MAX_CHUNKS=20 TRIM_STATIC=1 bash examples/getting_started/replay.sh
CKPT_NAME=adapt-g1 DATASET=g1_humanoid MAX_EPISODES=3 MAX_CHUNKS=20 bash examples/getting_started/replay.sh
```

See [Tips and Troubleshooting](#-tips-and-troubleshooting)
for what `--num-inference-steps` (25 here, vs. the real default of 50), `MAX_EPISODES`, and
`MAX_CHUNKS` (all used above) do.

Output: `eval_outputs/getting_started_replay/<CKPT_NAME>_<timestamp>/{video,info}/`.


### 🎮 Live Teleop for DROID, BRIDGE, TACO-PLAY, BIMANUAL YAM, G1 Humanoid

Interactively drive the world model's imagined next frame with your keyboard, seeded from a
sample episode (Ctrl-C to quit). `droid`/`bridge`/`taco_play` use a single-arm cartesian
scheme (`w`/`a`/`s`/`d`/`z`/`x` + roll/pitch/yaw + gripper); `bimanual_yam`/`g1_humanoid`
use a shared per-joint scheme targeting whichever arm/hand is active (`Tab` cycles it,
`Space` mirrors keypresses onto its left/right counterpart too). The live-preview browser
page (below) shows the actual key map and highlights each key as you press it, which is
easier to follow live than reading it here — see `clap.rollout.teleop_controls`'s module
docstring if you want the exact static reference instead:

```bash
bash examples/getting_started/teleop.sh                              # default DATASET=droid, CKPT_NAME=clap-curr
DATASET=bridge bash examples/getting_started/teleop.sh                # or: bridge | droid | taco_play
DATASET=bimanual_yam bash examples/getting_started/teleop.sh          # CKPT_NAME auto-set to adapt_bimanual_yam
DATASET=g1_humanoid bash examples/getting_started/teleop.sh           # CKPT_NAME auto-set to adapt_g1_humanoid
EPISODE=1002 DATASET=bridge bash examples/getting_started/teleop.sh   # other episodes: bridge -> 10|1002|1003, droid -> 2799|7099|9199, taco_play -> 1002|1003|1010
CKPT_NAME=clap-ee DATASET=droid bash examples/getting_started/teleop.sh  # or other checkpoints (clap-lang, clap-ee-droid, post-trained, ...); explicit CKPT_NAME always wins over the auto-default above
LIVE_VIEW_WS_PORT=7765 LIVE_VIEW_HTTP_PORT=7766 bash examples/getting_started/teleop.sh  # modify the websocket/html viewer ports
```

Output: `eval_outputs/getting_started_teleop/<CKPT_NAME>_<timestamp>/`. See
[Tips and Troubleshooting](#-tips-and-troubleshooting) for the live-preview viewer,
scripted (`KEYS=`) mode, `--num-inference-steps`, workspace bounds, g1_humanoid
control tips, and camera stacking order.


### 🚀 Deploy (policy-in-the-loop) on DROID and Bimanual YAM with Live Preview

Simulated closed-loop deployment: a real policy outputs actions and the world
model predicts the future each round with live preview. Needs MolmoAct2 or openpi
installed (see [Installation](#-installation)) and a policy server/checkpoint filled
into `examples/getting_started/deploy_config.yaml` first — see
[Tips and Troubleshooting](#-tips-and-troubleshooting) for other deploy-specific notes:

```bash
bash examples/getting_started/deploy.sh                                       # uses deploy_config.yaml's own policy_type/policy_server/policy_ckpt setting
POLICY_SERVER_OVERRIDE=127.0.0.1:8001 bash examples/getting_started/deploy.sh  # or a different policy server host/port
POLICY_TYPE=pi05 bash examples/getting_started/deploy.sh                      # or openpi/pi05 instead of the MolmoAct2 default
POLICY_TYPE=pi05 IN_PROCESS=1 bash examples/getting_started/deploy.sh         # or an in-process policy instead of a server
CKPT_NAME=clap-ee POLICY_SERVER_OVERRIDE=127.0.0.1:8001 bash examples/getting_started/deploy.sh  # or other checkpoints
LIVE_VIEW_WS_PORT=6765 LIVE_VIEW_HTTP_PORT=6766 bash examples/getting_started/deploy.sh  # modify the websocket/html viewer ports

bash examples/getting_started/deploy_yam.sh  # bimanual_yam instead of DROID -- joint-space, MolmoAct2 server mode only (see the bimanual_yam paragraph under Installation's MolmoAct2 section)
```

Output: `eval_outputs/getting_started_deploy/<CKPT_NAME>_<timestamp>/` (or
`eval_outputs/getting_started_deploy_yam/<CKPT_NAME>_<timestamp>/` for `deploy_yam.sh`).


## 🧩 Sample-efficient adaptation to new embodiments

CLAP establishes a paradigm for training high-fidelity single-embodiment
video world models via sample-efficient, few-shot adaptation of cross-embodiment models to target robot
platforms. 
This section provides a worked example using `bimanual_yam` (and `g1_humanoid` — see the notes inline).
This repo ships one train-split sample episode for both, alongside their existing val episodes
(`sample_data/oxe/{bimanual_yam,g1_humanoid}/{videos,annotation}/train/`), just enough to
exercise this whole pipeline end to end.

**1. Format your data** as `<oxe_base_path>/<dataset_name>/{videos,annotation}/<split>/...` —
`sample_data/oxe/bimanual_yam` is a concrete (tiny) reference for the expected layout: one
`videos/<split>/<episode_id>/<cam>.mp4` per camera, one `annotation/<split>/<episode_id>.json`
per episode (`state`, `texts`, ... — see [Dataset registry](docs/REFERENCE.md#dataset-registry)). `g1_humanoid`'s
annotations instead come from its own `clap-preprocess-g1` pipeline, which builds this same
layout from `unitreerobotics/*` HF datasets.

```bash
CLAP_OXE_BASE_PATH=$CLAP_OXE_BASE_PATH bash examples/preprocess/compute_g1_meta_info.sh  # only needed for g1_humanoid
```

**2. Compute action-normalization stats** (`dataset_meta_info/<name>/stat.json`, what
`BoundNormalizer` reads at train/eval time — **train split only**, so val stays unseen by
anything derived from training data, including the normalization bounds themselves):

```bash
clap-preprocess-oxe-meta --oxe-base-path sample_data/oxe --dataset-name bimanual_yam --full-state
# --full-state: bimanual_yam's 14-dim action is the raw `state` array, not state[:6]+gripper
# like cartesian ee7 datasets. g1_humanoid gets its stat.json from clap-preprocess-g1 (see step 1 instead)
```

**3. Point CLAP at your data** — two env vars (see `env.example.sh`):

```bash
export CLAP_OXE_BASE_PATH=sample_data/oxe    # your real dataset root, in practice
export CLAP_META_INFO_ROOT=dataset_meta_info # where step 2 wrote stat.json (this is the default -- only needed if you chose a different --meta-info-path)
```

**4. Launch adaptation training** (finetunes from the cross-embodiment curriculum checkpoint;
`configs/experiment/adapt_bimanual_yam.yaml` resets the action encoder since the input dim
changes 7 → 14; see [Tips and Troubleshooting](#-tips-and-troubleshooting) for how this
script picks up your `CLAP_*` env vars):

```bash
bash examples/adapt/adapt_bimanual_yam.sh
# g1_humanoid: bash examples/adapt/adapt_g1_humanoid.sh
```

**5. Build a test set** (episode selection cached under `clap/eval/test_sets_cache/`, so every
eval run pins to the same episodes):

```bash
clap-build-test-sets --names bimanual_yam_val --oxe-base-path "$CLAP_OXE_BASE_PATH" --no-lam-intersection
# --no-lam-intersection: episode selection normally also requires LAM latent-action data
# (ann_keys & lam_keys) -- skip that for datasets that never compute LAM at all, like
# bimanual_yam/g1_humanoid (joint-space-only), or it'll always select 0 episodes for them.
# g1_humanoid:
# clap-build-test-sets --names g1_humanoid_val --oxe-base-path "$CLAP_OXE_BASE_PATH" --no-lam-intersection
```

**6. Evaluate the adapted checkpoint**:

```bash
clap-eval --config configs/experiment/adapt_bimanual_yam.yaml --experiment adapt_bimanual_yam \
    --test-set bimanual_yam_val
# g1_humanoid:
# clap-eval --config configs/experiment/adapt_g1_humanoid.yaml --experiment adapt_g1_humanoid \
#     --test-set g1_humanoid_val
```

**Trying this locally without touching your repo's own `dataset_meta_info/bimanual_yam/stat.json`** — point every step at
a scratch dir instead of its default output location:

```bash
SCRATCH=$(mktemp -d)
clap-preprocess-oxe-meta --oxe-base-path sample_data/oxe --dataset-name bimanual_yam --full-state \
    --meta-info-path "$SCRATCH/dataset_meta_info"
export CLAP_OXE_BASE_PATH=sample_data/oxe
export CLAP_META_INFO_ROOT="$SCRATCH/dataset_meta_info"
clap-build-test-sets --names bimanual_yam_val --oxe-base-path "$CLAP_OXE_BASE_PATH" \
    --no-lam-intersection --out-dir "$SCRATCH/test_sets_cache"
clap-eval --config configs/experiment/adapt_bimanual_yam.yaml --experiment adapt_bimanual_yam \
    --test-set bimanual_yam_val --test-sets-cache-dir "$SCRATCH/test_sets_cache" \
    --max-chunks 1   # fast test -- 1 autoregressive chunk (config.model.num_frames predicted
                     # frames) per episode instead of a full rollout; omit for a full-duration eval
```


## 🗂️ Checkpoints

We provide a broad suite of cross-embodiment video model checkpoints at [omni-CLAP/CLAP](https://huggingface.co/omni-CLAP/CLAP/tree/main) — trained for 100K steps — and finetuned checkpoints derived from them for target embodiments.


### 🌐 Cross-embodiment models

The cross-embodiment models are trained on a mix of OXE datasets: (`fractal`/`fmb`/`bc_z`/`taco_play`/`furniture_bench`/`bridge`/`droid`, `egodex` too for `lam`). All `ee`-conditioned models use absolute actions, while the `lang`-conditioned model uses relative-to-anchor-frame language captions.

| Model name | Conditioning | Use case |
|---|---|---|
| [`clap-curr`](https://huggingface.co/omni-CLAP/CLAP/tree/main/cross_embodiment_oxe_curriculum_lam_ee) | `ee`, continued from a LAM-pretrained checkpoint | **Default** — getting-started default; base checkpoint for novel-embodiment adaptation |
| [`clap-ee`](https://huggingface.co/omni-CLAP/CLAP/tree/main/cross_embodiment_oxe_ee) | `ee` — 7-dim end-effector cartesian | Cross-embodiment world model, EE-only (no LAM pretraining stage) |
| [`clap-lam`](https://huggingface.co/omni-CLAP/CLAP/tree/main/cross_embodiment_oxe_lam_clap) | `lam` — 32-dim latent action | Curriculum LAM pretraining stage |
| [`clap-lang`](https://huggingface.co/omni-CLAP/CLAP/tree/main/cross_embodiment_oxe_language_relative) | `language`, relative-to-anchor-frame captions | Language-conditioned generation/eval |


### 🎯 Post-trained on a single robot platform

Each finetunes the matching cross-embodiment checkpoint above onto one dataset
alone (still `ee`-conditioned regardless of the base's own conditioning — see
`clap.eval.experiments`' `_POST_TRAIN_BASES` comment).

#### DROID

| Model name | Conditioning | Use case |
|---|---|---|
| [`clap-curr-droid`](https://huggingface.co/omni-CLAP/CLAP/tree/main/post_train_oxe_curriculum_lam_ee_droid) | `ee` — 7-dim EE cartesian | Post-trained on `droid`, from the `clap-curr` base |
| [`clap-ee-droid`](https://huggingface.co/omni-CLAP/CLAP/tree/main/post_train_oxe_ee_droid) | `ee` — 7-dim EE cartesian | Post-trained on `droid`, from the `clap-ee` base |
| [`clap-lam-droid`](https://huggingface.co/omni-CLAP/CLAP/tree/main/post_train_oxe_lam_clap_droid) | `ee` — 7-dim EE cartesian | Post-trained on `droid`, from the `clap-lam` base |
| [`clap-lang-droid`](https://huggingface.co/omni-CLAP/CLAP/tree/main/post_train_oxe_language_relative_droid) | `ee` — 7-dim EE cartesian | Post-trained on `droid`, from the `clap-lang` base |

#### Bridge

| Model name | Conditioning | Use case |
|---|---|---|
| [`clap-curr-bridge`](https://huggingface.co/omni-CLAP/CLAP/tree/main/post_train_oxe_curriculum_lam_ee_bridge) | `ee` — 7-dim EE cartesian | Post-trained on `bridge`, from the `clap-curr` base |
| [`clap-ee-bridge`](https://huggingface.co/omni-CLAP/CLAP/tree/main/post_train_oxe_ee_bridge) | `ee` — 7-dim EE cartesian | Post-trained on `bridge`, from the `clap-ee` base |
| [`clap-lam-bridge`](https://huggingface.co/omni-CLAP/CLAP/tree/main/post_train_oxe_lam_clap_bridge) | `ee` — 7-dim EE cartesian | Post-trained on `bridge`, from the `clap-lam` base |
| [`clap-lang-bridge`](https://huggingface.co/omni-CLAP/CLAP/tree/main/post_train_oxe_language_relative_bridge) | `ee` — 7-dim EE cartesian | Post-trained on `bridge`, from the `clap-lang` base |

### 🆕 Novel-embodiment adaptation

Finetune on data for a new robot embodiment, builds on physical priors from the cross-embodiment models (defaults to `clap-curr`)

| Model name | Conditioning | Use case |
|---|---|---|
| [`adapt-yam`](https://huggingface.co/omni-CLAP/CLAP/tree/main/adapt_bimanual_yam) | `ee`, 14-dim joint-space | Novel-embodiment adaptation target: `bimanual_yam` |
| [`adapt-g1`](https://huggingface.co/omni-CLAP/CLAP/tree/main/adapt_g1_humanoid) | `ee`, 26-dim joint-space | Novel-embodiment adaptation target: `g1_humanoid` |

## 📁 Project layout

```
src/clap/
  config/       dataclass configs (CLAPModelConfig, DataConfig, TrainingConfig, PathConfig, ...)
                + load_config() (Hydra-style YAML composition)
  data/         per-embodiment dataset registry + EE/LAM/Language dataset classes
  models/       CLAPModel (SVD-based UNet + action/text conditioning)
  training/     clap-train and its dataloader/checkpoint/validation helpers
  eval/         clap-eval and the checkpoint-comparison/test-set-building utilities
  rollout/      autoregressive replay, real-robot deployment, keyboard teleop
  preprocess/   per-dataset stat.json / G1-humanoid data prep
  utils/        shared rich-console logging setup

configs/        YAML configs consumed by load_config() (model/, data/, training/, experiment/)
examples/       runnable shell scripts wrapping the CLI entrypoints below (train/eval/rollout/adapt/slurm)
dataset_meta_info/   shipped stat.json per dataset (normalization bounds)
tests/          unit/ (no data needed) + integration/ (needs real OXE data / a GPU, auto-skipped otherwise)
```

## 📚 Reference

Full configuration reference, the complete entrypoint/CLI documentation, the dataset
registry, and testing instructions live in [`docs/REFERENCE.md`](docs/REFERENCE.md) —
split out to keep this README focused on getting up and running.


## 🛠️ Tips and Troubleshooting

<details>
<summary><b>Getting-started demo knobs (<code>--num-inference-steps</code>, <code>MAX_EPISODES</code>, <code>MAX_CHUNKS</code>, <code>KEYS</code>)</b></summary>

Every getting-started script (`replay.sh`/`teleop.sh`/`deploy.sh`/`deploy_yam.sh`) runs with
`--num-inference-steps 25` (vs. the real default of 50 — see each entrypoint's `--help`) to
keep the demo quick; drop the flag in the script, or pass a higher value, for full-quality
output at the cost of slower inference. `replay.sh`'s `MAX_EPISODES=<n>` rolls out `<n>`
sample episodes instead of the default 1, writing one video/metrics set per episode.
`MAX_CHUNKS=<n>` (replay/teleop) caps how many autoregressive chunks each episode runs for
— useful for the much longer bimanual_yam/g1_humanoid sample episodes (1000+ frames vs.
tens of frames for droid/bridge/taco_play). `teleop.sh`'s `KEYS=<sequence>` (e.g.
`KEYS=wasdcv`) replaces the default live interactive session with a fixed non-interactive
scripted key sequence — useful for a demo/CI run.

</details>

<details>
<summary><b>Teleop tips: workspace bounds & g1_humanoid controls</b></summary>

If teleop's tracked cartesian pose runs out of a sensible range mid-session (droid/bridge/
taco_play only), `_X_RANGE`/`_Y_RANGE`/`_Z_RANGE` in `clap.rollout.teleop_controls` can be
widened directly in that file — there's no CLI flag for it yet.

g1_humanoid's 26-dim joint-space control is higher-dimensional than the other targets and
can feel subtle at first — `e` (an arm dim) tends to produce more visually obvious motion
and similarly for `q` (a hand target's first dim, the thumb); `Tab` cycles which arm/hand is active. Its
`adapt_*` checkpoint was also finetuned on comparatively limited data despite a harder
prediction landscape than the other embodiments, so fidelity may be lower than what you see
on DROID/bridge.

</details>

<details>
<summary><b>Deploy tips: live viewer, <code>policy_type</code>, <code>history_idx</code>, bimanual_yam's <code>policy_skip_step_schedule</code></b></summary>

`policy_type` selects the policy: `pi05`/`pi0`/`pi0fast` (openpi, DROID only) or `molmoact2`
(DROID or bimanual_yam). The policy needs a language instruction per episode — the shipped
sample data already provides one, so there's nothing to set up there. Like Teleop, this also
starts a live-preview server by default — it prints a `http://localhost:<port>/deploy_viewer.html`
URL to open in a browser, showing each round's imagined prediction as it's generated; set
`NO_LIVE_VIEW=1` to skip it, or override `LIVE_VIEW_WS_PORT`/`LIVE_VIEW_HTTP_PORT` (see the
examples above).

`history_idx` in `deploy_config.yaml`/`deploy_config_yam.yaml` is set to sparse offsets by
default (`[0, 0, -48, -32, -24, -16]`), giving the world model longer temporal context each
round, which is useful if predictions degrade under fast robot motion; comment it out to fall back to the last `num_history` contiguous frames instead.

bimanual_yam's MolmoAct2 policy can produce robot motion with higher accelerations than
DROID's — `policy_skip_step_schedule` in `deploy_config_yam.yaml` (`[4, 2, 2, 2, 2, 1]` by
default) varies how many raw policy timesteps are skipped between world-model-conditioning
frames per round, covering more ground early on and settling to a finer step later; see
`RolloutDeployConfig.policy_skip_step_schedule`'s docstring for the exact semantics.

DROID and bimanual_yam only, for now — `clap.rollout.deploy` doesn't yet support
`g1_humanoid`'s 4-camera/26-dim-joint-space policy interface (no MolmoAct2/openpi server
exists for it either). Pointing `deploy_config.yaml` at `dataset_name: g1_humanoid` will not
raise an error; it'll just silently misbehave (see `PolicyInTheLoopAgent.__init__`'s
`is_joint_space` note in `deploy.py`).

</details>

<details>
<summary><b>Adaptation training/finetuning: env vars & <code>bash_scripts/setup.bash</code></b></summary>

`examples/adapt/*.sh` (used in [Sample-efficient adaptation to new embodiments](#-sample-efficient-adaptation-to-new-embodiments))
rely on whatever `CLAP_*` variables are already exported in your shell — they only source
`examples/_common.sh` (GPU count/port setup), not any cluster-specific env file. The
SLURM-submitted variants (`examples/slurm/*.slurm`) instead auto-source
`bash_scripts/setup.bash` for you; if you maintain your own cluster config there, source it
before running the plain `examples/adapt/*.sh` scripts too. When finetuning, ensure that the  CLAP model checkpoint is downloaded before training begins; if absent, the new model will train from scratch.

</details>

<details>
<summary><b>Camera stacking order (<code>four_view</code> / <code>three_view</code> / <code>two_view</code>)</b></summary>

Every embodiment's camera frames get resized to 192x320 and stacked vertically into one
fixed-shape input (`clap.data.camera_stacking`), so the world model always sees the same input
shape regardless of how many physical cameras an embodiment has:

| Stacking mode | Slots (top → bottom) | Datasets |
|---|---|---|
| `four_view` | 4 distinct cameras: right_high → left_high → right_wrist → left_wrist | `g1_humanoid` |
| `three_view` | 3 distinct cameras: right → left → wrist | `droid`, `fmb`, `bimanual_yam` |
| `two_view` | 2 distinct cameras: right (scene) → left (wrist) — wrist duplicated into the middle+bottom slots to fill out 3 | `furniture_bench`, `austin_sailor`, `berkeley_autolab_ur5`, `stanford_hydra`, `utaustin_mutex` |
| single-camera | 1 camera, tiled into all 3 slots | `bridge`, `fractal`, `bc_z`, `taco_play`, `egodex` |

**bimanual_yam deploy-time camera wire keys**: `clap.rollout.deploy` sends bimanual_yam's 3 stacked
camera crops to MolmoAct2's `host_server_yam.py`, whose own HTTP wire keys don't match these slot
names directly — `EmbodimentConfig`'s three_view slots map to the server's wire keys as
`right → top_cam`, `left → left_cam`, `wrist → right_cam`.

</details>

<details>
<summary><b>What's in the 7-dim EE-cartesian action (<code>[x, y, z, roll, pitch, yaw, gripper]</code>)</b></summary>

`ee`-conditioned datasets (droid/bridge/taco_play) represent each action as 7 floats:
`[x, y, z, roll, pitch, yaw, gripper]`. The first 6 dims (`state[:, :6]`) are the
end-effector's cartesian pose — xyz position plus roll/pitch/yaw orientation — and the
7th is the continuous gripper state, concatenated on separately (see
`EEDataset._load_action` in [`src/clap/data/ee.py`](src/clap/data/ee.py)). This is the
"ee7" convention (`action_mode="ee7"`, the default for these embodiments); it's distinct
from the raw joint-space actions (`joint14`/`joint26`) used by `bimanual_yam`/`g1_humanoid`,
which have no fixed cartesian meaning per dim. All 7 dims are normalized against the
dataset's `stat.json` p01/p99 bounds before being fed to the model. See also the **EE**
entry in the [Glossary](#-glossary).

</details>

<details>
<summary><b>MolmoAct2 server error: <code>predict_action() got an unexpected keyword argument 'action_mode'</code></b></summary>

`examples/droid/host_server_droid.py`'s `predict_action(...)` call in the `molmoact2` repo (cloned
per [Installation](#-installation), external to `clap`) passes `action_mode="continuous"`, but the
currently-published `allenai/MolmoAct2-DROID` checkpoint's remote code expects
`inference_action_mode="continuous"` instead (same kwarg `host_server_yam.py` already uses
correctly) — this fails at server warmup with that `TypeError`. It's a version-skew bug between
the `molmoact2` GitHub repo and its HF Hub checkpoint's remote code, not a `clap` issue, so it
won't be fixed by anything in this repo — patch your own clone:

```bash
# in the molmoact2 repo
sed -i 's/action_mode="continuous"/inference_action_mode="continuous"/' examples/droid/host_server_droid.py
```

If this resurfaces (e.g. after a fresh clone/checkpoint re-download), check the actually-cached
signature directly rather than trusting this note — `snapshot_download` there isn't
revision-pinned, so upstream can rename the kwarg again:
`~/.cache/huggingface/hub/models--allenai--MolmoAct2-DROID/snapshots/<rev>/modeling_molmoact2.py`.

</details>

<details>
<summary><b>openpi: checkpoints & in-process vs. server</b></summary>

**Checkpoints** are hosted on GCS under `gs://openpi-assets/checkpoints/<name>`
— e.g. the DROID pi05 checkpoint is `gs://openpi-assets/checkpoints/pi05_droid`.
Point `deploy_config.yaml`'s `policy_ckpt` directly at a `gs://...` path (openpi
downloads/caches it on first use) to use any published checkpoint, or a local
path if you've already fetched one yourself. `policy_type` selects which
training config it loads under (`pi05` / `pi0` / `pi0fast`, see
`OpenPIPolicy`'s docstring) — set it to match whichever checkpoint you pick.
`gs://...` downloads cache under `~/.cache/openpi` by default; set
`OPENPI_DATA_HOME=/path/to/cache` (before running `clap-rollout-deploy`) to
put that cache somewhere else.

**Server mode is recommended** (see Installation ▸ openpi) — start the server from the openpi repo:

```bash
cd optional_dependencies/openpi
OPENPI_POLICY_BASE_DIR=${OPENPI_POLICY_BASE_DIR:-~/.cache/openpi}
# an example with pi05
uv run scripts/serve_policy.py policy:checkpoint --port=8000 \
    --policy.config=pi05_droid --policy.dir=${OPENPI_POLICY_BASE_DIR}/pi05_droid
```

then set `deploy_config.yaml`'s `policy_server: "host:port"` instead of `policy_ckpt`.
`clap-rollout-deploy --policy-server host:port` / `--in-process` (see
[`clap-rollout-deploy`](docs/REFERENCE.md#clap-rollout-deploy)) override whichever the YAML sets, without
editing it — `examples/getting_started/deploy.sh` exposes the same override as
`POLICY_SERVER_OVERRIDE=host:port` / `IN_PROCESS=1`.

</details>

<details>
<summary><b>MolmoAct2: checkpoints & starting servers</b></summary>

**Checkpoints** (~22GB each) download from HF, from inside the molmoact2 repo. To put the
HF cache on a different disk, set `HF_HOME=/path/to/cache` before both the download below
*and* before starting the server (it re-resolves the checkpoint from the same cache at
load time).

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run hf download allenai/MolmoAct2-DROID          # for the DROID server
uv run hf download allenai/MolmoAct2-BimanualYAM    # for the YAM server
```

Then start the server (from the molmoact2 repo):

```bash
uv run python examples/droid/host_server_droid.py --host 0.0.0.0 --port 8000 --dtype bfloat16
# or, for bimanual YAM:
uv run python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202 --dtype bfloat16
```

**bimanual_yam** (joint-space, MolmoAct2 server mode only — no openpi checkpoint exists for it):
`clap.rollout.deploy` skips forward kinematics entirely and conditions the world model on the raw
predicted 14-dim dual-arm joint+gripper state directly, and sends 3 camera crops to
`host_server_yam.py` — see the **Camera stacking order** note below for the exact slot/wire-key
mapping. `examples/getting_started/deploy_config_yam.yaml`/`deploy_yam.sh` has a runnable example
against this repo's shipped sample bimanual_yam episodes.

</details>

## 🙏 Acknowledgment

CLAP builds on and integrates with several open-source projects:

- [**Stable Video Diffusion**](https://github.com/Stability-AI/generative-models) (Stability AI) — the pretrained video diffusion backbone `CLAPModel`'s UNet is built on.
- [**Ctrl-World**](https://github.com/Robert-gyj/Ctrl-World) — state-of-the-art action-conditioned video world model for the DROID robot platform.
- [**openpi**](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence) — policy backend supported for policy-in-the-loop deployment (π₀/π₀.₅/π₀-FAST).
- [**MolmoAct2**](https://github.com/allenai/molmoact2) (Allen Institute for AI) — policy backend supported for policy-in-the-loop deployment.

We thank the authors of these projects for making their code and models available.

(More coming soon)

## 📄 BibTeX
Coming soon


## 📖 Glossary

<details>
<summary><b>EE, LAM, OXE, DROID, and other terms used throughout</b></summary>

| Term | Meaning |
|---|---|
| **EE** | End-effector — the robot's gripper/hand; `ee`-conditioning uses its 7-dim cartesian pose (position + orientation) + gripper, as opposed to raw joint angles |
| **LAM** | Latent Action Model — a learned, unsupervised action representation extracted directly from video (not a physical quantity), letting CLAP pretrain on video with no paired action labels, including egocentric human video |
| **FK** | Forward kinematics — computing a robot's end-effector pose from its joint angles; used for DROID's cartesian-conditioned world model, skipped entirely for joint-space embodiments (bimanual_yam/g1_humanoid) |
| **OXE** | Open X-Embodiment — the multi-robot, multi-dataset collection CLAP's cross-embodiment training draws from (`fractal`/`fmb`/`bc_z`/`taco_play`/`furniture_bench`/`bridge`/`droid`, `egodex` for `lam`) |
| **DROID** | A large-scale real-world Franka-arm manipulation dataset/platform; one of CLAP's core cross-embodiment/post-training targets |
| **Franka** | The single-arm robot platform DROID's data was collected on |
| **bimanual YAM** / `bimanual_yam` | A dual-arm robot platform; one of CLAP's novel-embodiment adaptation targets (14-dim joint-space action) |
| **G1** / `g1_humanoid` | Unitree G1, a bipedal humanoid robot (2 arms + 2 hands); the other novel-embodiment adaptation target (26-dim joint-space action) |
| **SVD** | Stable Video Diffusion — the pretrained video diffusion backbone `CLAPModel`'s UNet is built on |
| **FVD / FID** | Fréchet Video/Inception Distance — distributional video/image-quality metrics `clap-eval` reports alongside PSNR/SSIM/LPIPS |
| **PSNR / SSIM / LPIPS** | Per-frame image-similarity metrics (pixel error, structural similarity, learned perceptual similarity) `clap-eval`/`clap-rollout-replay` report against ground truth |
| **policy-in-the-loop** | A deployment mode where a real trained policy (openpi or MolmoAct2), not ground-truth replay, drives what the world model conditions on each round — see [`clap-rollout-deploy`](docs/REFERENCE.md#clap-rollout-deploy) |

</details>

## ⚖️ License

MIT — see [LICENSE](LICENSE).
