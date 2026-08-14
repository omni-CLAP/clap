# CLAP Reference: Configuration, Entrypoints & Dataset Registry

[← Back to main README](../README.md)

## Configuration

A training run is fully described by a `TrainingRunConfig` (`model:`/`data:`/`training:`
sections), loaded from a `configs/experiment/*.yaml` file via
`clap.config.load_config()`. Each experiment file composes shared defaults
Hydra-style, then overrides just what's experiment-specific:

```yaml
# configs/experiment/cross_embodiment_oxe_ee.yaml
defaults:
  - model: base       # configs/model/base.yaml
  - data: ee           # configs/data/ee.yaml
  - training: default  # configs/training/default.yaml

model:
  conditioning: ee
  action_dim: 7

training:
  tag: cross_embodiment_oxe_ee
  output_dir: model_ckpt/cross_embodiment_oxe_ee
```

`clap-train --override key.path=value` applies dotted-key overrides on top of
everything (highest priority) — e.g. `--override training.learning_rate=5e-6`.

## Entrypoints

Every entrypoint below is installed as a console script by `pip install -e .`
(see `[project.scripts]` in `pyproject.toml`). `examples/*.sh` wraps the most
common ones with sensible defaults — start there if you just want a runnable
command.

| Command | Purpose |
|---|---|
| [`clap-train`](#clap-train) | Train or resume a checkpoint |
| [`clap-eval`](#clap-eval) | Full checkpoint evaluation: rollout + PSNR/SSIM/LPIPS/FVD/FID |
| [`clap-rollout-replay`](#clap-rollout-replay) | Lower-level offline autoregressive replay (no FVD/FID, no experiment registry) |
| [`clap-rollout-deploy`](#clap-rollout-deploy) | Real-robot (or simulated) policy-in-the-loop deployment |
| [`clap-teleop`](#clap-teleop) | Keyboard-driven interactive world-model session |
| [`clap-preprocess-oxe-meta`](#clap-preprocess-oxe-meta) | Build a dataset's `stat.json` normalization bounds |
| [`clap-preprocess-g1`](#clap-preprocess-g1) | Download + convert the G1-humanoid HF datasets into OXE layout |
| [`clap-build-test-sets`](#clap-build-test-sets) | Build/refresh the cached episode-ID test sets `clap-eval --test-set` reads |
| [`clap-eval-aggregate`](#clap-eval-aggregate) | Aggregate one experiment's per-checkpoint `clap-eval` results into a table + plots |
| [`clap-eval-compare`](#clap-eval-compare) | Cross-experiment comparison from multiple `aggregate.json` files |
| [`clap-eval-capped-chunk-metrics`](#clap-eval-capped-chunk-metrics) | Re-compare runs on a shared, stricter rollout-length cap |
| [`clap-eval-check-episode-sets`](#clap-eval-check-episode-sets) | Verify different experiments evaluated the exact same episode IDs |

### `clap-train`

Train from scratch, resume, or post-train/adapt a checkpoint.

```bash
clap-train --config configs/experiment/cross_embodiment_oxe_ee.yaml
clap-train --config configs/experiment/baseline_droid.yaml \
    --override training.learning_rate=5e-6 training.max_train_steps=50000
```

| Flag | Description |
|---|---|
| `--config` (required) | `TrainingRunConfig` YAML path |
| `--override` | Zero or more `dotted.key=value` overrides, applied last |

Resume behavior, `finetune_ckpt`/adapter/action-encoder-reset options, and
checkpoint cadence all live in `training:`/`model:` config fields — see
`clap.config.training.TrainingConfig` and the `examples/posttrain/`,
`examples/adapt/` scripts for the common patterns (post-training an
LAM/language-pretrained checkpoint on droid/bridge alone; freezing the UNet
and training only an action adapter for a new embodiment).

### `clap-eval`

The full evaluation pipeline: rollout every selected episode, write GT/prediction
videos + per-episode metrics, then add per-dataset FVD/FID and a cross-dataset
aggregate. Supports `--resume` to cheaply complete a killed job by reloading
already-written episodes instead of re-running them.

```bash
clap-eval --config configs/experiment/cross_embodiment_oxe_ee.yaml \
    --experiment cross_embodiment_oxe_ee --ckpt last --test-set oxe_mix_100
```

(`examples/rollout/rollout_eval.sh` wraps this: `EXPERIMENT=... TEST_SET=... bash examples/rollout/rollout_eval.sh`.)

| Flag | Description |
|---|---|
| `--config` (required) | `TrainingRunConfig` YAML the checkpoint was trained with |
| `--experiment` (required) | Key in `clap.eval.experiments.EXPERIMENTS` (selects `family`/`ckpts_root`/defaults) |
| `--ckpt` | `"last"` (default), a step number, or a path |
| `--datasets` | Defaults to the experiment's `default_datasets` |
| `--test-set` | Name of a cached test set (see `clap-build-test-sets`); overrides `--datasets` |
| `--test-sets-cache-dir` | Where `--test-set` is read from. Default: `clap/eval/test_sets_cache/` (the committed one) — point at a `clap-build-test-sets --out-dir` scratch dir to read a test set built elsewhere instead |
| `--split` | `val` (default) or `train` |
| `--save-dir` | Default: `eval_outputs/<experiment>/<timestamp>` |
| `--max-episodes-per-dataset` | Cap per dataset (0 = no cap) |
| `--no-fvd` / `--no-fid` | Skip those metrics |
| `--resume` | Reload already-written episodes instead of re-running |
| `--num-inference-steps`, `--guidance-scale`, `--max-chunks`, `--decode-chunk-size`, `--gt-cond`, `--history-idx` | Rollout knobs, forwarded to `CLAPRolloutAgent` |
| `--trim-static-prefix`, `--skip-first-n-frames` | Per-dataset defaults (`clap.eval.dataset_specs`), overridable here |

**Running eval across every model** (`examples/slurm/run_all_eval.sh`): submits one
`clap-eval` job per (experiment, test-set) combination — baselines, all cross-embodiment
variants (x3 test sets, +x4 held-out if `CLAP_OXE_HELD_OUT_PATH` is set), post-trained
DROID/Bridge models, and both adaptation targets — pulling the experiment list from
`clap.eval.experiments` itself so it can't drift out of sync.

```bash
bash examples/slurm/run_all_eval.sh              # submit one sbatch job per combination
DRY_RUN=1 bash examples/slurm/run_all_eval.sh     # print the commands only, submit nothing
RUN_LOCAL=1 bash examples/slurm/run_all_eval.sh   # run each eval_job.sh directly via bash,
                                                   # sequentially, no slurm -- for a machine
                                                   # without slurm, or debugging on an
                                                   # already-allocated GPU
MAX_EPISODES_PER_DATASET=2 MAX_CHUNKS=1 bash examples/slurm/run_all_eval.sh  # fast test
                                                   # across every model
```

| Env var | Description |
|---|---|
| `CKPT_ITER` | Checkpoint iteration (default: `last`) |
| `EVAL_OUTPUTS_ROOT` | Output root (default: `PROJECT_DIR/eval_outputs`) |
| `NUM_INFERENCE_STEPS` | Default `50` |
| `MAX_CHUNKS` | Default `0` = unlimited |
| `MAX_EPISODES_PER_DATASET` | Default `0` = unlimited (caps episodes even when `--test-set` pins a specific list) |
| `NO_FVD` / `NO_FID` | Set to `1` to skip that metric |
| `DATASETS` | Space-separated dataset filter, e.g. `"droid"` (default: all) |
| `SKIP_DONE` | Set to `0` to re-run finished evals (default: `1` — skips any `(experiment, test-set)` whose `per_view_summary.json` already exists) |
| `TIME_LIMIT` | Slurm time limit (default: `3:00:00`); ignored under `RUN_LOCAL=1` |
| `DRY_RUN` | Print commands only, submit/run nothing |
| `RUN_LOCAL` | Run via plain `bash` instead of `sbatch` |

Requires `CLAP_OXE_BASE_PATH` (and `CLAP_OXE_HELD_OUT_PATH` for the held-out section, skipped
with a warning if unset). Each `sbatch`/local job is `examples/slurm/eval_job.sh`, which can
also be run directly for a single `(experiment, test-set)` pair:
`EXPERIMENT=cross_embodiment_oxe_ee TEST_SET=oxe_mix_100 sbatch examples/slurm/eval_job.sh`.

### `clap-rollout-replay`

The lower-level replay primitive `clap-eval` builds on: same autoregressive
rollout + PSNR/SSIM/LPIPS, but no FVD/FID, no experiment registry lookup, and
you pass `--ckpt`/`--family`/`--datasets` directly. Useful for a quick
test or when you don't want the full experiment-registry machinery.

```bash
clap-rollout-replay --config configs/experiment/cross_embodiment_oxe_ee.yaml \
    --ckpt model_ckpt/cross_embodiment_oxe_ee/last.pt \
    --family ee --datasets droid --max-episodes-per-dataset 1 \
    --num-inference-steps 10 --max-chunks 1
```

| Flag | Description |
|---|---|
| `--config` (required) | `TrainingRunConfig`-shaped YAML (`model:`/`data:` sections only) |
| `--ckpt` (required) | Checkpoint path |
| `--family` (required) | `ee` / `lam` / `language` — selects the episode loader |
| `--datasets` (required) | One or more dataset names |
| `--split` | `val` (default) |
| `--save-dir`, `--max-episodes-per-dataset`, `--num-inference-steps`, `--guidance-scale`, `--max-chunks`, `--gt-cond` | Same meaning as `clap-eval`'s equivalents |

### `clap-rollout-deploy`

Real-robot policy-in-the-loop deployment (Franka/DROID, or bimanual YAM): an openpi
or MolmoAct2 policy predicts an action chunk each round; the world model
renders an "imagined" preview conditioned on the chosen trajectory (FK'd cartesian
for DROID, raw joint state for bimanual_yam — see `deploy.py`'s module docstring).
Simulation mode by default (no live robot) — override `DeploySession.get_observation`/`execute_action`
for a real connection.

```bash
clap-rollout-deploy --config configs/experiment/cross_embodiment_oxe_ee.yaml \
    --deploy-config path/to/deploy_config.yaml --ckpt model_ckpt/.../last.pt
```

| Flag | Description |
|---|---|
| `--config` (required) | `TrainingRunConfig` YAML the checkpoint was trained with |
| `--deploy-config` (required) | `RolloutDeployConfig`-shaped YAML (`policy_type`, `task_name`, `episode_ids`, ...) |
| `--ckpt` (required) | `CLAPModel` checkpoint path |
| `--family` | Default `ee` |
| `--save-dir` | Default `deploy_outputs` |
| `--policy-type` | Overrides `deploy_config.policy_type` (`pi05`\|`pi0`\|`pi0fast`\|`molmoact2`) regardless of what the YAML sets |
| `--policy-server host:port` | Overrides `deploy_config.policy_server`, forcing server mode against that host regardless of what the YAML sets |
| `--in-process` | Forces in-process mode (clears `deploy_config.policy_server`), using `deploy_config.policy_ckpt` regardless of what the YAML sets. Mutually exclusive with `--policy-server` |

Omitting `--policy-type`/`--policy-server`/`--in-process` leaves `deploy_config.policy_type`/`policy_server`/`policy_ckpt` as the YAML sets them (see the main README's [Tips and Troubleshooting ▸ In-process vs. server](../README.md#-tips-and-troubleshooting) note).

Requires the `openpi` extra or a `lerobot`-with-MolmoAct2 install, matching
`deploy_config.policy_type`. See `examples/rollout/deploy_policy_in_the_loop.sh`.

### `clap-teleop`

Interactive (or scripted) keyboard teleop against the world model: each
keypress nudges the tracked pose and asks the model to imagine the next
frame — useful for qualitatively probing a checkpoint's action-following
behavior without a robot.

```bash
# Live: reads keys from the terminal one at a time (no Enter needed), Ctrl-C to quit.
clap-teleop --config configs/experiment/cross_embodiment_oxe_ee.yaml \
    --ckpt model_ckpt/.../last.pt --dataset droid --episode 10099

# Scripted: replays a fixed key sequence non-interactively (e.g. for a demo/CI run).
clap-teleop --config configs/experiment/cross_embodiment_oxe_ee.yaml \
    --ckpt model_ckpt/.../last.pt --dataset droid --episode 10099 --keys wwaaz
```

| Flag | Description |
|---|---|
| `--config`, `--ckpt` (required) | Same as above |
| `--family` | Default `ee` |
| `--dataset` (required) | Dataset to seed the session from |
| `--episode` (required) | Episode id/rel to seed from |
| `--keys` | Scripted key sequence, e.g. `"wwaaz"`. Omit for a live interactive session |
| `--save-dir` | Default `teleop_outputs` |

Key vocabulary (`clap.rollout.teleop_controls.KEY_HELP`): `w`/`a`/`s`/`d` =
forward/left/backward/right, `z`/`x` = up/down, `c`/`v` = close/open gripper.
See `examples/getting_started/teleop.sh` for a ready-to-run demo against this
package's shipped sample data.

### `clap-preprocess-oxe-meta`

Computes p01/p99 action-normalization bounds for one dataset and writes
`<meta-info-path>/<dataset-name>/stat.json` — run once per dataset before
training on it (already done for every shipped `dataset_meta_info/` entry;
re-run only if you're adding a new dataset or a fresh data revision).

```bash
clap-preprocess-oxe-meta --oxe-base-path $CLAP_OXE_BASE_PATH --dataset-name bridge
```

| Flag | Description |
|---|---|
| `--oxe-base-path` (required) | Root of OXE mp4 datasets |
| `--dataset-name` (required) | e.g. `bridge`, `fractal`, `droid` |
| `--meta-info-path` | Default `dataset_meta_info` |
| `--n-workers` | Default 32 |
| `--full-state` | Use the raw `state` array as-is (joint-space embodiments) instead of the ee7 (state[:6]+gripper) convention |

### `clap-preprocess-g1`

Downloads the `unitreerobotics/*` HuggingFace G1-humanoid datasets and
converts them into the same OXE mp4/annotation layout every other dataset
uses, so `g1_humanoid` can be trained/evaluated through the normal `EEDataset` path.

```bash
clap-preprocess-g1 --out-base $CLAP_OXE_BASE_PATH
```

| Flag | Description |
|---|---|
| `--out-base` (required) | Root of the OXE-format dataset tree (`CLAP_OXE_BASE_PATH`) |
| `--meta-out` | Default `dataset_meta_info` |
| `--hf-cache-dir` | Default `$HF_HOME` |
| `--datasets` | `unitreerobotics/<name>` HF repo names to process |
| `--dataset-index` | Process only this index into `--datasets` (for SLURM array jobs) |
| `--val-episodes` | Last N episodes per sub-dataset held out for val (default 13) |
| `--out-h`, `--out-w`, `--out-fps` | Output video dimensions/framerate |

### `clap-build-test-sets`

Builds/refreshes the cached episode-ID snapshots `clap-eval --test-set` reads
(`clap/eval/test_sets_cache/<name>.json`) — the same episodes get evaluated
across every experiment, and (with `strict_eligibility`) only episodes long
enough for every conditioning family (EE/LAM/language) are included.

```bash
clap-build-test-sets --oxe-base-path $CLAP_OXE_BASE_PATH
```

| Flag | Description |
|---|---|
| `--names` | Test sets to build (default: all in `clap.eval.test_sets`) |
| `--oxe-base-path` | Default from `$CLAP_OXE_BASE_PATH` |
| `--oxe-lam-subdir` | Default from `$CLAP_OXE_LAM_SUBDIR` |
| `--out-dir` | Default `clap/eval/test_sets_cache/` |
| `--no-lam-intersection` | Select from EE-state annotations alone, skipping the LAM latent-action intersection — needed for datasets that never compute LAM (e.g. `bimanual_yam`/`g1_humanoid`), which otherwise always select 0 episodes |

### `clap-eval-aggregate`

Reads `<results-root>/iter_*/{aggregate,per_view_summary}.json` (the standard
`clap-eval` output layout across multiple checkpoints of one experiment) and
produces a table + metric-vs-training-step plots.

```bash
clap-eval-aggregate --results-root eval_outputs/cross_embodiment_oxe_ee/oxe_mix_100
```

| Flag | Description |
|---|---|
| `--results-root` (required) | Directory containing `iter_*/` subdirs |
| `--out-dir` | Default `<results-root>/_aggregate` |
| `--datasets` | Restrict to these datasets (default: all found) |

### `clap-eval-compare`

Cross-experiment comparison from one or more `aggregate.json` files, driven
by a JSON config (see the module docstring in `clap.eval.compare` for the
exact schema).

```bash
clap-eval-compare --config my_comparison.json
```

### `clap-eval-capped-chunk-metrics`

Longer episodes accumulate more autoregressive drift, so a run with a looser
`max_chunks` can look worse purely from evaluating longer rollouts. This
recomputes aggregate metrics for multiple runs using only the shared episode
intersection under one common chunk cap, for an apples-to-apples comparison.

```bash
clap-eval-capped-chunk-metrics --run-dirs eval_outputs/run_a eval_outputs/run_b \
    --output-dir eval_outputs/_capped_compare --max-chunks 10
```

### `clap-eval-check-episode-sets`

Verifies that different experiments' `clap-eval` runs actually evaluated the
exact same episode IDs per test set (a prerequisite for a fair metrics
comparison across experiments).

```bash
clap-eval-check-episode-sets --experiments cross_embodiment_oxe_ee:iter_last cross_embodiment_oxe_lam_clap:iter_last
```

| Flag | Description |
|---|---|
| `--experiments` (required) | One or more `exp:iter` pairs |
| `--eval-outputs` | Default `eval_outputs` |
| `--test-sets` | Default: the standard test-set list |
| `--missing-sample` | How many missing/extra ids to print per experiment on divergence (default 5) |

## Dataset registry

Every dataset/embodiment `clap` knows about is one `EmbodimentConfig` entry
in `clap.data.oxe_catalog.OXE_CATALOG` — camera layout (`stacking_mode`),
action representation (`action_mode`), and annotation directory name, keyed
by name:

`fractal`, `fmb`, `bc_z`, `taco_play`, `furniture_bench`, `bridge`, `droid`,
`egodex` (LAM-only, egocentric human video), `austin_sailor`,
`berkeley_autolab_ur5`, `stanford_hydra`, `utaustin_mutex` (held-out
generalization targets), `bimanual_yam`, `g1_humanoid` (novel-embodiment
adaptation targets).

Adding a new dataset means adding one entry here plus a `stat.json` (via
`clap-preprocess-oxe-meta`) — no new dataset class needed unless its data
layout genuinely differs from the OXE annotation-JSON + per-camera-mp4
convention.

Per-dataset annotation directory name defaults to `annotation`, overridable
via `CLAP_<NAME>_ANNOTATION_SUBDIR` (`<NAME>` = dataset name, upper-cased).
The pre-encoded SVD video-latent subdir (shared across all datasets) defaults
to `latent_videos_svd`, overridable via `CLAP_LATENT_VIDEO_SUBDIR` — e.g. to
point at a copy staged under a different name on faster local/scratch
storage. See `setup/env.example.sh` for both.

## Testing

```bash
pytest tests/unit          # no data/GPU needed
pytest tests/integration   # needs CLAP_OXE_BASE_PATH set (and a GPU for the training test); auto-skips otherwise
```