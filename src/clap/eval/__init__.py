"""Checkpoint evaluation: rollout + PSNR/SSIM/LPIPS/FVD/FID, experiment registry, test sets."""

from clap.eval.dataset_specs import DatasetSpec, get_spec
from clap.eval.evaluate import evaluate
from clap.eval.experiments import EXPERIMENTS, ExperimentSpec, get_experiment, list_experiments
from clap.eval.test_sets import TEST_SETS, get_test_set, list_test_sets

__all__ = [
    "evaluate",
    "EXPERIMENTS",
    "ExperimentSpec",
    "get_experiment",
    "list_experiments",
    "DatasetSpec",
    "get_spec",
    "TEST_SETS",
    "get_test_set",
    "list_test_sets",
]
