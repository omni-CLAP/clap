"""Shared fixtures: skip data/GPU-dependent tests gracefully when the resource isn't available.

Unit tests never need these — only tests/integration/ tests that touch real
OXE data, a real checkpoint, or a GPU.
"""

import os

import pytest
import torch


def _env_dir_exists(var_name: str) -> bool:
    path = os.environ.get(var_name)
    return bool(path) and os.path.isdir(path)


requires_oxe_data = pytest.mark.skipif(
    not _env_dir_exists("CLAP_OXE_BASE_PATH") and not _env_dir_exists("OXE_BASE_PATH"),
    reason="CLAP_OXE_BASE_PATH (or OXE_BASE_PATH) not set to an existing directory",
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU available")


@pytest.fixture
def oxe_base_path() -> str:
    """Resolve the OXE dataset root from either env var name in use across this repo."""
    return os.environ.get("CLAP_OXE_BASE_PATH") or os.environ["OXE_BASE_PATH"]


@pytest.fixture
def oxe_lam_subdir() -> str:
    """Real LAM latent-action subdir name for this deployment's data, from CLAP_OXE_LAM_SUBDIR.

    Falls back to DataConfig's generic placeholder default, which only works
    if a site happens to have data under that exact name.
    """
    return os.environ.get("CLAP_OXE_LAM_SUBDIR", "latent_actions")


@pytest.fixture
def egodex_dreamdojo_lam_subdir() -> str:
    """Real dreamdojo-extractor egodex LAM subdir name, from CLAP_EGODEX_DREAMDOJO_LAM_SUBDIR.

    Must match `clap.data.lam.EGODEX_ONLY_LAM_SUBDIR_MARKERS` so it also
    exercises the egodex-only mix selection, not just a real subdir lookup.
    """
    return os.environ.get("CLAP_EGODEX_DREAMDOJO_LAM_SUBDIR", "dreamdojo_latent_actions_skip_1")
