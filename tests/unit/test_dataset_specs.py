"""Unit tests for clap.eval.dataset_specs: eval-only per-dataset defaults."""

from clap.data.oxe_catalog import OXE_CATALOG
from clap.eval.dataset_specs import DATASET_SPECS, get_spec


def test_every_oxe_catalog_entry_has_a_spec():
    for name in OXE_CATALOG:
        assert name in DATASET_SPECS


def test_get_spec_known_dataset_inherits_stacking_mode():
    spec = get_spec("droid")
    assert spec.stacking_mode == "three_view"


def test_get_spec_unregistered_name_falls_back_to_defaults():
    spec = get_spec("not_a_real_dataset")
    assert spec.name == "not_a_real_dataset"
    assert spec.max_chunks == 0
    assert spec.trim_static_prefix is False


def test_stanford_hydra_override_applied():
    spec = get_spec("stanford_hydra")
    assert spec.skip_first_n_frames == 5


def test_droid_has_default_skip_first_n_frames():
    spec = get_spec("droid")
    assert spec.skip_first_n_frames == 0
