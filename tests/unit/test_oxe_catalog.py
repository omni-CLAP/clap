"""Unit tests for clap.data.oxe_catalog: the shared per-embodiment registry."""

import pytest

from clap.data.oxe_catalog import (
    OXE_CATALOG,
    OXE_EE_DATASET_ORDER,
    OXE_EE_SAMPLING_WEIGHTS,
    OXE_LAM_DATASET_ORDER,
    OXE_LAM_SAMPLING_WEIGHTS,
    OXE_LANG_DATASET_ORDER,
    OXE_LANG_SAMPLING_WEIGHTS,
    get_embodiment_config,
)


def test_get_embodiment_config_known_name():
    cfg = get_embodiment_config("droid")
    assert cfg.name == "droid"
    assert cfg.stacking_mode == "three_view"


def test_get_embodiment_config_unknown_name_raises():
    with pytest.raises(KeyError):
        get_embodiment_config("not_a_real_embodiment")


@pytest.mark.parametrize("order,weights", [
    (OXE_EE_DATASET_ORDER, OXE_EE_SAMPLING_WEIGHTS),
    (OXE_LAM_DATASET_ORDER, OXE_LAM_SAMPLING_WEIGHTS),
    (OXE_LANG_DATASET_ORDER, OXE_LANG_SAMPLING_WEIGHTS),
])
def test_dataset_order_and_weights_have_matching_length(order, weights):
    assert len(order) == len(weights)


@pytest.mark.parametrize("order", [OXE_EE_DATASET_ORDER, OXE_LAM_DATASET_ORDER, OXE_LANG_DATASET_ORDER])
def test_every_mix_dataset_is_registered_in_the_catalog(order):
    for name in order:
        assert name in OXE_CATALOG, name


def test_g1_humanoid_config():
    cfg = get_embodiment_config("g1_humanoid")
    assert cfg.stacking_mode == "four_view"
    assert cfg.action_mode == "joint26"
    assert cfg.hand_type == "brainco"
    assert cfg.cam_ids == [0, 1, 2, 3]


def test_bimanual_yam_config():
    cfg = get_embodiment_config("bimanual_yam")
    assert cfg.stacking_mode == "three_view"
    assert cfg.action_mode == "joint14"


def test_bridge_and_bimanual_yam_are_ee_dataset_config_entries_not_subclasses():
    """Confirms the architecture decision: bridge/bimanual_yam are plain EmbodimentConfig
    entries consumed by EEDataset, not separate dataset classes."""
    from clap.config.data import EmbodimentConfig

    assert isinstance(OXE_CATALOG["bridge"], EmbodimentConfig)
    assert isinstance(OXE_CATALOG["bimanual_yam"], EmbodimentConfig)


def test_annotation_subdir_defaults_to_annotation_for_every_entry():
    for name in OXE_CATALOG:
        assert get_embodiment_config(name).annotation_subdir == "annotation"


def test_ann_subdir_helper_overridable_via_env_var(monkeypatch):
    """A site whose DATASET annotations live under a different directory name sets
    CLAP_DATASET_ANNOTATION_SUBDIR rather than the package hardcoding that value.

    `_ann_subdir` is only read at OXE_CATALOG-construction (import) time, so this
    tests the helper directly rather than the already-built module-level catalog.
    """
    from clap.data.oxe_catalog import _ann_subdir

    assert _ann_subdir("droid") == "annotation"
    monkeypatch.setenv("CLAP_DROID_ANNOTATION_SUBDIR", "annotation2")
    assert _ann_subdir("droid") == "annotation2"
    assert _ann_subdir("bridge") == "annotation"
