"""Unit tests for clap.eval.experiments: the checkpoint registry every eval/example script keys off."""

import pytest

from clap.eval.experiments import EXPERIMENTS, get_experiment, list_experiments


def test_every_ckpts_root_matches_its_dict_key():
    """Every experiment's ckpts_root must equal its own registry key — the checkpoint
    directory name and the name used to look it up should never diverge."""
    mismatches = [k for k, v in EXPERIMENTS.items() if k != v.ckpts_root]
    assert mismatches == []


def test_adaptation_category_has_both_novel_embodiments():
    adaptation = list_experiments("adaptation")
    assert "adapt_bimanual_yam" in adaptation
    assert "adapt_g1_humanoid" in adaptation


def test_cross_embodiment_and_post_train_bases_line_up():
    """Every cross_embodiment_oxe_<X> should have a matching post_train_<X>_droid and _bridge."""
    cross_embodiment = list_experiments("cross_embodiment")
    bases = [name[len("cross_embodiment_"):] for name in cross_embodiment]
    for base in bases:
        assert f"post_train_{base}_droid" in EXPERIMENTS, base
        assert f"post_train_{base}_bridge" in EXPERIMENTS, base


def test_get_experiment_unknown_name_raises_with_categories_listed():
    with pytest.raises(KeyError, match="baselines"):
        get_experiment("not_a_real_experiment")


def test_list_experiments_unknown_category_raises():
    with pytest.raises(ValueError):
        list_experiments("not_a_real_category")


def test_list_experiments_no_category_returns_everything_sorted():
    assert list_experiments() == sorted(EXPERIMENTS)