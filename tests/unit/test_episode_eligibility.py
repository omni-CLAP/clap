"""Unit tests for clap.eval.episode_eligibility: pure length-arithmetic, no real data needed."""

from clap.eval.episode_eligibility import _ceildiv, chunk_eligible, per_path_t_total


def test_ceildiv():
    assert _ceildiv(10, 3) == 4
    assert _ceildiv(9, 3) == 3
    assert _ceildiv(1, 3) == 1


def test_chunk_eligible_missing_source_is_ineligible():
    lengths = {"n_video": 100, "n_state": None, "n_lam": 100}
    assert chunk_eligible(lengths, d=1, T_required=18) is False


def test_chunk_eligible_true_when_all_long_enough():
    lengths = {"n_video": 100, "n_state": 100, "n_lam": 100}
    assert chunk_eligible(lengths, d=1, T_required=18) is True


def test_chunk_eligible_false_when_video_too_short():
    lengths = {"n_video": 10, "n_state": 100, "n_lam": 100}
    assert chunk_eligible(lengths, d=1, T_required=18) is False


def test_chunk_eligible_respects_downsample_ratio():
    # n_video=52 downsampled by d=3 -> ceildiv(52, 3) = 18, exactly at the boundary.
    lengths = {"n_video": 52, "n_state": 52, "n_lam": 18}
    assert chunk_eligible(lengths, d=3, T_required=18) is True
    # n_video=51 -> ceildiv(51, 3) = 17, one below the boundary.
    lengths_short = {"n_video": 51, "n_state": 51, "n_lam": 18}
    assert chunk_eligible(lengths_short, d=3, T_required=18) is False


def test_chunk_eligible_lam_already_at_downsampled_rate():
    # n_lam is NOT divided by d (already downsampled); n_video/n_state are.
    lengths = {"n_video": 54, "n_state": 54, "n_lam": 17}
    assert chunk_eligible(lengths, d=3, T_required=18) is False  # lam_t = min(18, 17) = 17 < 18


def test_per_path_t_total_missing_video_gives_all_none():
    lengths = {"n_video": None, "n_state": 100, "n_lam": 100}
    out = per_path_t_total(lengths, d=1)
    assert out == {"ee": None, "lam": None}


def test_per_path_t_total_computes_per_family():
    lengths = {"n_video": 100, "n_state": 90, "n_lam": 80}
    out = per_path_t_total(lengths, d=1)
    assert out["ee"] == 90  # min(nv=100, n_state=90)
    assert out["lam"] == 80  # min(nv=100, n_lam=80)
