"""Finishing shrinkage, team multiplier, recency weights and the publish filter."""

import json

import numpy as np
import pandas as pd
import pytest

from pochettino_fit import finishing as F
from pochettino_fit import profile as P
from pochettino_fit.config import ROOT


def test_phi_no_xg_returns_prior() -> None:
    mean, sd = F.phi_posterior(np.array([0.0]), np.array([0.0]), 0.85, 0.05)
    assert mean[0] == 0.0 and sd[0] == pytest.approx(0.05)


def test_phi_shrinks_small_samples_and_follows_large_ones() -> None:
    raw = 0.2
    x = np.array([5.0, 50.0, 1e6])
    mean, sd = F.phi_posterior(x * (1 + raw), x, 0.85, 0.05)
    assert mean[0] < mean[1] < mean[2] and mean[0] < 0.02
    assert mean[2] == pytest.approx(raw, rel=1e-3)
    assert np.all(np.diff(sd) < 0)


def test_team_multiplier_is_one_with_zero_phi() -> None:
    fu = F.team_multiplier(np.array([0.6, 0.4]), np.zeros(2), np.full(2, 0.05), 0.9)
    assert fu["mean"] == 1.0 and fu["lo"] < 1.0 < fu["hi"]
    assert fu["sd"] == pytest.approx(0.05 * np.sqrt(0.36 + 0.16))


def test_shares_sum_to_one_and_missing_is_zero() -> None:
    s = F.npxg_shares(pd.Series([0.3, np.nan, 0.1]))
    assert s.sum() == pytest.approx(1.0) and s.iloc[1] == 0.0


def test_season_rates_divide_by_club_matches() -> None:
    shots = pd.DataFrame({"player_id": [1, 1, 2], "team": ["A", "A", "A"], "understat_id": [10, 11, 12],
                          "season": ["2025-26"] * 3, "npxg": [0.3, 0.3, 0.6], "goal": [0, 1, 0]})
    r = F.season_rates(shots, "2025-26").set_index("player_id").rate
    assert r[1] == pytest.approx(0.6 / 3) and r[2] == pytest.approx(0.6 / 3)


def test_recency_weights_halve_at_half_life() -> None:
    ref = pd.Timestamp("2026-11-13")
    w = P.recency_weights(pd.Series(pd.to_datetime(["2026-11-13", "2026-05-17"])), ref, 180)
    assert w.sum() == pytest.approx(1.0) and w[1] / w[0] == pytest.approx(0.5)


def test_weighted_ratio_with_equal_weights_is_ratio_of_sums() -> None:
    df = pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 2.0]})
    m = P.Metric("r", "r", "a", "b")
    assert P.weighted_ratio(df, m, np.array([0.5, 0.5])) == pytest.approx(4 / 4)


def test_percentile_extremes() -> None:
    vals = np.arange(20.0)
    assert P.percentile(19.0, vals) == (100.0, 1)
    assert P.percentile(0.0, vals) == (0.0, 20)


def test_publish_filter_drops_local_only_blocks() -> None:
    full = {"finishing": {}, "profile": [{"source": "understat", "season": "x", "publishable": True},
                                         {"source": "fotmob", "season": "y", "publishable": False}]}
    out = P.publishable(full)
    assert [b["source"] for b in out["profile"]] == ["understat"]
    assert len(out["withheld"]) == 1 and len(full["profile"]) == 2


def test_published_profile_has_no_fotmob_values() -> None:
    path = ROOT / "docs" / "data" / "profile.json"
    if not path.exists():
        pytest.skip("profile.json not built")
    d = json.loads(path.read_text(encoding="utf-8"))
    assert all(b["publishable"] and "fotmob" not in b["source"] for b in d["profile"])
