"""Tracker: fixture matching, points band, normal update, publishable split, and the frozen-file guard."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pochettino_fit import tracker as T


def frozen() -> dict:
    fx = [{"opponent": "Liverpool FC", "venue": "away", "npxg_diff_mean": -0.5,
           "npxg_diff_interval": [-1.5, 0.5], "interval": 0.8, "baseline_npxg_diff_mean": -0.6,
           "p_win_draw_loss": [0.2, 0.3, 0.5], "expected_points": 0.9, "points_draws": "3100"},
          {"opponent": "Brighton & Hove Albion", "venue": "home", "npxg_diff_mean": 0.3,
           "npxg_diff_interval": [-0.7, 1.3], "interval": 0.8, "baseline_npxg_diff_mean": 0.2,
           "p_win_draw_loss": [0.5, 0.3, 0.2], "expected_points": 1.8, "points_draws": "3310"}]
    return {"united_fixtures": fx, "united_now": {"played": 2},
            "detectability": {"effect_mean": 0.1, "effect_sd_now": 0.15, "sigma": 1.2}}


def notes(rows: list[tuple]) -> pd.DataFrame:
    cols = ["gameweek", "opponent", "venue", "goals_for", "goals_against", "points", "npxg_for",
            "npxg_against", "shots_against", "big_chances_against"]
    return pd.DataFrame(rows, columns=cols)


BEFORE = [(1, "Hull City", "away", 0, 2, 0, 1.8, 1.0, 8, 3), (2, "Everton", "away", 2, 2, 1, 1.2, 1.1, 12, 2)]


def test_club_key_matches_source_spellings() -> None:
    assert T.club_key("Brighton & Hove Albion") == T.club_key("Brighton") == "brighton"
    assert T.club_key("AFC Bournemouth") == T.club_key("Bournemouth")
    assert T.club_key("Liverpool FC") == T.club_key("Liverpool")


def test_match_fixture_needs_venue_and_one_hit() -> None:
    fx = frozen()["united_fixtures"]
    assert T.match_fixture(fx, "Brighton", "home") == 1
    with pytest.raises(ValueError):
        T.match_fixture(fx, "Brighton", "away")


def test_predictive_interval_adds_match_noise() -> None:
    fx = frozen()["united_fixtures"][0]
    lo, hi = T.predictive_interval(fx, 1.2)
    assert lo < -1.5 and hi > 0.5 and (lo + hi) / 2 == pytest.approx(-0.5)
    assert T.predictive_interval(fx, 0.0) == pytest.approx([-1.5, 0.5])


def test_track_with_no_new_matches_returns_the_prior() -> None:
    out = T.track(frozen(), notes(BEFORE), [0.5, 0.8])
    assert out["matches"] == [] and out["effect"]["mean"] == pytest.approx(0.1)
    assert out["effect"]["sd"] == pytest.approx(0.15)


def test_track_one_match() -> None:
    rows = BEFORE + [(3, "Liverpool", "away", 1, 1, 1, 0.9, 1.4, 10, 1)]
    out = T.track(frozen(), notes(rows), [0.5, 0.8])
    m = out["matches"][0]
    assert m["actual_npxg_diff"] == pytest.approx(-0.5) and m["inside_interval"]
    assert m["observed_effect"] == pytest.approx(0.1)
    assert out["cumulative_points"]["actual"] == 1 and out["cumulative_points"]["forecast_mean"] == 1.0
    assert len(out["cumulative_by_match"]) == 1 and out["cumulative_by_match"][-1] == out["cumulative_points"]
    cq = out["chance_quality_conceded"]
    assert cq["before"]["npxg_per_shot_against"] == pytest.approx(2.1 / 20)
    assert cq["since"]["big_chances_against_per_match"] == 1.0


def test_points_band_sums_draws_across_fixtures() -> None:
    band = T.points_band(frozen()["united_fixtures"], [0, 1], 4, [0.8])
    assert band["forecast_mean"] == pytest.approx(np.mean([6, 4, 1, 0]))
    assert band["pit"] == pytest.approx(0.5 + 0.5 * 0.25)


def test_effect_update_matches_conjugate_formula() -> None:
    e = T.effect_update(0.1, 0.15, 1.2, [0.5, -0.2, 0.3])
    prec = 1 / 0.15**2 + 3 / 1.2**2
    assert e["sd"] == pytest.approx(prec**-0.5)
    assert e["mean"] == pytest.approx((0.1 / 0.15**2 + 0.6 / 1.2**2) / prec)


def test_publishable_drops_fotmob_values() -> None:
    rows = BEFORE + [(3, "Liverpool", "away", 1, 1, 1, 0.9, 1.4, 10, 1)]
    out = T.track(frozen(), notes(rows), [0.8])
    for mode in T.MODES:
        text = json.dumps(T.publishable(out, mode))
        assert not any(f in text for f in ("actual_npxg_diff", "chance_quality", "observed_effect"))
    pub = T.publishable(out, "derived_only")
    assert pub["matches"][0]["points"] == 1 and "effect" in pub and "inside_interval" in pub["matches"][0]
    strict = T.publishable(out, "results_only")
    assert strict["matches"][0]["points"] == 1 and "cumulative_points" in strict
    assert not any(f in json.dumps(strict) for f in T.MODEL_FIELDS)
    with pytest.raises(ValueError):
        T.publishable(out, "everything")


def test_tracker_refuses_to_write_into_forecasts(tmp_path: Path) -> None:
    fdir = tmp_path / "forecasts"
    fdir.mkdir()
    frozen_file = fdir / "2026-11-20.json"
    frozen_file.write_text('{"x": 1}', encoding="utf-8")
    with pytest.raises(PermissionError):
        T.guarded_write(frozen_file, {"x": 2}, fdir)
    with pytest.raises(PermissionError):
        T.guarded_write(fdir / "tracker.json", {"x": 2}, fdir)
    assert frozen_file.read_text(encoding="utf-8") == '{"x": 1}'
    T.guarded_write(tmp_path / "tracker.json", {"x": 2}, fdir)


def test_original_forecast_is_the_earliest_file(tmp_path: Path) -> None:
    (tmp_path / "2026-11-20.json").write_text('{"v": 1}', encoding="utf-8")
    (tmp_path / "2026-12-30.json").write_text('{"v": 2}', encoding="utf-8")
    path, f = T.original_forecast(tmp_path)
    assert path.name == "2026-11-20.json" and f["v"] == 1
    with pytest.raises(FileNotFoundError):
        T.original_forecast(tmp_path / "empty")
