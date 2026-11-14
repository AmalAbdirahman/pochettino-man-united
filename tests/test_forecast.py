"""Attack/defence model, season simulation, manager-term split, back-test scores and the freeze guard."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from pochettino_fit import backtest as B  # noqa: E402
from pochettino_fit import season_sim as sim  # noqa: E402
from pochettino_fit.strength_model import Strength, fit_strength  # noqa: E402


def toy_league(n_teams: int = 6, rounds: int = 12,
               seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = np.r_[0.0, rng.normal(0, 0.25, n_teams - 1)]
    d = np.r_[0.0, rng.normal(0, 0.25, n_teams - 1)]
    rows, day = [], pd.Timestamp("2025-08-01")
    for _ in range(rounds):
        for h in range(n_teams):
            for w in range(n_teams):
                if h != w:
                    rows.append({"home_team": h, "away_team": w, "date": day,
                                 "home_npxg": np.exp(0.2 + 0.15 + a[h] - d[w]) * rng.gamma(20, 1 / 20),
                                 "away_npxg": np.exp(0.2 + a[w] - d[h]) * rng.gamma(20, 1 / 20)})
            day += pd.Timedelta(days=1)
    return pd.DataFrame(rows), a, d


def test_strength_model_recovers_attack_and_defence() -> None:
    m, a, d = toy_league()
    st = fit_strength(m, pd.Timestamp("2026-01-01"), 1e6)
    k = len(st.teams) - 1
    assert np.allclose(st.coef[2:2 + k], a[1:], atol=0.08)
    assert np.allclose(st.coef[2 + k:], d[1:], atol=0.08)
    assert st.coef[1] == pytest.approx(0.15, abs=0.05)


def test_remaining_fixtures_complete_the_double_round_robin() -> None:
    played = pd.DataFrame({"home_team": [1, 2], "away_team": [2, 3]})
    fix = sim.remaining_fixtures(played, [1, 2, 3])
    assert len(fix) == 6 - 2 and (1, 2) not in set(zip(fix.home_team, fix.away_team, strict=True))


def test_season_table_positions_are_a_permutation() -> None:
    rng = np.random.default_rng(1)
    gh, ga = rng.integers(0, 4, (50, 6)), rng.integers(0, 4, (50, 6))
    hi, ai = np.array([0, 1, 2, 1, 2, 0]), np.array([1, 2, 0, 0, 1, 2])
    base = {k: np.zeros(3) for k in ("pts", "gf", "ga")}
    tab = sim.season_table(gh, ga, hi, ai, 3, base, rng.random((50, 3)))
    assert np.all(np.sort(tab["pos"], axis=1) == np.array([1, 2, 3]))
    top_pts = tab["pts"][tab["pos"] == 1]
    assert np.all(top_pts == tab["pts"].max(axis=1))


def test_goals_mean_matches_rate_without_shock() -> None:
    rng = np.random.default_rng(2)
    lam = np.full((200_000, 1), 1.3)
    g = sim.goals(lam, rng.normal(size=lam.shape), rng.random(lam.shape), 0.0, 1.0, 0.1, 0.9)
    assert g.mean() == pytest.approx(0.9 * 1.3 + 0.1, rel=0.01)


def test_goal_calibration_ratio() -> None:
    m = pd.DataFrame({"home_goals": [2, 1], "away_goals": [0, 1], "home_pen_goals": [1, 0],
                      "away_pen_goals": [0, np.nan], "home_npxg": [1.0, 1.0], "away_npxg": [0.5, 0.5]})
    k = sim.goal_calibration(m, np.ones(2), pen_rate=0.0)
    assert k == pytest.approx((1 + 1 + 0 + 1) / 3.0)


def test_split_effect_moves_npxg_by_half_each_way() -> None:
    from pochettino_fit.united_effects import split_effect

    coef = np.array([0.1, 0.2, 0.3, -0.1, 0.2, 0.1])          # 3 teams: a2, a3, d2, d3
    st = Strength([0, 1, 2], coef, np.eye(6) * 1e-6, 0.0)
    a, d = np.r_[0, coef[2:4]], np.r_[0, coef[4:6]]
    u, eff = 1, np.array([0.3])
    ea, ed = split_effect(eff, st, u)
    mu_f = np.exp(coef[0] + coef[1] / 2 + a[u] - d.mean())
    mu_a = np.exp(coef[0] + coef[1] / 2 + a.mean() - d[u])
    assert mu_f * np.exp(ea[0]) - mu_f == pytest.approx(0.15)
    assert mu_a - mu_a * np.exp(-ed[0]) == pytest.approx(0.15)


def test_tenure_effect_coding() -> None:
    from pochettino_fit.united_effects import tenure_effect

    delta = np.array([[0.1, 0.2]])
    assert tenure_effect(delta, 1)[0] == 0.1 and tenure_effect(delta, 2)[0] == 0.2
    assert tenure_effect(delta, 3)[0] == 0.0 and tenure_effect(delta, np.nan)[0] == 0.0


def test_crps_matches_brute_force() -> None:
    x = np.random.default_rng(3).integers(0, 30, 300)
    brute = np.mean(np.abs(x - 12)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert B.crps(x, 12) == pytest.approx(brute)


def test_scores_for_a_point_mass() -> None:
    x = np.full(100, 20)
    assert B.interval_covers(x, 20, 0.5) and not B.interval_covers(x, 21, 0.8)
    assert B.pit(x, 20, 0.25) == pytest.approx(0.25) and B.pit(x, 30, 0.5) == 1.0
    assert B.log_score(x, 20, 60) > B.log_score(x, 21, 60)


def test_appointments_found_mid_season() -> None:
    dates = pd.date_range("2020-08-01", periods=20, freq="7D")
    mgr = ["Old"] * 10 + ["New"] * 10
    m = pd.DataFrame({"league": "ENG", "season": "2020-21", "date": dates, "home_team": 1, "away_team": 2,
                      "home_manager": mgr, "away_manager": "Other"})
    a = B.appointments(m, 6, 8)
    assert len(a) == 1 and a.iloc[0].cutoff == dates[10] and a.iloc[0].incoming == "New"


def test_freeze_refuses_to_overwrite_and_late_freeze(tmp_path: pytest.TempPathFactory) -> None:
    from pochettino_fit.forecast import write_frozen

    cfg = {"paths": {"forecasts": str(tmp_path)}, "forecast": {"first_league_match": "2026-11-22"}}
    day = dt.date(2026, 11, 15)
    path = write_frozen(cfg, {"x": 1}, day)
    assert path.exists()
    with pytest.raises(FileExistsError):
        write_frozen(cfg, {"x": 2}, day)
    assert '"x": 1' in path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        write_frozen(cfg, {"x": 3}, dt.date(2026, 11, 22))
