"""Style fit: metric preparation, real managerial changes, D^2, natural units, and a recovery run (slow)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")

from pochettino_fit import style as S  # noqa: E402
from pochettino_fit import style_model as SM  # noqa: E402


def rows() -> pd.DataFrame:
    return pd.DataFrame({
        "league": "ENG", "season": "2025-26", "team": [1, 1, 2, 2, 1],
        "date": pd.to_datetime(["2025-08-01", "2025-08-08", "2025-08-01", "2025-08-08", "2025-08-15"]),
        "manager": ["A", "A", "B", "B", "C"], "shots_close_for": [10, 0, 5, 20, 10],
        "npxg_close_for": [1.0, 0.0, 0.5, 3.0, 2.0], "ppda_for": [10.0, 0.0, 8.0, 12.0, 9.0]})


def test_ratio_metric_drops_zero_denominators_and_weights_by_shots() -> None:
    d = SM.prepare(rows(), SM.StyleMetric("x", "x", "npxg_close_for", "shots_close_for"))
    assert len(d.z) == 4
    assert d.w == pytest.approx(np.array([10, 5, 20, 10]) / 11.25)
    assert np.average(d.z, weights=d.w) == pytest.approx(0.0, abs=1e-12)


def test_log_metric_drops_non_positive_values() -> None:
    d = SM.prepare(rows(), SM.BY_NAME["ppda"])
    assert len(d.z) == 4 and d.managers == ["A", "B", "C"]


def test_real_changes_within_a_club_only() -> None:
    ch = S.real_changes(rows())
    assert list(zip(ch.outgoing, ch.incoming, strict=True)) == [("A", "C")]


def test_mahalanobis_with_identity_and_correlated_metrics() -> None:
    d = np.array([1.0, 1.0])
    assert d @ np.linalg.inv(np.eye(2)) @ d == pytest.approx(2.0)
    r = np.array([[1.0, 0.9], [0.9, 1.0]])
    assert d @ np.linalg.inv(r) @ d == pytest.approx(2 / 1.9)   # correlated metrics are not double counted


def fake_fit(log: bool = False) -> dict:
    n = 4000
    rng = np.random.default_rng(0)
    return {"theta_keep": np.column_stack([rng.normal(0.3, 0.05, n), rng.normal(0.0, 0.05, n)]),
            "centre": 2.0 if log else 10.0, "scale": 0.5 if log else 2.0, "lam_ref": np.zeros(n),
            "kappa_utd": np.zeros(n), "sigma_theta": np.full(n, 0.2), "sigma_kappa": np.full(n, 0.4),
            "theta_mean": np.array([0.1, -0.2, 0.3]), "n_matches": np.array([100, 100, 10]),
            "max_rhat": 1.0, "n_rows": 10}


def test_metric_summary_in_natural_units() -> None:
    m = SM.StyleMetric("x", "x", "x", better="lower")
    s = S.metric_summary(fake_fit(), m, 0.8, 76)
    assert s["change"]["mean"] == pytest.approx(0.6, abs=0.01)          # 2.0 * 0.3
    assert s["p_improves"] < 0.01 and s["manager_share_of_style_variation"] == pytest.approx(0.2)
    assert s["pochettino_abs_effect_percentile"] == 1.0                  # |0.3| > |0.1|, |-0.2|
    lg = S.metric_summary(fake_fit(log=True), SM.StyleMetric("p", "p", "p", log=True), 0.8, 76)
    assert lg["united_now"]["mean"] == pytest.approx(np.exp(2.0), rel=0.01)


@pytest.mark.slow
def test_style_model_recovers_manager_effects() -> None:
    rng = np.random.default_rng(5)
    n_club, n_mgr, n = 20, 40, 6000
    kappa, theta = rng.normal(0, 0.4, n_club), rng.normal(0, 0.25, n_mgr)
    club = rng.integers(0, n_club, n)
    mgr = (club * 2 + rng.integers(0, 2, n)) % n_mgr
    moved = rng.random(n) < 0.3                       # managers also appear at other clubs
    mgr[moved] = rng.integers(0, n_mgr, moved.sum())
    x = kappa[club] + theta[mgr] + rng.normal(0, 1, n)
    t = pd.DataFrame({"league": "ENG", "season": "2020-21", "team": club,
                      "manager": [f"m{i:02d}" for i in mgr], "x": x, "date": pd.Timestamp("2020-08-01")})
    d = SM.prepare(t, SM.StyleMetric("x", "x", "x"))
    p = SM.sample(d, {"draws": 500, "tune": 500, "chains": 2, "cores": 1, "target_accept": 0.9,
                      "mode": "NUMBA"}, 1)
    est = p["theta"].mean(0) * d.scale
    truth = theta[[int(m[1:]) for m in d.managers]]
    assert np.corrcoef(est, truth)[0, 1] > 0.6
