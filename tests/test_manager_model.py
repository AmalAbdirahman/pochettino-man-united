"""Manager model: data preparation, graph build and (slow) a short recovery run."""

import numpy as np
import pandas as pd
import pytest

pm = pytest.importorskip("pymc")

from pochettino_fit.config import load_config  # noqa: E402
from pochettino_fit.manager_model import build_model, prepare  # noqa: E402
from pochettino_fit.simulate import simulate  # noqa: E402


@pytest.fixture(scope="module")
def tiny() -> tuple:
    cfg = load_config()["simulation"]
    return simulate(cfg["sizes"]["tiny"], cfg["truth"], seed=2)


def test_prepare_states_and_gaps(tiny: tuple) -> None:
    m, intl, _ = tiny
    d = prepare(m, intl)
    st = d.states
    assert len(st) == len(st.drop_duplicates(["manager", "block"]))
    # states of one manager are in time order; first gap 0, later gaps multiples of half a year
    for _, g in st.groupby("manager"):
        gaps = d.gap_years[g.m_idx.iloc[0], : len(g)]
        assert gaps[0] == 0 and (gaps[1:] >= 0.5).all() and np.allclose(gaps[1:] % 0.5, 0)
    assert (d.state_matrix[d.state_matrix < len(st)] >= 0).all()
    assert len(d.y) == len(m)


def test_prepare_drops_missing_managers_and_imputes_values(tiny: tuple) -> None:
    m, intl, _ = tiny
    m = m.copy()
    m.loc[0, "home_manager"] = None
    m.loc[m.home_team == m.home_team.iloc[5], "home_log_squad_value"] = np.nan
    m.loc[m.away_team == m.home_team.iloc[5], "away_log_squad_value"] = np.nan
    d = prepare(m, intl)
    assert d.dropped["no_manager"] == 1
    assert d.dropped["missing_value"] >= 1 and len(d.logv_fill) == d.dropped["missing_value"]


def test_model_logp_is_finite(tiny: tuple) -> None:
    m, intl, _ = tiny
    model = build_model(prepare(m, intl))
    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    grad = model.compile_dlogp()(point)
    assert np.isfinite(grad).all()


def test_prepare_accepts_real_columns() -> None:
    """Real tables have dates but no block column, and NaN tenure/luck."""
    m = pd.DataFrame({
        "league": ["ENG"] * 2, "season": ["2020-21"] * 2,
        "date": pd.to_datetime(["2020-09-01", "2021-02-01"]),
        "home_team": [1, 2], "away_team": [2, 1], "home_manager": ["A", "B"], "away_manager": ["B", "A"],
        "home_tenure": [1.0, np.nan], "away_tenure": [3.0, 2.0], "home_luck": [np.nan, 0.2],
        "away_luck": [0.1, np.nan], "home_log_squad_value": [19.0, 20.0],
        "away_log_squad_value": [20.0, 19.0], "w": [1.0, 0.5], "y": [0.3, -0.2],
    })
    d = prepare(m)
    assert len(d.states) == 4  # two managers x two half-season blocks
    assert d.home_ten.tolist() == [0, 2] and d.away_ten.tolist() == [2, 1]


@pytest.mark.slow
def test_tiny_recovery_runs_clean(tiny: tuple) -> None:
    m, intl, truth = tiny
    with build_model(prepare(m, intl)):
        idata = pm.sample(300, tune=300, chains=2, cores=1, random_seed=1, progressbar=False,
                          compile_kwargs={"mode": "NUMBA"})
    assert int(idata.sample_stats["diverging"].values.sum()) < 10
    sig = idata.posterior["sigma"].values.ravel()
    assert np.quantile(sig, 0.01) < truth["scalars"]["sigma"] < np.quantile(sig, 0.99)
