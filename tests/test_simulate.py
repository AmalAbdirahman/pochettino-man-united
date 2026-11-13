"""Simulator checks (fast, no PyMC needed)."""

from collections import Counter

import numpy as np
import pandas as pd

from pochettino_fit.config import load_config
from pochettino_fit.features import one_row_problems
from pochettino_fit.simulate import block_of, ou_paths, round_robin, simulate


def test_round_robin_each_pair_home_and_away_once() -> None:
    days = round_robin(6)
    games = [g for d in days for g in d]
    assert len(days) == 10 and len(games) == 30
    assert Counter(games).most_common(1)[0][1] == 1
    for d in days:  # nobody plays twice on a matchday
        teams = [t for g in d for t in g]
        assert len(set(teams)) == 6


def test_ou_paths_are_stationary() -> None:
    rng = np.random.default_rng(0)
    eta = ou_paths(20000, 10, tau=0.12, gamma=0.7, rng=rng)
    assert np.allclose(eta.std(axis=0), 0.12, rtol=0.05)
    rho = np.exp(-0.7 * 0.5)
    lag1 = np.corrcoef(eta[:, 4], eta[:, 5])[0, 1]
    assert abs(lag1 - rho) < 0.03


def test_block_of() -> None:
    assert block_of(pd.Timestamp("2014-08-20")) == 4028
    assert block_of(pd.Timestamp("2015-01-10")) == 4029
    assert block_of(pd.Timestamp("2015-07-01")) == 4030


def test_simulated_table_shape() -> None:
    cfg = load_config()["simulation"]
    m, intl, truth = simulate(cfg["sizes"]["tiny"], cfg["truth"], seed=3)
    assert len(m) == 2 * 90
    assert one_row_problems(m) == []
    assert set(m.home_tenure) <= {1, 2, 3}
    assert m.w.between(10 / 90, 1).all()
    assert m.home_luck_n.max() <= 10
    # luck uses earlier matches only: every club's first match has no luck
    first = pd.concat([m[["date", "home_team", "home_luck_n"]].set_axis(["d", "c", "n"], axis=1),
                       m[["date", "away_team", "away_luck_n"]].set_axis(["d", "c", "n"], axis=1)])
    assert (first.sort_values("d").drop_duplicates("c").n == 0).all()
    assert set(truth["theta"].index.get_level_values(0)) >= set(m.home_manager)
    assert np.isclose(truth["kappa"].groupby(truth["kappa"].index.str[:2]).sum(), 0).all()
