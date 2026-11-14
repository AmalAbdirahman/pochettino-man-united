"""United's manager terms for the forecast (methods section 5b), from the prompt-3 posterior.

The attack/defence fit for United averages over the matches in its recency window, which were played
under Amorim, an interim coach and Carrick. So the term to strip is the same recency-weighted average of
(manager effect + tenure effect) over those matches, draw by draw. Then:
  baseline   = -stripped + delta_year1
  pochettino = -stripped + theta_Pochettino(now) + delta_year1
All on the npxG-difference-per-90 scale of the manager model.
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pochettino_fit.config import ROOT
from pochettino_fit.fit import theta_now
from pochettino_fit.simulate import block_of
from pochettino_fit.strength_model import Strength, recency


def load_posterior(cfg: dict[str, Any]) -> tuple[Any, pd.DataFrame]:
    path = ROOT / cfg["paths"]["processed"] / cfg["forecast"]["posterior"]
    post = xr.open_datatree(path, engine="h5netcdf")["posterior"]
    return post, pd.read_parquet(path.with_suffix(".states.parquet"))


def flat(post: Any, var: str) -> np.ndarray:
    v = post[var].values
    return v.reshape(-1, *v.shape[2:])


def state_effect(post: Any, states: pd.DataFrame, manager: str | None, block: int) -> np.ndarray:
    """Draws of theta for (manager, block); zeros (an average manager) if the state is not in the fit."""
    theta = flat(post, "theta")
    hit = states[(states.manager == manager) & (states.block == block)]
    return theta[:, hit.index[0]] if len(hit) else np.zeros(theta.shape[0])


def tenure_effect(delta: np.ndarray, tenure: float) -> np.ndarray:
    """delta[:, 0] for year 1, delta[:, 1] for year 2, zero for 3+ or unknown (same coding as the fit)."""
    if tenure == 1:
        return delta[:, 0]
    if tenure == 2:
        return delta[:, 1]
    return np.zeros(delta.shape[0])


def united_rows(m: pd.DataFrame, utd: int) -> pd.DataFrame:
    u = m[(m.home_team == utd) | (m.away_team == utd)]
    home = u.home_team == utd
    return pd.DataFrame({"date": u.date, "manager": u.home_manager.where(home, u.away_manager),
                         "tenure": u.home_tenure.where(home, u.away_tenure)})


def manager_draws(cfg: dict[str, Any], m: pd.DataFrame, st: Strength, utd: int) -> dict[str, np.ndarray]:
    fc = cfg["forecast"]
    post, states = load_posterior(cfg)
    delta = flat(post, "delta")
    cutoff = pd.Timestamp(cfg["features"]["data_cutoff"])
    rows = united_rows(m[m.date <= cutoff], utd)
    w = recency(rows.date, cutoff + pd.Timedelta(days=1), fc["half_life_days"])
    terms = [state_effect(post, states, r.manager, block_of(r.date)) + tenure_effect(delta, r.tenure)
             for r in rows.itertuples()]
    stripped = np.average(np.column_stack(terms), axis=1, weights=w)
    now = block_of(pd.Timestamp(cfg["fit"]["now"]))
    theta_p = theta_now(post, states, cfg["fit"]["manager"], now)
    base = -stripped + delta[:, 0]
    return {"stripped": stripped, "baseline": base, "pochettino": base + theta_p, "theta_p": theta_p,
            "sigma": flat(post, "sigma")}


def split_effect(eff: np.ndarray, st: Strength, u: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn an npxG-difference shift into log attack and log defence shifts, half each.

    Against an average opponent at a neutral ground United expect mu_f for and mu_a against; the shift
    adds eff/2 to npxG for and removes eff/2 from npxG against.
    """
    c, h = st.coef[0], st.coef[1]
    k = len(st.teams) - 1
    a = np.concatenate([[0.0], st.coef[2:2 + k]])
    d = np.concatenate([[0.0], st.coef[2 + k:]])
    mu_f = np.exp(c + h / 2 + a[u] - d.mean())
    mu_a = np.exp(c + h / 2 + a.mean() - d[u])
    ea = np.log1p(eff / (2 * mu_f))
    ed = -np.log1p(-np.clip(eff / (2 * mu_a), None, 0.9))
    return ea, ed
