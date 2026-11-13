"""Single-stage pairwise manager model (docs/methods.md sections 1-2), in PyMC.

y_k = h[l,s] + (S_H - S_A) + (theta_H - theta_A) + (delta_H - delta_A) + beta (L_H - L_A) + eps,
eps ~ N(0, sigma^2 / w_k), theta = mu_m + eta_m,b (OU per block),
S_c,s = kappa_c + b_between * mean_c(logV) + b_value * (logV_c,s - mean_c(logV))   (Mundlak split).

The league offset lambda[l,s] is in S in methods.md but cancels exactly in y, because both
teams in a league match share it, so it is left out of the likelihood (see recovery notes).
The Mundlak split keeps b_value from absorbing persistent club differences when kappa is
shrunk; kappa alone is not separately identified from b_between, only club strength S is.
"""

import os
from dataclasses import dataclass

from pochettino_fit.config import load_config

# Must be set before pytensor is imported so Windows worker processes inherit it.
os.environ.setdefault("PYTENSOR_FLAGS", f"mode={load_config()['simulation']['sampling']['mode']}")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402
import pytensor.tensor as pt  # noqa: E402

from pochettino_fit.simulate import BLOCK_YEARS, block_of  # noqa: E402


@dataclass
class ModelData:
    y: np.ndarray
    w: np.ndarray
    ls_idx: np.ndarray
    ls_labels: list[tuple[str, str]]
    home_club: np.ndarray
    away_club: np.ndarray
    club_league: np.ndarray
    clubs: list
    home_state: np.ndarray
    away_state: np.ndarray
    states: pd.DataFrame          # manager, block, m_idx, pos
    state_matrix: np.ndarray      # (n_mgr, J) flat state index, padded with n_states
    gap_years: np.ndarray         # (n_mgr, J) years since previous state (0 at j = 0 and padding)
    managers: list[str]
    home_ten: np.ndarray          # 0 = year 1, 1 = year 2, 2 = year 3+ (baseline)
    away_ten: np.ndarray
    luck_diff: np.ndarray
    cs_home: np.ndarray           # index into club-season log value vector
    cs_away: np.ndarray
    cs_table: pd.DataFrame        # one row per club-season: league, season, club, club_idx
    logv_obs: np.ndarray          # observed log values, NaN where missing
    logv_fill: np.ndarray         # prior mean for each missing value
    logv_club_mean: np.ndarray    # per club-season row: the club's mean observed log value
    intl_state: np.ndarray
    intl_r: np.ndarray
    intl_w: np.ndarray
    dropped: dict[str, int]


def _tenure_cat(t: pd.Series) -> np.ndarray:
    return np.where(t == 1, 0, np.where(t == 2, 1, 2)).astype(int)


def _states(m: pd.DataFrame, intl: pd.DataFrame) -> pd.DataFrame:
    parts = [m[["home_manager", "block"]].set_axis(["manager", "block"], axis=1),
             m[["away_manager", "block"]].set_axis(["manager", "block"], axis=1),
             intl[["manager", "block"]]]
    st = pd.concat(parts).drop_duplicates().sort_values(["manager", "block"]).reset_index(drop=True)
    st["pos"] = st.groupby("manager").cumcount()
    st["m_idx"] = pd.factorize(st.manager, sort=True)[0]
    return st


def _state_arrays(st: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_mgr, J = st.m_idx.max() + 1, st.pos.max() + 1
    mat = np.full((n_mgr, J), len(st), dtype=int)
    mat[st.m_idx, st.pos] = st.index
    gaps = np.full((n_mgr, J), 100.0)  # padding: rho ~ 0 keeps sqrt(1 - rho^2) differentiable
    prev = st.groupby("manager").block.shift()
    gap = ((st.block - prev) * BLOCK_YEARS).fillna(0).to_numpy()
    gaps[st.m_idx, st.pos] = gap
    return mat, gaps


def _club_values(m: pd.DataFrame, cidx: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Club-season log values; missing ones get a prior centred on the league-season 10th percentile.

    Returns home and away row indices, the club-season table (with club mean of observed values)
    and the prior means for the missing values.
    """
    long = pd.concat([
        m[["league", "season", "home_team", "home_log_squad_value"]].set_axis(["l", "s", "c", "v"], axis=1),
        m[["league", "season", "away_team", "away_log_squad_value"]].set_axis(["l", "s", "c", "v"], axis=1),
    ]).drop_duplicates(["s", "c"]).reset_index(drop=True)
    q10 = long.groupby(["l", "s"]).v.transform(lambda v: v.quantile(0.1))
    prior = q10.fillna(long.v.quantile(0.1))
    long["club_mean"] = long.groupby("c").v.transform("mean").fillna(prior)
    long["club_idx"] = long.c.map(cidx)
    fill = prior[long.v.isna()].to_numpy()
    key = {(s, c): i for i, (s, c) in enumerate(zip(long.s, long.c, strict=True))}
    home = np.array([key[(s, c)] for s, c in zip(m.season, m.home_team, strict=True)])
    away = np.array([key[(s, c)] for s, c in zip(m.season, m.away_team, strict=True)])
    return home, away, long, fill


def prepare(matches: pd.DataFrame, intl: pd.DataFrame | None = None) -> ModelData:
    """Turn a matches table (synthetic or real) into model arrays."""
    m = matches[matches.y.notna()].copy()
    no_mgr = m.home_manager.isna() | m.away_manager.isna()
    m = m[~no_mgr].reset_index(drop=True)
    if "block" not in m:
        m["block"] = m.date.map(block_of)
    intl = intl if intl is not None else pd.DataFrame(columns=["manager", "block", "w", "r"])
    intl = intl[intl.manager.isin(set(m.home_manager) | set(m.away_manager))].reset_index(drop=True)
    st = _states(m, intl)
    mat, gaps = _state_arrays(st)
    sidx = {(a, b): i for i, (a, b) in enumerate(zip(st.manager, st.block, strict=True))}
    ls_codes, ls_uni = pd.factorize(pd.MultiIndex.from_arrays([m.league, m.season]), sort=True)
    clubs = sorted(set(m.home_team) | set(m.away_team))
    cidx = {c: i for i, c in enumerate(clubs)}
    club_lg = pd.concat([m[["home_team", "league"]].set_axis(["c", "l"], axis=1),
                         m[["away_team", "league"]].set_axis(["c", "l"], axis=1)]).drop_duplicates("c")
    lg_codes = pd.factorize(club_lg.set_index("c").reindex(clubs).l, sort=True)[0]
    cs_h, cs_a, cs, fill = _club_values(m, cidx)
    logv = cs.v.to_numpy(float)
    luck = (m.home_luck.fillna(0) - m.away_luck.fillna(0)).to_numpy()
    return ModelData(
        y=m.y.to_numpy(float), w=m.w.to_numpy(float), ls_idx=ls_codes, ls_labels=list(ls_uni),
        home_club=m.home_team.map(cidx).to_numpy(), away_club=m.away_team.map(cidx).to_numpy(),
        club_league=lg_codes, clubs=clubs,
        home_state=np.array([sidx[k] for k in zip(m.home_manager, m.block, strict=True)]),
        away_state=np.array([sidx[k] for k in zip(m.away_manager, m.block, strict=True)]),
        states=st, state_matrix=mat, gap_years=gaps, managers=sorted(st.manager.unique()),
        home_ten=_tenure_cat(m.home_tenure), away_ten=_tenure_cat(m.away_tenure), luck_diff=luck,
        cs_home=cs_h, cs_away=cs_a, logv_obs=logv, logv_fill=fill,
        cs_table=cs.rename(columns={"l": "league", "s": "season", "c": "club"})[
            ["league", "season", "club", "club_idx"]],
        logv_club_mean=cs.club_mean.to_numpy(float),
        intl_state=np.array([sidx[k] for k in zip(intl.manager, intl.block, strict=True)], dtype=int),
        intl_r=intl.r.to_numpy(float), intl_w=intl.w.to_numpy(float),
        dropped={"no_manager": int(no_mgr.sum()),
                 "unknown_tenure": int(m.home_tenure.isna().sum() + m.away_tenure.isna().sum()),
                 "missing_value": int(np.isnan(logv).sum())},
    )


def _ou_states(d: ModelData, tau: pt.TensorVariable, gamma: pt.TensorVariable) -> pt.TensorVariable:
    """Non-centred OU deviations eta for every manager state (exact discretisation)."""
    n_states = len(d.states)
    z = pm.Normal("eta_z", 0, 1, shape=n_states)
    zp = pt.concatenate([z, pt.zeros(1)])[d.state_matrix]          # (n_mgr, J)
    rho = pt.exp(-gamma * d.gap_years)
    cols = [tau * zp[:, 0]]
    for j in range(1, d.state_matrix.shape[1]):
        r = rho[:, j]
        cols.append(r * cols[-1] + pt.sqrt(1 - r**2) * tau * zp[:, j])
    eta_mat = pt.stack(cols, axis=1)
    return eta_mat[d.states.m_idx.to_numpy(), d.states.pos.to_numpy()]


def build_model(d: ModelData) -> pm.Model:
    coords = {"manager": d.managers, "club": d.clubs, "state": np.arange(len(d.states)),
              "league_season": [f"{a} {b}" for a, b in d.ls_labels], "tenure": ["year1", "year2"]}
    n_league = int(d.club_league.max()) + 1
    with pm.Model(coords=coords) as model:
        sigma = pm.HalfNormal("sigma", 2.0)
        h0 = pm.Normal("h0", 0.3, 0.3)
        sigma_h = pm.HalfNormal("sigma_h", 0.2)
        h = pm.Deterministic("h", h0 + sigma_h * pm.Normal("h_z", 0, 1, dims="league_season"),
                             dims="league_season")
        sigma_kappa = pm.HalfNormal("sigma_kappa", 0.5)
        k_raw = sigma_kappa * pm.Normal("kappa_z", 0, 1, dims="club")
        lg_mean = pt.zeros(n_league)
        lg_mean = pt.inc_subtensor(lg_mean[d.club_league], k_raw) / np.bincount(d.club_league)
        kappa = pm.Deterministic("kappa", k_raw - lg_mean[d.club_league], dims="club")
        b_value = pm.Normal("b_value", 0, 0.5)
        b_between = pm.Normal("b_between", 0, 0.5)
        logv = pt.as_tensor(np.nan_to_num(d.logv_obs))
        miss = np.flatnonzero(np.isnan(d.logv_obs))
        if len(miss):
            logv = pt.set_subtensor(logv[miss], pm.Normal("logv_missing", d.logv_fill, 0.7))
        cm = d.logv_club_mean - d.logv_club_mean.mean()
        strength = pm.Deterministic(
            "strength",
            kappa[d.cs_table.club_idx.to_numpy()] + b_between * cm
            + b_value * (logv - d.logv_club_mean),
        )
        sigma_mu = pm.HalfNormal("sigma_mu", 0.3)
        mu = pm.Deterministic("mu", sigma_mu * pm.Normal("mu_z", 0, 1, dims="manager"), dims="manager")
        tau = pm.HalfNormal("tau", 0.2)
        gamma = pm.LogNormal("gamma", np.log(0.5), 1.0)
        pm.Deterministic("half_life", np.log(2) / gamma)
        eta = _ou_states(d, tau, gamma)
        theta = pm.Deterministic("theta", mu[d.states.m_idx.to_numpy()] + eta, dims="state")
        delta = pm.Normal("delta", 0, 0.2, dims="tenure")
        delta_full = pt.concatenate([delta, pt.zeros(1)])
        beta = pm.Normal("beta", 0, 0.25)
        mean = (h[d.ls_idx]
                + strength[d.cs_home] - strength[d.cs_away]
                + theta[d.home_state] - theta[d.away_state]
                + delta_full[d.home_ten] - delta_full[d.away_ten]
                + beta * d.luck_diff)
        pm.Normal("y", mean, sigma / np.sqrt(d.w), observed=d.y)
        if len(d.intl_r):
            lam_int = pm.Normal("lambda_int", 0.5, 0.5)
            sigma_int = pm.HalfNormal("sigma_int", 1.0)
            pm.Normal("r", lam_int * theta[d.intl_state], sigma_int / np.sqrt(d.intl_w), observed=d.intl_r)
    return model
