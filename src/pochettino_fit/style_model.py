"""Style model (methods section 4a), one metric at a time, on Understat or football-data team-match rows.

z_k = lambda[league, season] + kappa[club] + theta[manager] + eps,  eps ~ N(0, sigma^2 / w_k)

z is the metric standardised over all rows (so effects are in team-match sd units and are never rescaled
by their own spread). Ratio metrics are per-match ratios weighted by their denominator; PPDA is logged.
kappa and theta are partially pooled (non-centred). Managers are static here: no OU drift, no tenure.
"""

import os
from dataclasses import dataclass

from pochettino_fit.config import load_config

os.environ.setdefault("PYTENSOR_FLAGS", f"mode={load_config()['simulation']['sampling']['mode']}")

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402


@dataclass(frozen=True)
class StyleMetric:
    name: str
    label: str
    num: str
    den: str | None = None
    log: bool = False
    better: str = ""          # "lower"/"higher" for weakness metrics (from United's point of view)
    source: str = "understat"  # "understat" (archive to 2026-04-18) or "fd" (football-data, to the cut-off)


METRICS = [
    StyleMetric("shots_for", "Shots for per match", "shots_for", source="fd"),
    StyleMetric("shots_against", "Shots against per match", "shots_against", better="lower", source="fd"),
    StyleMetric("corners_for", "Corners for per match", "corners_for", source="fd"),
    StyleMetric("corners_against", "Corners against per match", "corners_against", source="fd"),
    StyleMetric("npxg_per_shot_for", "npxG per shot for (level or one-goal states)", "npxg_close_for",
                "shots_close_for"),
    StyleMetric("npxg_per_shot_against", "npxG per shot against (level or one-goal states)",
                "npxg_close_against", "shots_close_against", better="lower"),
    StyleMetric("setpiece_share_for", "Set-piece share of npxG for", "setpiece_npxg_for", "npxg_for"),
    StyleMetric("setpiece_npxg_against", "Set-piece npxG against per match", "setpiece_npxg_against",
                better="lower"),
    StyleMetric("npxg_against", "npxG against per match", "npxg_against", better="lower"),
    StyleMetric("ppda", "PPDA (lower = more pressing)", "ppda_for", log=True),
    StyleMetric("deep_for", "Deep completions for per match", "deep_for"),
    StyleMetric("deep_against", "Deep completions against per match", "deep_against", better="lower"),
]
BY_NAME = {m.name: m for m in METRICS}


@dataclass
class StyleData:
    z: np.ndarray
    w: np.ndarray
    ls: np.ndarray
    club: np.ndarray
    mgr: np.ndarray
    clubs: list[int]
    managers: list[str]
    centre: float
    scale: float
    rows: pd.DataFrame        # the rows used (for correlations and change detection)


def raw_values(t: pd.DataFrame, m: StyleMetric) -> tuple[pd.Series, pd.Series]:
    """Per-match metric value and its weight (denominator relative to its mean, else 1)."""
    if m.den is None:
        x, w = t[m.num].astype(float), pd.Series(1.0, index=t.index)
    else:
        den = t[m.den].astype(float)
        x, w = t[m.num] / den.where(den > 0), den / den[den > 0].mean()
    if m.log:
        x = np.log(x.where(x > 0))
    return x, w


def prepare(t: pd.DataFrame, m: StyleMetric) -> StyleData:
    x, w = raw_values(t, m)
    ok = x.notna() & np.isfinite(x) & (w > 0) & t.manager.notna()
    r, x, w = t[ok].reset_index(drop=True), x[ok].to_numpy(), w[ok].to_numpy()
    centre, scale = float(np.average(x, weights=w)), float(np.sqrt(np.cov(x, aweights=w)))
    ls = pd.factorize(r.league + " " + r.season)[0]
    club, clubs = pd.factorize(r.team, sort=True)
    mgr, managers = pd.factorize(r.manager, sort=True)
    return StyleData((x - centre) / scale, w, ls, club, mgr, list(clubs), list(managers), centre, scale,
                     r.assign(z=(x - centre) / scale))


def build_model(d: StyleData) -> pm.Model:
    with pm.Model() as model:
        lam = pm.Normal("lam", 0.0, 1.0, shape=int(d.ls.max()) + 1)
        s_kappa = pm.HalfNormal("sigma_kappa", 0.5)
        s_theta = pm.HalfNormal("sigma_theta", 0.5)
        sigma = pm.HalfNormal("sigma", 1.0)
        kappa = pm.Deterministic("kappa", s_kappa * pm.Normal("kappa_raw", 0, 1, shape=len(d.clubs)))
        theta = pm.Deterministic("theta", s_theta * pm.Normal("theta_raw", 0, 1, shape=len(d.managers)))
        mu = lam[d.ls] + kappa[d.club] + theta[d.mgr]
        pm.Normal("z", mu, sigma / np.sqrt(d.w), observed=d.z)
    return model


def sample(d: StyleData, s: dict, seed: int) -> dict[str, np.ndarray]:
    """Posterior draws, flattened over chains: theta (draws, managers), kappa (draws, clubs), scalars."""
    with build_model(d):
        idata = pm.sample(draws=s["draws"], tune=s["tune"], chains=s["chains"], cores=s["cores"],
                          target_accept=s["target_accept"], random_seed=seed,
                          compile_kwargs={"mode": s["mode"]}, progressbar=False)
    post = idata.posterior

    def flat(v: str) -> np.ndarray:
        a = post[v].values
        return a.reshape(-1, *a.shape[2:])

    rhat = max(float(np.max(az.rhat(idata, var_names=[v])[v]))
               for v in ("sigma_kappa", "sigma_theta", "sigma"))
    return {"theta": flat("theta"), "kappa": flat("kappa"), "lam": flat("lam"),
            "sigma_kappa": flat("sigma_kappa"),
            "sigma_theta": flat("sigma_theta"), "sigma": flat("sigma"), "max_rhat": np.array(rhat)}
