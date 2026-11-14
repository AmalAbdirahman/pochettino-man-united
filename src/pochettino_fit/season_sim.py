"""Match and season simulation (methods section 5c), shared by forecast.py and backtest.py.

For each draw and fixture: expected npxG from the attack/defence model, a mean-preserving log-normal
match shock, then goals ~ Poisson(f * npxG + penalty rate). Random numbers are drawn once and passed in,
so two runs (baseline and Pochettino) use common random numbers and their difference has little
simulation noise.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Noise:
    """Common random numbers for n draws and F fixtures."""
    z_home: np.ndarray
    z_away: np.ndarray
    u_home: np.ndarray
    u_away: np.ndarray
    tie: np.ndarray           # (n, teams) tie-breaker

    @classmethod
    def make(cls, n: int, n_fix: int, n_teams: int, rng: np.random.Generator) -> "Noise":
        return cls(rng.normal(size=(n, n_fix)), rng.normal(size=(n, n_fix)), rng.random((n, n_fix)),
                   rng.random((n, n_fix)), rng.random((n, n_teams)))


def fill_npxg(m: pd.DataFrame, pen_xg_rate: float) -> pd.DataFrame:
    """Where only total xG is known (football-data 2026-27), npxG = xG minus the league's penalty xG rate."""
    m = m.copy()
    for side in ("home", "away"):
        est = (m[f"{side}_xg_fd"] - pen_xg_rate).clip(lower=0.05)
        m[f"{side}_npxg"] = m[f"{side}_npxg"].fillna(est)
    return m


def goal_calibration(m: pd.DataFrame, w: np.ndarray, pen_rate: float) -> float:
    """k = recency-weighted non-penalty goals / npxG over both sides of every match (methods risk 5)."""
    num = den = 0.0
    for side in ("home", "away"):
        npg = m[f"{side}_goals"] - m[f"{side}_pen_goals"].fillna(pen_rate)
        ok = npg.notna() & m[f"{side}_npxg"].notna()
        num += float(np.sum(w[ok.to_numpy()] * npg[ok]))
        den += float(np.sum(w[ok.to_numpy()] * m.loc[ok, f"{side}_npxg"]))
    return num / den


def remaining_fixtures(played: pd.DataFrame, teams: list[int]) -> pd.DataFrame:
    """Every ordered (home, away) pair of a double round robin that has not been played yet."""
    done = set(zip(played.home_team, played.away_team, strict=True))
    rows = [(h, a) for h in teams for a in teams if h != a and (h, a) not in done]
    return pd.DataFrame(rows, columns=["home_team", "away_team"])


def goals(lam: np.ndarray, z: np.ndarray, u: np.ndarray, s: float, f: np.ndarray | float,
          pen_rate: float, k: float = 1.0) -> np.ndarray:
    """Goals from expected npxG lam with shock exp(s z - s^2/2), league calibration k (non-penalty goals
    per npxG), finishing f and penalties. s = 0 switches the shock off."""
    rate = k * np.asarray(f) * lam * np.exp(s * z - s**2 / 2) + pen_rate
    return stats.poisson.ppf(u, rate)


def points(gf: np.ndarray, ga: np.ndarray) -> np.ndarray:
    return np.where(gf > ga, 3, np.where(gf == ga, 1, 0))


def one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(idx), n))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def season_table(gh: np.ndarray, ga: np.ndarray, hi: np.ndarray, ai: np.ndarray, n_teams: int,
                 base: dict[str, np.ndarray], tie: np.ndarray) -> dict[str, np.ndarray]:
    """Final points, goal difference and position for every draw (position 1 = top)."""
    H, A = one_hot(hi, n_teams), one_hot(ai, n_teams)
    pts = points(gh, ga) @ H + points(ga, gh) @ A + base["pts"]
    gf = gh @ H + ga @ A + base["gf"]
    gd = gf - (ga @ H + gh @ A) - base["ga"]
    key = pts * 1e6 + (gd + 500) * 1e3 + gf + tie * 0.5
    pos = np.empty_like(pts, dtype=int)
    order = np.argsort(-key, axis=1)
    np.put_along_axis(pos, order, np.arange(1, n_teams + 1)[None, :].repeat(len(pts), 0), axis=1)
    return {"pts": pts, "gd": gd, "pos": pos, "rest_pts": points(gh, ga) @ H + points(ga, gh) @ A}


def current_table(played: pd.DataFrame, teams: list[int]) -> dict[str, np.ndarray]:
    pos = {t: i for i, t in enumerate(teams)}
    out = {k: np.zeros(len(teams)) for k in ("pts", "gf", "ga")}
    for r in played.itertuples():
        h, a = pos[r.home_team], pos[r.away_team]
        out["pts"][h] += points(r.home_goals, r.away_goals)
        out["pts"][a] += points(r.away_goals, r.home_goals)
        out["gf"][h] += r.home_goals
        out["gf"][a] += r.away_goals
        out["ga"][h] += r.away_goals
        out["ga"][a] += r.home_goals
    return out
