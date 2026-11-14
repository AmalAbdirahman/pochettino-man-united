"""Log-link attack and defence model for npxG (methods section 5a).

log E[npxG_{i vs j}] = c + h * home + a_i - d_j

Fitted as a recency-weighted quasi-Poisson GLM on one league's recent matches (two long rows per match:
each side's npxG is its own observation; the attack/defence model is not the mirrored-row problem of
section 1 because each row has its own outcome). Parameter uncertainty: multivariate normal from the
GLM covariance, scaled by the Pearson dispersion. Manager effects are not in this model; forecast.py
adds United's on top (section 5b).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class Strength:
    teams: list[int]
    coef: np.ndarray          # [c, h, a_1..a_{n-1}, d_1..d_{n-1}] (team 0 is the reference: a = d = 0)
    cov: np.ndarray
    shock_sd: float           # s: sd of the log match-level xG shock

    def draws(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Parameter draws: c, h (n,), attack and defence (n, teams)."""
        x = rng.multivariate_normal(self.coef, self.cov, size=n, method="cholesky")
        k = len(self.teams) - 1
        zero = np.zeros((n, 1))
        return {"c": x[:, 0], "h": x[:, 1],
                "a": np.hstack([zero, x[:, 2:2 + k]]), "d": np.hstack([zero, x[:, 2 + k:]])}

    def index(self, team: int) -> int:
        return self.teams.index(team)


def long_rows(m: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: (attacking team, defending team, home flag, npxG, date)."""
    home = pd.DataFrame({"att": m.home_team, "dfn": m.away_team, "home": 1.0, "npxg": m.home_npxg,
                         "date": m.date})
    away = pd.DataFrame({"att": m.away_team, "dfn": m.home_team, "home": 0.0, "npxg": m.away_npxg,
                         "date": m.date})
    return pd.concat([home, away], ignore_index=True).dropna(subset=["npxg"])


def design(rows: pd.DataFrame, teams: list[int]) -> np.ndarray:
    n, k = len(rows), len(teams) - 1
    pos = {t: i for i, t in enumerate(teams)}
    x = np.zeros((n, 2 + 2 * k))
    x[:, 0], x[:, 1] = 1.0, rows.home.to_numpy()
    for r, (a, d) in enumerate(zip(rows.att.map(pos), rows.dfn.map(pos), strict=True)):
        if a > 0:
            x[r, 1 + a] = 1.0
        if d > 0:
            x[r, 1 + k + d] = -1.0
    return x


def recency(dates: pd.Series, ref: pd.Timestamp, half_life_days: float) -> np.ndarray:
    age = (ref - pd.to_datetime(dates)).dt.days.clip(lower=0).to_numpy(float)
    return 0.5 ** (age / half_life_days)


def shock_sd(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    """s with E[(y - mu)^2 / mu^2] = exp(s^2) - 1 (actual = expected * exp(zeta))."""
    r2 = np.average(((y - mu) / mu) ** 2, weights=w)
    return float(np.sqrt(np.log1p(r2)))


def fit_strength(matches: pd.DataFrame, ref: pd.Timestamp, half_life_days: float,
                 teams: list[int] | None = None) -> Strength:
    rows = long_rows(matches)
    rows = rows[rows.date < ref]
    teams = teams or sorted(set(rows.att) | set(rows.dfn))
    rows = rows[rows.att.isin(teams) & rows.dfn.isin(teams)]
    w = recency(rows.date, ref, half_life_days)
    x, y = design(rows, teams), np.clip(rows.npxg.to_numpy(float), 1e-3, None)
    res = sm.GLM(y, x, family=sm.families.Poisson(), var_weights=w).fit(scale="X2")
    return Strength(teams, np.asarray(res.params), np.asarray(res.cov_params()),
                    shock_sd(y, res.fittedvalues, w))
