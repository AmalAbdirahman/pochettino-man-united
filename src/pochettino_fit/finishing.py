"""Finishing skill of the current United squad and the team multiplier f_U (methods section 3).

G_p - X_p = phi_p X_p + e_p, Var(e_p) = k X_p, prior phi_p ~ N(0, sd^2).
The raw estimate (G - X) / X has variance k / X, so the posterior is the usual normal-normal update.
f_U = 1 + sum_p pi_p phi_p, with pi_p each player's expected share of United's npxG.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pochettino_fit.config import ROOT


def load_squad(path: Path) -> pd.DataFrame:
    sq = pd.read_csv(path, comment="#", dtype={"understat_id": "Int64"})
    return sq[["number", "player", "position", "understat_id"]]


def phi_posterior(goals: np.ndarray, npxg: np.ndarray, k: float,
                  prior_sd: float) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and sd of phi for each player. npxg = 0 returns the prior."""
    goals, npxg = np.asarray(goals, float), np.asarray(npxg, float)
    data_prec = npxg / k                       # 1 / Var(phi_hat) = X / k
    resid = goals - npxg                       # phi_hat * data_prec = (G - X) / k
    prec = 1.0 / prior_sd**2 + data_prec
    return (resid / k) / prec, 1.0 / np.sqrt(prec)


def career_totals(shots: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Career non-penalty goals and npxG per Understat player id (big-five leagues)."""
    s = shots[shots.player_id.isin(ids.dropna().astype(int))]
    out = s.groupby("player_id").agg(goals=("goal", "sum"), npxg=("npxg", "sum"), shots=("npxg", "size"))
    return out.reset_index()


def season_rates(shots: pd.DataFrame, season: str) -> pd.DataFrame:
    """npxG per club league match in one season: player npxG / matches played by the club(s) he shot for."""
    s = shots[shots.season == season]
    team_matches = s.groupby("team").understat_id.nunique()
    per = s.groupby(["player_id", "team"]).npxg.sum().reset_index()
    per["club_matches"] = per.team.map(team_matches)
    agg = per.groupby("player_id")[["npxg", "club_matches"]].sum()
    return pd.DataFrame({"player_id": agg.index, "rate": (agg.npxg / agg.club_matches).values})


def npxg_shares(rate: pd.Series) -> pd.Series:
    """Expected share of team npxG, proportional to each player's recent rate; zero rate -> zero share."""
    r = rate.fillna(0.0).clip(lower=0.0)
    total = r.sum()
    return r / total if total > 0 else r


def team_multiplier(pi: np.ndarray, mean: np.ndarray, sd: np.ndarray, level: float) -> dict[str, float]:
    """f_U = 1 + sum pi phi with independent normal phi: mean, sd and central interval."""
    m = 1.0 + float(np.sum(pi * mean))
    s = float(np.sqrt(np.sum(pi**2 * sd**2)))
    z = stats.norm.ppf(0.5 + level / 2)
    return {"mean": m, "sd": s, "lo": m - z * s, "hi": m + z * s, "level": level}


def finishing(cfg: dict[str, Any], shots: pd.DataFrame) -> dict[str, Any]:
    f = cfg["finishing"]
    squad = load_squad(ROOT / f["squad_file"])
    car = career_totals(shots, squad.understat_id)
    rates = season_rates(shots, f["share_season"])
    t = squad.merge(car, left_on="understat_id", right_on="player_id", how="left")
    t = t.merge(rates, on="player_id", how="left").drop(columns="player_id")
    t[["goals", "npxg", "shots"]] = t[["goals", "npxg", "shots"]].fillna(0)
    t["phi_raw"] = np.where(t.npxg > 0, (t.goals - t.npxg) / t.npxg.where(t.npxg > 0), np.nan)
    t["phi_mean"], t["phi_sd"] = phi_posterior(t.goals.values, t.npxg.values, f["shot_noise_factor"],
                                               f["phi_prior_sd"])
    t["share"] = npxg_shares(t.rate)
    fu = team_multiplier(t.share.values, t.phi_mean.values, t.phi_sd.values, f["interval"])
    players = t.sort_values("share", ascending=False)
    return {
        "f_U": fu,
        "share_season": f["share_season"],
        "players_without_history": players.loc[players.npxg == 0, "player"].tolist(),
        "players": [_player_row(r) for r in players.itertuples()],
    }


def _player_row(r: Any) -> dict[str, Any]:
    def num(x: float, nd: int = 4) -> float | None:
        return None if pd.isna(x) else round(float(x), nd)

    return {"player": r.player, "position": r.position, "goals_np": int(r.goals), "npxg": num(r.npxg, 2),
            "shots": int(r.shots), "phi_raw": num(r.phi_raw), "phi_mean": num(r.phi_mean),
            "phi_sd": num(r.phi_sd), "npxg_share": num(r.share)}
