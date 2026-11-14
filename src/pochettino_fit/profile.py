"""United finishing and style profile (prompt 4; methods sections 3 and 4).

python -m pochettino_fit.profile
  writes data/processed/profile_local.json (everything, including FotMob-based values)
  and docs/data/profile.json (publishable blocks only: never FotMob-derived values).

Each metric is a ratio of recency-weighted sums, sum(w * num) / sum(w * den), with den = 1 for
per-match averages. Weights halve every half_life_days before the reference date. Intervals come
from a bootstrap over matches; percentiles compare United with every team in the same league-season
under the same weights (Understat block only: there is no league-wide FotMob data).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pochettino_fit import sources as S
from pochettino_fit.config import ROOT, load_config
from pochettino_fit.finishing import finishing


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    num: str
    den: str | None = None
    game_state: str = "all"


CLOSE = "level or one goal"
SHARED = [
    Metric("npxg_for_pm", "npxG for per match", "npxg_for"),
    Metric("npxg_against_pm", "npxG against per match", "npxg_against"),
    Metric("shots_for_pm", "Shots for per match", "shots_for"),
    Metric("shots_against_pm", "Shots against per match", "shots_against"),
    Metric("setpiece_share_for", "Set-piece share of npxG for", "setpiece_npxg_for", "npxg_for"),
    Metric("setpiece_share_against", "Set-piece share of npxG against", "setpiece_npxg_against",
           "npxg_against"),
]
UNDERSTAT = SHARED + [
    Metric("npxg_per_shot_for_close", "npxG per shot for", "npxg_close_for", "shots_close_for", CLOSE),
    Metric("npxg_per_shot_against_close", "npxG per shot against", "npxg_close_against",
           "shots_close_against", CLOSE),
    Metric("ppda", "PPDA (own pressing; lower = more pressing)", "ppda_for"),
    Metric("deep_for_pm", "Deep completions for per match", "deep_for"),
    Metric("deep_against_pm", "Deep completions against per match", "deep_against"),
]
FOTMOB = SHARED + [
    Metric("possession", "Possession %", "possession_for"),
    Metric("field_tilt", "Field tilt (share of accurate passes in the opposition half)", "opp_half_for",
           "opp_half_total"),
    Metric("npxg_per_shot_for", "npxG per shot for", "npxg_for", "shots_for"),
    Metric("npxg_per_shot_against", "npxG per shot against", "npxg_against", "shots_against"),
    Metric("big_chances_for_pm", "Big chances for per match", "big_chances_for"),
    Metric("big_chances_against_pm", "Big chances against per match", "big_chances_against"),
]


def recency_weights(dates: pd.Series, ref: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Weights 0.5 ** (age / half-life), normalised to sum to 1."""
    age = (ref - pd.to_datetime(dates)).dt.days.to_numpy(float)
    w = 0.5 ** (np.clip(age, 0, None) / half_life_days)
    return w / w.sum()


def weighted_ratio(df: pd.DataFrame, m: Metric, w: np.ndarray) -> float:
    den = df[m.den].to_numpy(float) if m.den else np.ones(len(df))
    return float(np.sum(w * df[m.num].to_numpy(float)) / np.sum(w * den))


def bootstrap(df: pd.DataFrame, m: Metric, w: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    num = df[m.num].to_numpy(float)
    den = df[m.den].to_numpy(float) if m.den else np.ones(len(df))
    idx = rng.integers(0, len(df), size=(reps, len(df)))
    return (w[idx] * num[idx]).sum(1) / (w[idx] * den[idx]).sum(1)


def percentile(value: float, others: np.ndarray) -> tuple[float, int]:
    """Percentile (0 = lowest, 100 = highest) of value among all teams, and its rank from the top."""
    allv = np.asarray(others, float)
    below = np.sum(allv < value) + 0.5 * (np.sum(allv == value) - 1)
    return float(100 * below / (len(allv) - 1)), int(np.sum(allv > value) + 1)


def summarise(df: pd.DataFrame, metrics: list[Metric], w: np.ndarray, pc: dict[str, Any],
              league: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    rng = np.random.default_rng(pc["seed"])
    a = (1 - pc["interval"]) / 2
    out = []
    for m in metrics:
        if df[m.num].isna().any() or (m.den and df[m.den].isna().any()):
            continue
        v = weighted_ratio(df, m, w)
        lo, hi = np.quantile(bootstrap(df, m, w, pc["bootstrap"], rng), [a, 1 - a])
        row = {"metric": m.name, "label": m.label, "game_state": m.game_state, "value": v,
               "lo": float(lo), "hi": float(hi)}
        if league is not None:
            vals = league_values(league, m, pc)
            row["percentile"], row["rank"] = percentile(v, vals)
            row["n_teams"] = len(vals)
        out.append(row)
    return out


def league_values(league: pd.DataFrame, m: Metric, pc: dict[str, Any]) -> np.ndarray:
    ref = pd.Timestamp(pc["reference_date"])
    return np.array([weighted_ratio(g, m, recency_weights(g.date, ref, pc["half_life_days"]))
                     for _, g in league.groupby("team")])


def fotmob_frame(raw: Path, matches: pd.DataFrame, team: int, season: str) -> pd.DataFrame:
    """FotMob United rows with dates from the match table (same date-order matching as dataset.py)."""
    fm = S.load_fotmob_united(raw).sort_values("gameweek").reset_index(drop=True)
    utd = matches[(matches.season == season) & (matches.xg_source == "fotmob")
                  & ((matches.home_team == team) | (matches.away_team == team))].sort_values("date")
    if len(utd) != len(fm):
        raise ValueError(f"FotMob rows {len(fm)} vs match table {len(utd)}")
    fm["date"] = utd.date.to_numpy()
    fm = fm.rename(columns={"xg_set_for": "setpiece_npxg_for", "xg_set_against": "setpiece_npxg_against",
                            "acc_passes_opp_half_for": "opp_half_for"})
    fm["opp_half_total"] = fm.opp_half_for + fm.acc_passes_opp_half_against
    return fm


def block(source: str, season: str, df: pd.DataFrame, metrics: list[Metric], pc: dict[str, Any],
          publishable: bool, league: pd.DataFrame | None = None) -> dict[str, Any]:
    w = recency_weights(df.date, pd.Timestamp(pc["reference_date"]), pc["half_life_days"])
    return {"source": source, "season": season, "publishable": publishable, "matches": len(df),
            "first": str(pd.Timestamp(df.date.min()).date()), "last": str(pd.Timestamp(df.date.max()).date()),
            "metrics": summarise(df, metrics, w, pc, league)}


def build(cfg: dict[str, Any]) -> dict[str, Any]:
    pc, raw = cfg["profile"], ROOT / cfg["paths"]["raw"]
    proc = ROOT / cfg["paths"]["processed"]
    tm = pd.read_parquet(proc / "team_matches.parquet")
    league = tm[(tm.season == pc["understat_season"]) & (tm.league == pc["understat_league"])]
    us = league[league.team == pc["team_tm_id"]]
    fm = fotmob_frame(raw, pd.read_parquet(proc / "matches.parquet"), pc["team_tm_id"], pc["fotmob_season"])
    shared = list(dict.fromkeys([m.num for m in SHARED] + [m.den for m in SHARED if m.den] + ["date"]))
    both = pd.concat([us[shared], fm[shared]], ignore_index=True)
    blocks = [
        block("understat", pc["understat_season"], us, UNDERSTAT, pc, True, league),
        block("fotmob", pc["fotmob_season"], fm, FOTMOB, pc, False),
        block("understat+fotmob", f"{pc['understat_season']} and {pc['fotmob_season']}", both, SHARED, pc,
              False),
    ]
    return {"reference_date": str(pc["reference_date"]), "half_life_days": pc["half_life_days"],
            "interval": pc["interval"], "finishing": finishing(cfg, S.load_understat_player_shots(raw)),
            "profile": blocks}


def publishable(full: dict[str, Any]) -> dict[str, Any]:
    """Drop every block built from local-only data, and say what was withheld."""
    out = dict(full)
    out["profile"] = [b for b in full["profile"] if b["publishable"]]
    out["withheld"] = [f"{b['source']} {b['season']}: local-only FotMob data, not published"
                       for b in full["profile"] if not b["publishable"]]
    return out


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def print_summary(full: dict[str, Any]) -> None:
    f = full["finishing"]["f_U"]
    print(f"f_U = {f['mean']:.4f}  ({f['level']:.0%}: {f['lo']:.4f} to {f['hi']:.4f})")
    for b in full["profile"]:
        print(f"\n[{b['source']} {b['season']}] {b['matches']} matches, publishable={b['publishable']}")
        for r in b["metrics"]:
            pct = (f"  pct {r['percentile']:.0f} (rank {r['rank']}/{r['n_teams']})"
                   if "percentile" in r else "")
            print(f"  {r['label']:<62} {r['value']:7.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]{pct}")


def main() -> None:
    cfg = load_config()
    full = build(cfg)
    write_json(full, ROOT / cfg["paths"]["processed"] / "profile_local.json")
    write_json(publishable(full), ROOT / cfg["paths"]["docs_data"] / "profile.json")
    print_summary(full)


if __name__ == "__main__":
    main()
