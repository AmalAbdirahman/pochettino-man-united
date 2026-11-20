"""Team-match shots and corners from football-data.co.uk, with managers from the match table.

python -m pochettino_fit.fd_stats   -> data/processed/fd_team_matches.parquet (local)

One provider for every season (2014-15 to the data cut-off, 2026-27 included), so the style model's
volume metrics cover Carrick's 2026-27 matches. Rows are matched to matches.parquet on league, season,
home and away club (unique within a league season); unmatched rows are dropped and counted.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from pochettino_fit import sources as S
from pochettino_fit.config import ROOT, load_config
from pochettino_fit.fetch import football_data_season_code, seasons_from

STATS = ["shots", "sot", "corners"]
KEYS = ["league", "season", "home_team", "away_team"]


def load_all(raw: Path, seasons: list[str]) -> pd.DataFrame:
    parts = []
    for s in seasons:
        for lg in S.BIG5:
            if (raw / "football_data" / f"{lg}_{football_data_season_code(s)}.csv").exists():
                d = S.load_football_data(raw, lg, football_data_season_code(s))
                parts.append(d.assign(season=s))     # by file: 2019-20 ran into July 2020
    return pd.concat([p.dropna(axis=1, how="all") for p in parts], ignore_index=True)


def attach_ids(fd: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    sub = names[names.source == "football_data"]
    to_tm = dict(zip(sub.alias, sub.tm_id, strict=True))
    fd = fd.assign(home_team=fd.home.map(to_tm), away_team=fd.away.map(to_tm))
    missing = sorted(set(fd.home[fd.home_team.isna()]) | set(fd.away[fd.away_team.isna()]))
    if missing:
        raise ValueError(f"unmapped football-data names: {missing}")
    return fd.astype({"home_team": int, "away_team": int})


def join_managers(fd: pd.DataFrame, m: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cols = KEYS + ["date", "home_manager", "away_manager", "match_id"]
    stats = [f"{side}_{x}" for side in ("home", "away") for x in STATS]
    j = fd[KEYS + stats].merge(m[cols], on=KEYS, how="inner", validate="one_to_one")
    return j, len(fd) - len(j)


def long_form(j: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match: own stats as *_for, opponent's as *_against."""
    out = []
    for me, op in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({"match_id": j.match_id, "league": j.league, "season": j.season, "date": j.date,
                          "team": j[f"{me}_team"], "opponent": j[f"{op}_team"],
                          "manager": j[f"{me}_manager"], "is_home": me == "home"})
        for x in STATS:
            d[f"{x}_for"] = pd.to_numeric(j[f"{me}_{x}"], errors="coerce")
            d[f"{x}_against"] = pd.to_numeric(j[f"{op}_{x}"], errors="coerce")
        out.append(d)
    out_df = pd.concat(out, ignore_index=True)
    return out_df.sort_values(["date", "match_id", "is_home"]).reset_index(drop=True)


def build(cfg: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    raw = ROOT / cfg["paths"]["raw"]
    seasons = seasons_from(cfg["scope"]["first_season"], cfg["appointment"]["season"])
    names = pd.read_csv(ROOT / "data" / "team_names.csv")
    m = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "matches.parquet")
    fd = attach_ids(load_all(raw, seasons), names)
    fd = fd[fd.date <= pd.Timestamp(cfg["features"]["data_cutoff"])]
    j, dropped = join_managers(fd, m)
    return long_form(j), dropped


def main() -> None:
    cfg = load_config()
    t, dropped = build(cfg)
    path = ROOT / cfg["paths"]["processed"] / "fd_team_matches.parquet"
    t.to_parquet(path, index=False)
    print(f"{len(t)} team-match rows, {t.match_id.nunique()} matches, {dropped} football-data rows "
          f"without a match-table row; {t.date.min().date()} to {t.date.max().date()}; "
          f"missing shots {t.shots_for.isna().mean():.3%}, corners {t.corners_for.isna().mean():.3%}")
    print("written", path)


if __name__ == "__main__":
    main()
