"""Team name matching across sources.

Canonical team = Transfermarkt club id (tm_id) and name. Aliases from other sources are
matched to TM games played on the same day (+/- 1) in the same league with the same score;
each alias takes the TM club it co-occurs with most often. Clubs never in the TM big-five
data come from the hand file data/team_names_manual.csv.
"""

from pathlib import Path

import pandas as pd

KEYS = ["league", "home_goals", "away_goals"]


def _candidate_pairs(src: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """Rows (alias, tm_id) for every source match that lines up with a TM game."""
    out = []
    for shift in (-1, 0, 1):
        s = src.assign(date=src.date + pd.Timedelta(days=shift))
        j = s.merge(tm, on=KEYS + ["date"], suffixes=("", "_tm"))
        out.append(pd.DataFrame({"alias": j.home, "tm_id": j.home_tm_id, "tm_name": j.home_tm_name}))
        out.append(pd.DataFrame({"alias": j.away, "tm_id": j.away_tm_id, "tm_name": j.away_tm_name}))
    return pd.concat(out, ignore_index=True)


def match_aliases(
    src: pd.DataFrame, tm: pd.DataFrame, min_votes: int, min_share: float
) -> tuple[pd.DataFrame, list[str]]:
    """Map each alias in src (columns league, date, home, away, goals) to a TM club.

    Returns the mapping and the aliases that could not be matched confidently.
    """
    pairs = _candidate_pairs(src, tm)
    votes = pairs.groupby(["alias", "tm_id", "tm_name"]).size().rename("votes").reset_index()
    overlap = src[src.date <= tm.date.max() + pd.Timedelta(days=1)]
    played = pd.concat([overlap.home, overlap.away]).value_counts()
    votes["share"] = votes.votes / votes.alias.map(played)
    best = votes.sort_values("votes", ascending=False).drop_duplicates("alias")
    ok = best[(best.votes >= min_votes) & (best.share >= min_share)]
    all_aliases = set(src.home) | set(src.away)
    failed = sorted(all_aliases - set(ok.alias))
    return ok[["alias", "tm_id", "tm_name", "votes", "share"]].reset_index(drop=True), failed


def check_mapping(names: pd.DataFrame) -> list[str]:
    """Problems in a full mapping table (source, alias, tm_id, tm_name)."""
    problems = []
    dup = names[names.duplicated(["source", "alias"], keep=False)]
    if len(dup):
        problems.append(f"alias mapped twice: {sorted(set(dup.alias))}")
    if names.tm_id.isna().any():
        problems.append("missing tm_id")
    return problems


def load_manual(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")
