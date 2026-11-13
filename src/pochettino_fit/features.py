"""Pure transforms used to build the match tables (methods.md sections 0-1).

No file access here, so every function can be tested on small synthetic tables.
"""

import unicodedata

import numpy as np
import pandas as pd

SET_PIECES = {"FromCorner", "SetPiece", "DirectFreekick"}
_TRANSLIT = str.maketrans({"æ": "ae", "ø": "o", "ß": "ss", "œ": "oe", "ł": "l", "đ": "d",
                           "Æ": "Ae", "Ø": "O", "Œ": "Oe", "Ł": "L", "Đ": "D"})


def name_key(name: object) -> str:
    """Accent- and case-insensitive key for manager names."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name.translate(_TRANSLIT))
    s = s.encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").split())


# ---------- shots -> match npxG, truncated at the first red card ----------

def add_game_state(shots: pd.DataFrame) -> pd.DataFrame:
    """Add the score difference (home minus away) just before each shot.

    Understat lists an own goal under the side that conceded it.
    """
    s = shots.sort_values(["understat_id", "minute", "shot_id"]).copy()
    home_goal = ((s.result == "Goal") & (s.side == "home")) | (
        (s.result == "OwnGoal") & (s.side == "away")
    )
    away_goal = ((s.result == "Goal") & (s.side == "away")) | (
        (s.result == "OwnGoal") & (s.side == "home")
    )
    diff = home_goal.astype(int) - away_goal.astype(int)
    s["score_diff_before"] = diff.groupby(s.understat_id).cumsum() - diff
    return s


def is_np_shot(shots: pd.DataFrame) -> pd.Series:
    return (shots.situation != "Penalty") & (shots.result != "OwnGoal")


def match_npxg(shots: pd.DataFrame, first_red: pd.Series) -> pd.DataFrame:
    """Per understat_id: full-match npxG, 11 v 11 npxG and minutes, penalty goals.

    first_red: first red-card minute indexed by understat_id (missing = no red card).
    Shots in or after the red-card minute are dropped from the 11 v 11 totals.
    """
    s = shots[is_np_shot(shots)].copy()
    red = s.understat_id.map(first_red)
    s["in_11v11"] = red.isna() | (s.minute < red)
    full = s.pivot_table(index="understat_id", columns="side", values="xg", aggfunc="sum")
    part = s[s.in_11v11].pivot_table(
        index="understat_id", columns="side", values="xg", aggfunc="sum"
    )
    pens = shots[(shots.situation == "Penalty") & (shots.result == "Goal")]
    pen_goals = pens.pivot_table(index="understat_id", columns="side", values="xg", aggfunc="size")
    ids = pd.Index(shots.understat_id.unique(), name="understat_id")
    out = pd.DataFrame(index=ids)
    for side in ("home", "away"):
        out[f"{side}_npxg"] = full.get(side, pd.Series(dtype=float)).reindex(ids).fillna(0.0)
        out[f"{side}_npxg_11v11"] = part.get(side, pd.Series(dtype=float)).reindex(ids).fillna(0.0)
        out[f"{side}_pen_goals"] = (
            pen_goals.get(side, pd.Series(dtype=float)).reindex(ids).fillna(0).astype(int)
        )
    red_ids = first_red.reindex(ids)
    out["minutes_11v11"] = red_ids.clip(upper=90).fillna(90.0).astype(float)
    return out.reset_index()


def outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Add w = 11 v 11 minutes / 90 and y = npxG difference per 90 over those minutes."""
    out = df.copy()
    out["w"] = out.minutes_11v11 / 90.0
    out["y"] = (out.home_npxg_11v11 - out.away_npxg_11v11) / out.w
    return out


def one_row_problems(m: pd.DataFrame) -> list[str]:
    """Checks that each match appears once, from the home side only."""
    problems = []
    if m.match_id.duplicated().any():
        problems.append("duplicate match_id")
    if (m.home_team == m.away_team).any():
        problems.append("team plays itself")
    sides = pd.concat([m[["date", "home_team"]].set_axis(["date", "team"], axis=1),
                       m[["date", "away_team"]].set_axis(["date", "team"], axis=1)])
    if sides.duplicated().any():
        problems.append("a team appears twice on one date")
    pair = m.apply(lambda r: frozenset((r.home_team, r.away_team)), axis=1)
    if pd.DataFrame({"s": m.season, "d": m.date, "p": pair}).duplicated().any():
        problems.append("mirrored rows")
    return problems


# ---------- recent luck ----------

def team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match (one per team) with non-penalty goals and npxG for and against."""
    cols = ["match_id", "date"]
    sides = []
    for me, op in (("home", "away"), ("away", "home")):
        t = matches[cols].copy()
        t["team"] = matches[f"{me}_team"]
        t["npg_for"] = matches[f"{me}_goals"] - matches[f"{me}_pen_goals"]
        t["npg_against"] = matches[f"{op}_goals"] - matches[f"{op}_pen_goals"]
        t["npxg_for"] = matches[f"{me}_npxg"]
        t["npxg_against"] = matches[f"{op}_npxg"]
        sides.append(t)
    long = pd.concat(sides, ignore_index=True)
    long["luck_match"] = (long.npg_for - long.npxg_for) - (long.npg_against - long.npxg_against)
    return long


def recent_luck(long: pd.DataFrame, window: int) -> pd.DataFrame:
    """Mean luck over each team's previous `window` league matches (earlier matches only).

    Matches without xG count towards the window but not the mean; luck_n says how many had xG.
    """
    t = long.sort_values(["team", "date", "match_id"]).copy()
    prev = t.groupby("team").luck_match.shift(1)
    grp = prev.groupby(t.team)
    t["luck"] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
    t["luck_n"] = grp.transform(lambda x: x.rolling(window, min_periods=1).count()).fillna(0)
    t["luck_n"] = t.luck_n.astype(int)
    return t[["match_id", "team", "luck", "luck_n"]]


# ---------- manager spells and tenure ----------

def manager_spells(games: pd.DataFrame) -> pd.DataFrame:
    """Spells from per-club game lists (tm_id, date, manager), all competitions.

    A spell is a run of consecutive games with the same manager. censored = the spell starts
    at the club's first game in the data, so the true start is earlier.
    """
    g = games.assign(mkey=games.manager.map(name_key)).sort_values(["tm_id", "date"])
    new = g.mkey != g.groupby("tm_id").mkey.shift()
    g["spell"] = new.cumsum()
    sp = g.groupby("spell").agg(
        tm_id=("tm_id", "first"), manager=("manager", "first"), mkey=("mkey", "first"),
        start=("date", "min"), last_game=("date", "max"),
    )
    first_seen = g.groupby("tm_id").date.min()
    sp["censored"] = sp.start.eq(sp.tm_id.map(first_seen))
    return sp.reset_index(drop=True)


def tenure_year(date: pd.Series, start: pd.Series, cap: int) -> pd.Series:
    """Year of the spell at each date: 1, 2, ..., cap (cap means cap or more)."""
    years = (date - start).dt.days / 365.25
    return np.floor(years).add(1).clip(lower=1, upper=cap)


# ---------- squad value ----------

def squad_values(
    vals: pd.DataFrame, seasons: list[str], mmdd: str, window_days: int, min_players: int
) -> pd.DataFrame:
    """Pre-season squad value per club and season.

    For each player, take the latest valuation on or before <start year>-<mmdd> within
    window_days, and credit it to the club recorded with that valuation. Clubs with fewer
    than min_players valued players (mostly promoted clubs the data barely tracks) get NaN.
    """
    rows = []
    for season in seasons:
        cut = pd.Timestamp(f"{season[:4]}-{mmdd}")
        v = vals[(vals.date <= cut) & (vals.date > cut - pd.Timedelta(days=window_days))]
        latest = v.sort_values("date").drop_duplicates("player_id", keep="last")
        agg = latest.groupby("tm_id").agg(
            squad_value_eur=("value_eur", "sum"), squad_value_players=("player_id", "size")
        )
        rows.append(agg.reset_index().assign(season=season))
    out = pd.concat(rows, ignore_index=True)
    ok = (out.squad_value_eur > 0) & (out.squad_value_players >= min_players)
    out["log_squad_value"] = np.log(out.squad_value_eur.where(ok))
    return out
