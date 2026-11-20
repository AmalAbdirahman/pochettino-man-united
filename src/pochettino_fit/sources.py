"""Load raw source files into tidy DataFrames. Nothing here writes to disk."""

from pathlib import Path

import pandas as pd

BIG5 = ["ENG", "ESP", "GER", "ITA", "FRA"]


def season_label(start_year: int) -> str:
    """2014 -> '2014-15'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_of_date(d: pd.Timestamp) -> str:
    """European season containing date d (season turns on 1 July)."""
    return season_label(d.year if d.month >= 7 else d.year - 1)


# ---------- Understat archive ----------

def load_understat_shots(raw: Path) -> pd.DataFrame:
    s = pd.read_parquet(raw / "understat_archive" / "understat_shots.parquet")
    s = s[s.league_code.isin(BIG5)].drop_duplicates("id")
    out = pd.DataFrame({
        "shot_id": s.id.astype(int),
        "understat_id": s.match_id.astype(int),
        "minute": pd.to_numeric(s.minute).astype(int),
        "side": s.h_a.map({"h": "home", "a": "away"}),
        "xg": pd.to_numeric(s.x_g),
        "situation": s.situation,
        "result": s.result,
    })
    return out.reset_index(drop=True)


def load_understat_player_shots(raw: Path) -> pd.DataFrame:
    """Non-penalty, non-own-goal shots with shooter id, shooter's team and season."""
    s = pd.read_parquet(raw / "understat_archive" / "understat_shots.parquet")
    s = s[s.league_code.isin(BIG5)].drop_duplicates("id")
    s = s[(s.situation != "Penalty") & (s.result != "OwnGoal")]
    return pd.DataFrame({
        "player_id": s.player_id.astype(int),
        "player": s.player,
        "team": s.h_team.where(s.h_a == "h", s.a_team),
        "understat_id": s.match_id.astype(int),
        "season": s.season.astype(int).map(season_label),
        "npxg": pd.to_numeric(s.x_g),
        "goal": (s.result == "Goal").astype(int),
    }).reset_index(drop=True)


def load_understat_matches(raw: Path) -> pd.DataFrame:
    m = pd.read_parquet(raw / "understat_archive" / "understat_metadata.parquet")
    m = m[m.league_code.isin(BIG5)].drop_duplicates("understat_id")
    date = pd.to_datetime(m.match_date)
    return pd.DataFrame({
        "understat_id": m.understat_id.astype(int),
        "league": m.league_code,
        "season": m.season.astype(int).map(season_label),
        "date": date.dt.normalize(),
        "home": m.home_team,
        "away": m.away_team,
        "home_goals": m.home_score.astype(int),
        "away_goals": m.away_score.astype(int),
        "home_deep": m.home_deep,
        "away_deep": m.away_deep,
        "home_ppda": m.home_ppda,
        "away_ppda": m.away_ppda,
    }).reset_index(drop=True)


# ---------- Transfermarkt (transfermarkt-datasets) ----------

def _tm(raw: Path, name: str, **kw: object) -> pd.DataFrame:
    return pd.read_csv(raw / "transfermarkt" / f"{name}.csv.gz", encoding="utf-8", **kw)


def load_tm_league_games(raw: Path, tm_codes: dict[str, str]) -> pd.DataFrame:
    """Big-five league games, one row per game (home view)."""
    g = _tm(raw, "games", parse_dates=["date"])
    inv = {v: k for k, v in tm_codes.items()}
    g = g[g.competition_id.isin(inv)]
    return pd.DataFrame({
        "tm_game_id": g.game_id,
        "league": g.competition_id.map(inv),
        "season": g.season.astype(int).map(season_label),
        "date": g.date,
        "home_tm_id": g.home_club_id,
        "away_tm_id": g.away_club_id,
        "home_tm_name": g.home_club_name,
        "away_tm_name": g.away_club_name,
        "home_goals": g.home_club_goals.astype(int),
        "away_goals": g.away_club_goals.astype(int),
        "home_manager": g.home_club_manager_name,
        "away_manager": g.away_club_manager_name,
    }).reset_index(drop=True)


def load_tm_club_manager_games(raw: Path) -> pd.DataFrame:
    """Every TM game (all competitions) per club with its manager: for spell detection."""
    cg = _tm(raw, "club_games", usecols=["game_id", "club_id", "own_manager_name"])
    g = _tm(raw, "games", usecols=["game_id", "date"], parse_dates=["date"])
    out = cg.merge(g, on="game_id").rename(columns={"club_id": "tm_id", "own_manager_name": "manager"})
    return out.dropna(subset=["manager"])[["tm_id", "date", "manager"]]


def load_tm_first_red(raw: Path) -> tuple[pd.DataFrame, set[int]]:
    """First red-card minute (straight or second yellow) per game, and the games with events."""
    ev = _tm(raw, "game_events", usecols=["game_id", "minute", "type", "description"])
    red = ev[(ev.type == "Cards") & ev.description.str.contains("Red card|Second yellow", na=False)]
    first = red.groupby("game_id", as_index=False).minute.min()
    first = first.rename(columns={"game_id": "tm_game_id", "minute": "first_red_minute"})
    return first, set(ev.game_id.unique())


def load_tm_valuations(raw: Path) -> pd.DataFrame:
    v = _tm(raw, "player_valuations", parse_dates=["date"])
    return v.rename(columns={"current_club_id": "tm_id", "market_value_in_eur": "value_eur"})[
        ["player_id", "date", "value_eur", "tm_id", "current_club_name"]
    ]


# ---------- football-data.co.uk ----------

def load_football_data(raw: Path, league: str, season_code: str) -> pd.DataFrame:
    path = raw / "football_data" / f"{league}_{season_code}.csv"
    d = pd.read_csv(path, encoding="latin-1").dropna(subset=["HomeTeam"])
    d.columns = [c.lstrip("﻿").lstrip("ï»¿") for c in d.columns]
    out = pd.DataFrame({
        "league": league,
        "date": pd.to_datetime(d.Date, dayfirst=True, format="mixed"),
        "home": d.HomeTeam,
        "away": d.AwayTeam,
        "home_goals": d.FTHG.astype(int),
        "away_goals": d.FTAG.astype(int),
        "home_reds": d.get("HR"),
        "away_reds": d.get("AR"),
        "home_xg_fd": d.get("HxG"),
        "away_xg_fd": d.get("AxG"),
        "home_shots": d.get("HS"),
        "away_shots": d.get("AS"),
        "home_sot": d.get("HST"),
        "away_sot": d.get("AST"),
        "home_corners": d.get("HC"),
        "away_corners": d.get("AC"),
    })
    out["season"] = out.date.map(season_of_date)
    return out.reset_index(drop=True)


# ---------- ClubElo snapshots (league averages only) ----------

def load_league_elo(raw: Path, clubelo_codes: dict[str, str]) -> pd.DataFrame:
    """Mean Elo of top-tier clubs per league at each season-start snapshot."""
    rows = []
    for f in sorted((raw / "clubelo").glob("*.csv")):
        snap = pd.read_csv(f)
        season = season_of_date(pd.Timestamp(f.stem))
        for lg, code in clubelo_codes.items():
            top = snap[(snap.Country == code) & (snap.Level == 1)]
            rows.append({"league": lg, "season": season, "league_elo": top.Elo.mean()})
    return pd.DataFrame(rows, columns=["league", "season", "league_elo"])


# ---------- local hand files ----------

def load_fotmob_united(raw: Path) -> pd.DataFrame:
    """United 2026-27 league matches from the local FotMob notes (never published)."""
    return pd.read_csv(raw / "united_2026_27_fotmob.csv")
