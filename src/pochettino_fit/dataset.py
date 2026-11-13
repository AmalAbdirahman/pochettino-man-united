"""Build data/processed/matches.parquet and team_matches.parquet (prompt 1).

Run: python -m pochettino_fit.dataset
Backbone: Transfermarkt league games to 2025-26, football-data.co.uk for 2026-27.
npxG: Understat archive (to 2026-04-18); United 2026-27 from local FotMob notes.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pochettino_fit import features as F
from pochettino_fit import names as N
from pochettino_fit import sources as S
from pochettino_fit.config import ROOT, load_config
from pochettino_fit.fetch import football_data_season_code, seasons_from

PEN_XG = 0.79  # FotMob penalty xG per attempt, used only to back out penalties from xG - npxG


def _team_maps(names: pd.DataFrame, source: str) -> tuple[dict[str, int], dict[int, str]]:
    sub = names[names.source == source]
    return dict(zip(sub.alias, sub.tm_id, strict=True)), dict(zip(names.tm_id, names.tm_name, strict=True))


def build_names(cfg: dict[str, Any], tm: pd.DataFrame, us: pd.DataFrame, fd: pd.DataFrame) -> pd.DataFrame:
    fc = cfg["features"]
    manual = N.load_manual(ROOT / "data" / "team_names_manual.csv")
    parts = [manual]
    for source, df in (("understat", us), ("football_data", fd)):
        ok, failed = N.match_aliases(df, tm, fc["name_match_min_votes"], fc["name_match_min_share"])
        missing = sorted(set(failed) - set(manual[manual.source == source].alias))
        if missing:
            raise ValueError(f"unmatched {source} team names: {missing}")
        parts.append(ok.assign(source=source)[["source", "alias", "tm_id", "tm_name"]])
    names = pd.concat(parts, ignore_index=True)
    problems = N.check_mapping(names)
    if problems:
        raise ValueError("; ".join(problems))
    return names.sort_values(["source", "alias"]).reset_index(drop=True)


def tm_base(tm: pd.DataFrame, raw: Path, seasons: list[str]) -> pd.DataFrame:
    """TM league games in scope, with the first red-card minute."""
    first_red, with_events = S.load_tm_first_red(raw)
    b = tm[tm.season.isin(seasons)].merge(first_red, on="tm_game_id", how="left")
    b["red_known"] = b.tm_game_id.isin(with_events)
    b["match_id"] = "tm" + b.tm_game_id.astype(str)
    b = b.rename(columns={"home_tm_id": "home_team", "away_tm_id": "away_team",
                          "home_tm_name": "home_name", "away_tm_name": "away_name"})
    b["home_tenure_start"] = pd.NaT
    b["away_tenure_start"] = pd.NaT
    return b


def attach_understat(base: pd.DataFrame, raw: Path, names: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add Understat npxG (truncated at first red card) to TM games. Returns (matches, shots)."""
    us = S.load_understat_matches(raw)
    to_tm, _ = _team_maps(names, "understat")
    us["home_team"], us["away_team"] = us.home.map(to_tm), us.away.map(to_tm)
    keys = ["league", "season", "home_team", "away_team"]
    m = base.merge(us[keys + ["understat_id", "home_goals", "away_goals"]], on=keys, how="left",
                   suffixes=("", "_us"))
    # Awarded or abandoned games: TM has the official score, Understat the on-pitch one.
    bad = m.understat_id.notna() & (
        (m.home_goals != m.home_goals_us) | (m.away_goals != m.away_goals_us))
    if bad.sum() > 10:
        raise ValueError(f"score mismatch TM vs Understat in {int(bad.sum())} games")
    m["score_awarded"] = bad
    m.loc[bad, "understat_id"] = np.nan
    shots = F.add_game_state(S.load_understat_shots(raw))
    red = m.dropna(subset=["understat_id"]).set_index("understat_id").first_red_minute.dropna()
    red.index = red.index.astype(int)
    npx = F.match_npxg(shots, red)
    m = m.drop(columns=["home_goals_us", "away_goals_us"]).merge(npx, on="understat_id", how="left")
    m["xg_source"] = np.where(m.understat_id.notna(), "understat", None)
    return m, shots.merge(us[["understat_id"] + keys[2:]], on="understat_id")


def managers_2026_27(path: Path, names: pd.DataFrame) -> pd.DataFrame:
    mg = pd.read_csv(path, comment="#", parse_dates=["start", "end"])
    to_tm, _ = _team_maps(names, "football_data")
    mg["tm_id"] = mg.club.map(to_tm)
    if mg.tm_id.isna().any():
        raise ValueError(f"unmapped clubs in managers file: {sorted(mg.club[mg.tm_id.isna()])}")
    return mg


def _manager_on(mg: pd.DataFrame, tm_id: int, date: pd.Timestamp) -> pd.Series:
    rows = mg[(mg.tm_id == tm_id)
              & (mg.start.isna() | (mg.start <= date)) & (mg.end.isna() | (mg.end >= date))]
    if len(rows) != 1:
        raise ValueError(f"{len(rows)} manager rows for club {tm_id} on {date.date()}")
    return rows.iloc[0]


def fd_current(raw: Path, season: str, names: pd.DataFrame, mg: pd.DataFrame) -> pd.DataFrame:
    """2026-27 league matches from football-data.co.uk with managers from the hand file."""
    code = football_data_season_code(season)
    fd = pd.concat([S.load_football_data(raw, lg, code) for lg in S.BIG5], ignore_index=True)
    to_tm, tm_name = _team_maps(names, "football_data")
    fd["home_team"], fd["away_team"] = fd.home.map(to_tm), fd.away.map(to_tm)
    fd["home_name"], fd["away_name"] = fd.home_team.map(tm_name), fd.away_team.map(tm_name)
    for side in ("home", "away"):
        info = [_manager_on(mg, t, d) for t, d in zip(fd[f"{side}_team"], fd.date, strict=True)]
        fd[f"{side}_manager"] = [r.manager if isinstance(r.manager, str) else None for r in info]
        fd[f"{side}_tenure_start"] = [r.start for r in info]
    no_red = (fd.home_reds.fillna(0) + fd.away_reds.fillna(0)) == 0
    fd["red_known"] = no_red
    fd["first_red_minute"] = np.nan
    fd["minutes_11v11"] = np.where(no_red, 90.0, np.nan)
    fd["match_id"] = ("fd" + fd.league + fd.date.dt.strftime("%Y%m%d")
                      + fd.home_team.astype(str) + "v" + fd.away_team.astype(str))
    return fd.drop(columns=["home", "away", "home_reds", "away_reds"])


def attach_fotmob(cur: pd.DataFrame, raw: Path, united_id: int) -> pd.DataFrame:
    """United 2026-27 npxG from local FotMob notes, matched in date order and checked on score."""
    fm = S.load_fotmob_united(raw).sort_values("gameweek").reset_index(drop=True)
    is_utd = (cur.home_team == united_id) | (cur.away_team == united_id)
    utd = cur[is_utd].sort_values("date")
    if len(utd) != len(fm):
        raise ValueError(f"United matches: football-data {len(utd)} vs FotMob {len(fm)}")
    cur = cur.copy()
    for idx, (_, r) in zip(utd.index, fm.iterrows(), strict=True):
        home = r.venue == "home"
        me, op = ("home", "away") if home else ("away", "home")
        if (cur.at[idx, f"{me}_goals"], cur.at[idx, f"{op}_goals"]) != (r.goals_for, r.goals_against):
            raise ValueError(f"FotMob score mismatch at gameweek {r.gameweek}")
        for side, k in ((me, "for"), (op, "against")):
            cur.at[idx, f"{side}_npxg"] = r[f"npxg_{k}"]
            cur.at[idx, f"{side}_pen_goals"] = round((r[f"xg_{k}"] - r[f"npxg_{k}"]) / PEN_XG)
        if cur.at[idx, "red_known"]:
            for side in ("home", "away"):
                cur.at[idx, f"{side}_npxg_11v11"] = cur.at[idx, f"{side}_npxg"]
        cur.at[idx, "xg_source"] = "fotmob"
    return cur


def add_tenure(m: pd.DataFrame, raw: Path, cap: int) -> pd.DataFrame:
    """Tenure year from TM spells, or from the hand-file start date in 2026-27."""
    spells = F.manager_spells(S.load_tm_club_manager_games(raw))
    for side in ("home", "away"):
        t = m[["match_id", "date", f"{side}_team", f"{side}_manager"]].copy()
        t.columns = ["match_id", "date", "tm_id", "manager"]
        t["mkey"] = t.manager.map(F.name_key)
        t = t.sort_values("date")
        sp = spells[["tm_id", "mkey", "start", "censored"]].sort_values("start")
        j = pd.merge_asof(t, sp, left_on="date", right_on="start", by=["tm_id", "mkey"])
        j = j.set_index("match_id").reindex(m.match_id)
        given = m[f"{side}_tenure_start"].to_numpy()
        start = pd.Series(np.where(pd.notna(given), given, j.start.to_numpy()), index=m.index)
        start = pd.to_datetime(start)
        m[f"{side}_tenure"] = F.tenure_year(m.date, start, cap)
        m[f"{side}_tenure_censored"] = np.where(pd.notna(given), False, j.censored.to_numpy())
        m.loc[m[f"{side}_manager"].isna(), f"{side}_tenure"] = np.nan
    return m.drop(columns=["home_tenure_start", "away_tenure_start"])


def add_luck(m: pd.DataFrame, window: int) -> pd.DataFrame:
    long = F.team_long(m)
    luck = F.recent_luck(long, window)
    for side in ("home", "away"):
        lk = luck.rename(columns={"team": f"{side}_team", "luck": f"{side}_luck",
                                  "luck_n": f"{side}_luck_n"})
        m = m.merge(lk, on=["match_id", f"{side}_team"], how="left")
    return m


def add_strength_inputs(m: pd.DataFrame, raw: Path, cfg: dict[str, Any], seasons: list[str]) -> pd.DataFrame:
    fc = cfg["features"]
    vals = F.squad_values(S.load_tm_valuations(raw), seasons, fc["squad_value_date"],
                          fc["squad_value_window_days"], fc["squad_value_min_players"])
    for side in ("home", "away"):
        v = vals.rename(columns={"tm_id": f"{side}_team", "log_squad_value": f"{side}_log_squad_value",
                                 "squad_value_players": f"{side}_squad_value_players"})
        m = m.merge(v.drop(columns="squad_value_eur"), on=["season", f"{side}_team"], how="left")
    elo = S.load_league_elo(raw, {k: v["clubelo"] for k, v in cfg["leagues"].items()})
    return m.merge(elo, on=["league", "season"], how="left")


def build(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = ROOT / cfg["paths"]["raw"]
    current = cfg["appointment"]["season"]
    seasons = seasons_from(cfg["scope"]["first_season"], current)
    tm = S.load_tm_league_games(raw, {k: v["transfermarkt"] for k, v in cfg["leagues"].items()})
    fd_all = pd.concat([S.load_football_data(raw, lg, football_data_season_code(s))
                        for s in seasons for lg in S.BIG5], ignore_index=True)
    names = build_names(cfg, tm, S.load_understat_matches(raw), fd_all)

    hist, shots = attach_understat(tm_base(tm, raw, seasons[:-1]), raw, names)
    mg = managers_2026_27(ROOT / "data" / "managers_2026_27.csv", names)
    united = int(names[(names.source == "football_data") & (names.alias == "Man United")].tm_id.iloc[0])
    cur = attach_fotmob(fd_current(raw, current, names, mg), raw, united)
    m = pd.concat([hist, cur], ignore_index=True).sort_values(["date", "match_id"])
    m = m[m.date <= pd.Timestamp(cfg["features"]["data_cutoff"])]

    m = add_luck(m, cfg["luck"]["window_matches"])
    m = add_tenure(m, raw, cfg["features"]["tenure_cap_years"])
    m = add_strength_inputs(m, raw, cfg, seasons)
    m = F.outcome(m)
    short = m.minutes_11v11 < cfg["features"]["min_minutes_11v11"]
    m = m[~short].reset_index(drop=True)
    problems = F.one_row_problems(m)
    if problems:
        raise ValueError("; ".join(problems))
    return m, team_matches(m, shots, raw), names


def team_matches(m: pd.DataFrame, shots: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """One row per team per match (Understat matches only) for style metrics."""
    s = shots[F.is_np_shot(shots)].copy()
    s["close"] = s.score_diff_before.abs() <= 1
    s["setpiece_xg"] = s.xg.where(s.situation.isin(F.SET_PIECES), 0.0)
    g = s.groupby(["understat_id", "side"]).agg(
        shots=("xg", "size"), npxg=("xg", "sum"), setpiece_npxg=("setpiece_xg", "sum"),
        shots_close=("close", "sum"),
        npxg_close=("xg", lambda x: x[s.loc[x.index, "close"]].sum()),
    ).unstack("side", fill_value=0)
    us = S.load_understat_matches(raw).set_index("understat_id")
    base = m.dropna(subset=["understat_id"]).copy()
    base["understat_id"] = base.understat_id.astype(int)
    rows = []
    for me, op in (("home", "away"), ("away", "home")):
        t = base[["match_id", "understat_id", "league", "season", "date"]].copy()
        t["team"], t["opponent"] = base[f"{me}_team"], base[f"{op}_team"]
        t["manager"], t["opponent_manager"] = base[f"{me}_manager"], base[f"{op}_manager"]
        t["is_home"] = me == "home"
        for k, src in (("for", me), ("against", op)):
            for col in ("shots", "npxg", "setpiece_npxg", "shots_close", "npxg_close"):
                t[f"{col}_{k}"] = t.understat_id.map(g[(col, src)]).fillna(0)
            t[f"deep_{k}"] = t.understat_id.map(us[f"{src}_deep"])
            t[f"ppda_{k}"] = t.understat_id.map(us[f"{src}_ppda"])
        rows.append(t)
    tmm = pd.concat(rows, ignore_index=True)
    for k in ("for", "against"):
        tmm[f"npxg_per_shot_{k}"] = tmm[f"npxg_{k}"] / tmm[f"shots_{k}"].replace(0, np.nan)
    return tmm.sort_values(["date", "match_id", "is_home"]).reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    m, tmm, names = build(cfg)
    out = ROOT / cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "matches.parquet", index=False)
    tmm.to_parquet(out / "team_matches.parquet", index=False)
    names.to_csv(ROOT / "data" / "team_names.csv", index=False)
    from pochettino_fit.coverage import write_report
    print(write_report(m, tmm, cfg, out / "coverage.md"))


if __name__ == "__main__":
    main()
