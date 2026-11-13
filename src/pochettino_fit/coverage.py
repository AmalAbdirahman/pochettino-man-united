"""Coverage report for the processed match tables (local only, not published)."""

from pathlib import Path
from typing import Any

import pandas as pd

from pochettino_fit import sources as S


def _pct(s: pd.Series) -> str:
    return f"{100 * s.mean():.0f}%"


def league_season_table(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby(["league", "season"])
    return pd.DataFrame({
        "matches": g.size(),
        "with_y": g.y.apply(lambda s: _pct(s.notna())),
        "red_known": g.red_known.apply(lambda s: _pct(s.astype(bool))),
        "managers": g.apply(lambda d: _pct(d.home_manager.notna() & d.away_manager.notna()),
                            include_groups=False),
        "squad_value": g.home_log_squad_value.apply(lambda s: _pct(s.notna())),
        "league_elo": g.league_elo.apply(lambda s: _pct(s.notna())),
        "last_xg_date": g.apply(lambda d: d.date[d.y.notna()].max(), include_groups=False),
    }).reset_index()


def manager_rows(m: pd.DataFrame, surname: str) -> pd.DataFrame:
    rows = []
    for side in ("home", "away"):
        hit = m[m[f"{side}_manager"].fillna("").str.contains(surname)]
        rows.append(pd.DataFrame({"club": hit[f"{side}_name"], "season": hit.season,
                                  "has_y": hit.y.notna(), "date": hit.date}))
    r = pd.concat(rows)
    return r.groupby(["club", "season"]).agg(
        matches=("has_y", "size"), with_y=("has_y", "sum"),
        first=("date", "min"), last=("date", "max")).reset_index()


def united_rows(m: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    u = m[(m.home_name == "Manchester United") | (m.away_name == "Manchester United")]
    u = u[u.season.isin(["2025-26", cfg["appointment"]["season"]])]
    return u.groupby(["season", u.xg_source.fillna("none")]).size().rename("matches").reset_index()


def fd_vs_fotmob(m: pd.DataFrame, raw: Path) -> str:
    """United 2026-27: football-data xG vs FotMob xG (both include penalties)."""
    fm = S.load_fotmob_united(raw).sort_values("gameweek").reset_index(drop=True)
    u = m[(m.xg_source == "fotmob")].sort_values("date").reset_index(drop=True)
    if len(u) != len(fm):
        return "not available"
    home = fm.venue == "home"
    fd_for = u.home_xg_fd.where(home, u.away_xg_fd)
    fd_against = u.away_xg_fd.where(home, u.home_xg_fd)
    fd = pd.concat([fd_for, fd_against], ignore_index=True)
    fot = pd.concat([fm.xg_for, fm.xg_against], ignore_index=True)
    return (f"n = {len(fd)} team-matches; mean football-data xG {fd.mean():.2f} vs FotMob "
            f"{fot.mean():.2f}; correlation {fd.corr(fot):.2f}; mean abs diff "
            f"{(fd - fot).abs().mean():.2f}")


def write_report(m: pd.DataFrame, tmm: pd.DataFrame, cfg: dict[str, Any], path: Path) -> str:
    raw = Path(path).parents[1] / "raw"
    parts = [
        "# Coverage report (local only; contains no raw source data)\n",
        f"matches.parquet: {len(m)} rows, {m.y.notna().sum()} with y. "
        f"team_matches.parquet: {len(tmm)} rows.\n",
        "## By league and season\n", league_season_table(m).to_markdown(index=False),
        "\n## Pochettino\n", manager_rows(m, "Pochettino").to_markdown(index=False),
        "\n## Carrick\n", manager_rows(m, "Carrick").to_markdown(index=False),
        "\n## United, xG source by season\n", united_rows(m, cfg).to_markdown(index=False),
        "\n## football-data xG vs FotMob xG (United 2026-27)\n", fd_vs_fotmob(m, raw),
        "\n## Known gaps\n",
        "- No npxG after 2026-04-18 (Understat archive frozen; site offline since ~2026-11-02).\n"
        "- 2026-27: npxG only for United (FotMob, different provider); other teams have "
        "football-data xG (includes penalties, provider unstated) in home_xg_fd/away_xg_fd.\n"
        "- 2026-27 red-card minutes unknown: matches with a red card have no y.\n"
        "- league_elo is empty while the ClubElo API is down.\n"
        "- Squad value is missing where fewer than squad_value_min_players players are valued "
        "(mostly promoted clubs that Transfermarkt-datasets barely tracks).\n"
        "- football-data xG matches FotMob xG almost exactly (section above), so it is very "
        "likely FotMob's numbers: treat it under the same terms (derived figures only).\n"
        "- Awarded or abandoned games (score_awarded) keep the official score but have no xG.\n",
    ]
    text = "\n".join(parts)
    Path(path).write_text(text, encoding="utf-8")
    return text
