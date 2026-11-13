"""Team name matching on synthetic fixtures."""

import pandas as pd

from pochettino_fit import names as N


def tm_games() -> pd.DataFrame:
    rows = []
    pairs = [(1, "Alpha FC", 2, "Beta United"), (2, "Beta United", 3, "Gamma CF"),
             (3, "Gamma CF", 1, "Alpha FC")]
    for k in range(6):
        h, hn, a, an = pairs[k % 3]
        rows.append({"league": "ENG", "date": pd.Timestamp("2020-08-01") + pd.Timedelta(days=7 * k),
                     "home_tm_id": h, "home_tm_name": hn, "away_tm_id": a, "away_tm_name": an,
                     "home_goals": k % 3, "away_goals": (k + 1) % 2})
    return pd.DataFrame(rows)


def alias_games(tm: pd.DataFrame, rename: dict[str, str], shift_days: int = 0) -> pd.DataFrame:
    return pd.DataFrame({
        "league": tm.league, "date": tm.date + pd.Timedelta(days=shift_days),
        "home": tm.home_tm_name.map(rename), "away": tm.away_tm_name.map(rename),
        "home_goals": tm.home_goals, "away_goals": tm.away_goals,
    })


RENAME = {"Alpha FC": "Alpha", "Beta United": "Beta", "Gamma CF": "Gamma"}


def test_aliases_match_on_date_and_score() -> None:
    tm = tm_games()
    ok, failed = N.match_aliases(alias_games(tm, RENAME), tm, min_votes=3, min_share=0.8)
    assert failed == []
    assert dict(zip(ok.alias, ok.tm_id, strict=True)) == {"Alpha": 1, "Beta": 2, "Gamma": 3}


def test_aliases_tolerate_one_day_shift() -> None:
    tm = tm_games()
    ok, failed = N.match_aliases(alias_games(tm, RENAME, shift_days=1), tm, 3, 0.8)
    assert failed == [] and len(ok) == 3


def test_unknown_club_is_reported() -> None:
    tm = tm_games()
    src = alias_games(tm, RENAME)
    extra = src.iloc[[0]].assign(home="Delta", date=pd.Timestamp("2020-07-01"))
    ok, failed = N.match_aliases(pd.concat([src, extra]), tm, 3, 0.8)
    assert failed == ["Delta"]


def test_check_mapping_flags_duplicates_and_missing() -> None:
    names = pd.DataFrame({"source": ["a", "a", "b"], "alias": ["X", "X", "Y"],
                          "tm_id": [1, 2, None], "tm_name": ["x", "y", None]})
    problems = N.check_mapping(names)
    assert any("twice" in p for p in problems) and any("missing" in p for p in problems)


def test_committed_mapping_is_clean() -> None:
    from pochettino_fit.config import ROOT
    names = pd.read_csv(ROOT / "data" / "team_names.csv")
    assert N.check_mapping(names) == []
    utd = names[(names.source == "football_data") & (names.alias == "Man United")]
    assert utd.tm_name.tolist() == ["Manchester United"]
