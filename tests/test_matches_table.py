"""One row per match: synthetic check always, real table when it has been built locally."""

import pandas as pd
import pytest

from pochettino_fit.config import ROOT
from pochettino_fit.features import one_row_problems

PROCESSED = ROOT / "data" / "processed" / "matches.parquet"


def base() -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": ["a", "b"], "season": ["2020-21"] * 2,
        "date": pd.to_datetime(["2020-09-01", "2020-09-01"]),
        "home_team": [1, 3], "away_team": [2, 4],
    })


def test_clean_table_passes() -> None:
    assert one_row_problems(base()) == []


def test_mirrored_row_is_caught() -> None:
    m = pd.concat([base(), pd.DataFrame({"match_id": ["c"], "season": ["2020-21"],
                                         "date": pd.to_datetime(["2020-09-01"]),
                                         "home_team": [2], "away_team": [1]})])
    assert "a team appears twice on one date" in one_row_problems(m)
    assert "mirrored rows" in one_row_problems(m)


def test_duplicate_id_is_caught() -> None:
    m = base().assign(match_id=["a", "a"])
    assert "duplicate match_id" in one_row_problems(m)


@pytest.mark.skipif(not PROCESSED.exists(), reason="matches.parquet not built (data is local only)")
def test_real_matches_table() -> None:
    m = pd.read_parquet(PROCESSED)
    assert one_row_problems(m) == []
    assert (m.w.dropna() > 0).all() and (m.w.dropna() <= 1).all()
    assert m.home_luck_n.max() <= 10
    leagues = set(m.league)
    assert leagues == {"ENG", "ESP", "GER", "ITA", "FRA"}
