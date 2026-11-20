"""football-data shots and corners: name mapping, manager join and long form."""

import pandas as pd
import pytest

from pochettino_fit import fd_stats as F


def fd() -> pd.DataFrame:
    return pd.DataFrame({"league": "ENG", "season": "2019-20", "date": pd.to_datetime(["2020-07-01"]),
                         "home": ["Everton"], "away": ["Leicester"], "home_shots": [7], "away_shots": [12],
                         "home_sot": [3], "away_sot": [4], "home_corners": [2], "away_corners": [6]})


def names() -> pd.DataFrame:
    return pd.DataFrame({"source": "football_data", "alias": ["Everton", "Leicester"], "tm_id": [29, 1003],
                         "tm_name": ["Everton FC", "Leicester City"]})


def test_attach_ids_and_unmapped_names() -> None:
    d = F.attach_ids(fd(), names())
    assert (d.home_team.iloc[0], d.away_team.iloc[0]) == (29, 1003)
    with pytest.raises(ValueError):
        F.attach_ids(fd().assign(away="Nowhere"), names())


def test_join_keeps_the_file_season_and_long_form_mirrors() -> None:
    d = F.attach_ids(fd(), names())
    m = pd.DataFrame({"league": "ENG", "season": "2019-20", "home_team": [29], "away_team": [1003],
                      "date": pd.to_datetime(["2020-07-01"]), "home_manager": ["A"], "away_manager": ["B"],
                      "match_id": ["x"]})
    j, dropped = F.join_managers(d, m)
    assert dropped == 0
    t = F.long_form(j)
    home, away = t[t.is_home].iloc[0], t[~t.is_home].iloc[0]
    assert (home.manager, home.shots_for, home.shots_against, home.corners_against) == ("A", 7, 12, 6)
    assert (away.manager, away.shots_for, away.sot_against) == ("B", 12, 3)
    _, dropped = F.join_managers(d, m.assign(season="2020-21"))
    assert dropped == 1
