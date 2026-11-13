"""Feature transforms on small synthetic tables (no real data)."""

import numpy as np
import pandas as pd
import pytest

from pochettino_fit import features as F


def shots_df(rows: list[tuple]) -> pd.DataFrame:
    cols = ["shot_id", "understat_id", "minute", "side", "xg", "situation", "result"]
    return pd.DataFrame(rows, columns=cols)


def test_red_card_truncation_and_weights() -> None:
    shots = shots_df([
        (1, 10, 5, "home", 0.3, "OpenPlay", "Goal"),
        (2, 10, 40, "away", 0.2, "OpenPlay", "SavedShot"),
        (3, 10, 60, "home", 0.5, "OpenPlay", "MissedShots"),   # after red at 45 -> dropped
        (4, 10, 70, "home", 0.76, "Penalty", "Goal"),          # penalty: never in npxG
        (5, 11, 10, "away", 0.1, "OpenPlay", "MissedShots"),
    ])
    red = pd.Series({10: 45.0})
    out = F.outcome(F.match_npxg(shots, red)).set_index("understat_id")
    assert out.loc[10, "home_npxg"] == pytest.approx(0.8)
    assert out.loc[10, "home_npxg_11v11"] == pytest.approx(0.3)
    assert out.loc[10, "minutes_11v11"] == 45
    assert out.loc[10, "w"] == pytest.approx(0.5)
    assert out.loc[10, "y"] == pytest.approx((0.3 - 0.2) / 0.5)
    assert out.loc[10, "home_pen_goals"] == 1
    assert out.loc[11, "minutes_11v11"] == 90 and out.loc[11, "w"] == 1


def test_stoppage_time_red_keeps_full_weight() -> None:
    shots = shots_df([(1, 1, 30, "home", 0.4, "OpenPlay", "Goal")])
    out = F.match_npxg(shots, pd.Series({1: 93.0}))
    assert out.minutes_11v11.iloc[0] == 90


def test_game_state_counts_own_goals_for_the_other_side() -> None:
    shots = shots_df([
        (1, 1, 10, "away", 0.0, "OpenPlay", "OwnGoal"),  # away player scores own goal: home +1
        (2, 1, 20, "home", 0.1, "OpenPlay", "MissedShots"),
        (3, 1, 30, "away", 0.2, "OpenPlay", "Goal"),
        (4, 1, 40, "home", 0.1, "OpenPlay", "MissedShots"),
    ])
    s = F.add_game_state(shots).set_index("shot_id")
    assert s.loc[1, "score_diff_before"] == 0
    assert s.loc[2, "score_diff_before"] == 1
    assert s.loc[4, "score_diff_before"] == 0


def luck_matches(n: int) -> pd.DataFrame:
    """Team A plays n home matches; luck differs by match so leakage is visible."""
    return pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)],
        "date": pd.date_range("2020-01-01", periods=n, freq="7D"),
        "home_team": ["A"] * n, "away_team": [f"B{i}" for i in range(n)],
        "home_goals": list(range(n)), "away_goals": [0] * n,
        "home_pen_goals": [0] * n, "away_pen_goals": [0] * n,
        "home_npxg": [1.0] * n, "away_npxg": [0.0] * n,
    })


def test_recent_luck_uses_only_earlier_matches() -> None:
    long = F.team_long(luck_matches(15))
    luck = F.recent_luck(long, window=10).query("team == 'A'").set_index("match_id")
    assert np.isnan(luck.loc["m0", "luck"]) and luck.loc["m0", "luck_n"] == 0
    # match k: previous luck values are (i - 1) for i < k
    assert luck.loc["m3", "luck"] == pytest.approx(np.mean([-1, 0, 1]))
    assert luck.loc["m12", "luck"] == pytest.approx(np.mean([i - 1 for i in range(2, 12)]))
    assert luck.loc["m12", "luck_n"] == 10


def test_recent_luck_unaffected_by_future_rows() -> None:
    full = F.recent_luck(F.team_long(luck_matches(15)), 10).set_index(["match_id", "team"])
    cut = F.recent_luck(F.team_long(luck_matches(8)), 10).set_index(["match_id", "team"])
    pd.testing.assert_frame_equal(cut, full.loc[cut.index])


def test_recent_luck_skips_missing_xg() -> None:
    df = luck_matches(4)
    df.loc[1, "home_npxg"] = np.nan
    luck = F.recent_luck(F.team_long(df), 10).query("team == 'A'").set_index("match_id")
    assert luck.loc["m3", "luck_n"] == 2
    assert luck.loc["m3", "luck"] == pytest.approx(np.mean([-1, 1]))


def test_manager_spells_and_tenure() -> None:
    games = pd.DataFrame({
        "tm_id": [1, 1, 1, 1, 2],
        "date": pd.to_datetime(["2012-08-01", "2013-01-01", "2014-06-01", "2017-01-01",
                                "2015-01-01"]),
        "manager": ["X", "X", "Yé", "Ye", "X"],
    })
    sp = F.manager_spells(games)
    assert len(sp) == 3  # accents ignored: "Yé" and "Ye" are one spell
    club1 = sp[sp.tm_id == 1].set_index("mkey")
    assert bool(club1.loc["x", "censored"]) and not bool(club1.loc["ye", "censored"])
    dates = pd.Series(pd.to_datetime(["2014-07-01", "2015-07-01", "2019-01-01"]))
    start = pd.Series(pd.to_datetime(["2014-06-01"] * 3))
    assert F.tenure_year(dates, start, cap=3).tolist() == [1, 2, 3]


def test_squad_values_use_pre_season_club_and_window() -> None:
    vals = pd.DataFrame({
        "player_id": [1, 1, 2, 3, 4],
        "date": pd.to_datetime(["2020-01-01", "2020-07-01", "2020-06-01", "2018-01-01",
                                "2020-09-01"]),
        "value_eur": [10.0, 20.0, 5.0, 99.0, 50.0],
        "tm_id": [7, 8, 8, 8, 8],
    })
    out = F.squad_values(vals, ["2020-21"], "08-01", 365, min_players=1).set_index("tm_id")
    # player 1 moved to club 8 by July; player 3 too old; player 4 valued after the cut-off
    assert out.loc[8, "squad_value_eur"] == 25.0
    assert 7 not in out.index
    few = F.squad_values(vals, ["2020-21"], "08-01", 365, min_players=3)
    assert few.log_squad_value.isna().all()


def test_name_key() -> None:
    assert F.name_key("Ole Gunnar Solskjær") == F.name_key("ole gunnar solskjaer")
    assert F.name_key(None) == ""
