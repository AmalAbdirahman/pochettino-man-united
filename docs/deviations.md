# Deviations and post-freeze notes

The frozen forecast (docs/forecasts/2026-11-20.json) is never edited. Pre-freeze deviations from the method
are listed inside that file. Notes added after the freeze go here, dated. None of them changes the frozen
numbers.

## 2026-11-20: xG providers are mixed without calibration

The strength window combines Understat xG (all league matches to 18 April 2026) with FotMob-based xG (2026-27). The two providers never cover the same season here, so no calibration between them was possible. A shared shift between providers (one rating every chance a little higher) largely cancels in an xG difference and in attack against defence; a difference in spread (one provider separating good and poor chances more sharply) does not cancel, and would make 2026-27 strengths look more or less extreme than earlier ones.

Why it matters for the forecast: United's 2026-27 attack and defence, and every opponent's, are estimated
partly from FotMob-based xG while the earlier part of the window is Understat. If the providers differ in
spread, the 2026-27 part of each team's strength is scaled differently from the rest. The recency weighting
(half-life 180 days) gives 2026-27 more weight, so the forecast leans towards FotMob's scale. The back-test
used Understat only, so it does not measure this.

## 2026-11-20: no league bridge

methods.md (version 2) described a league-season offset anchored by league-average ClubElo. In the
implementation that offset cancels in the within-league outcome and was never estimated, and ClubElo was
never downloaded successfully. Cross-league comparison of managers rests on the assumption that an effect in
xG difference means the same in every league. methods.md sections 1a, 1b, 8 and 9 now say so. This does not
change the frozen forecast, which never used a league rating.

## 2026-11-21: squad list fact-checked against Wikipedia

`data/united_squad_2026_27.csv` (33 players, from the Wikipedia 2026-27 season page, squad statistics table
as of 2026-11-08) was rechecked against the same page on 2026-11-21. Every shirt number, name and position
matches, and no player was added or removed. The page's summer 2026 transfers explain the players with no
United league matches in 2025-26: Andrey Santos (from Chelsea), Karl Darlow (Leeds United), Youri Tielemans
(Aston Villa), Tynan Thompson (Tottenham Hotspur) and Carlos Baleba (Brighton & Hove Albion). Marcus Rashford
is listed in the squad after his 2025-26 loan; the page does not list a further loan. None of the summer
departures or loans out (for example Højlund, Casemiro, Onana, Bayındır) appear in the CSV.

The CSV itself is not edited, including its `verified` column, because the frozen forecast reads it. No
2026-27 player-level match data is available locally (the FotMob notes are team-level), so appearances this
season were not checked.
