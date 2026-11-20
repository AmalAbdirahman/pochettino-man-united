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
