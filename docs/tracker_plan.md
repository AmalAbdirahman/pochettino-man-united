# Tracker plan: what is public after each match

The tracker compares every United league match after 20 November 2026 with the frozen forecast
(`docs/forecasts/2026-11-20.json`), which is never edited. This file says exactly what is published and what
stays on my machine. The rule behind it: FotMob data is not ours to republish, so **no FotMob figure is ever
published**, in any mode.

## After each match

1. Add the match to the local FotMob notes in `data/raw/` (git-ignored, never committed).
2. Run `python -m pochettino_fit.tracker`. It writes two files:
   - `data/processed/tracker_local.json`: everything, including the FotMob values. Local only.
   - `docs/data/tracker.json`: the public file, filtered by `tracker.publication` in `config.yaml`.
3. The tracker then runs the export step, so `docs/data/manifest.json` and the page footer match the commit.
4. Commit and push `docs/data/`. Nothing else changes; the frozen forecast is checked by a SHA-256 test.

## What is public

| Item | Source | `derived_only` (default) | `results_only` |
|---|---|---|---|
| Result: opponent, venue, score, points | public result | yes | yes |
| Points since the freeze against the frozen 50% and 80% bands, inside or outside | result + frozen file | yes | yes |
| Frozen expectations per match: expected points, win/draw/loss chances, npxG difference expectation and interval | frozen file | yes | yes |
| Whether the match's npxG difference fell inside its frozen interval (yes or no) | our model, computed from FotMob xG | yes | no |
| Share of matches inside their interval | our model, computed from FotMob xG | yes | no |
| His updated effect: mean, 80% interval, chance it is positive | our model, computed from FotMob xG | yes | no |

## What stays local, always

| Item | Why |
|---|---|
| Actual npxG for and against, and their difference | FotMob figure |
| Per-match effect observation (actual minus the frozen expectation) | a FotMob figure shifted by a known constant |
| Chance quality conceded: npxG per shot, big chances, npxG against | FotMob figures |

The page never shows these. A test (`tests/test_tracker.py`) fails if any of them reaches the public file
in either mode.

## Choosing a mode

- **`derived_only`** (default) publishes our own model outputs. They are computed from FotMob xG but are
  not FotMob figures: a yes or no per match, a share, and a posterior for his effect. One caveat: the
  updated effect mean is a weighted average of the prior and the per-match observations, so with the prior
  and noise level (both in the frozen file) a reader could work back to the **average** npxG difference
  over the matches so far, though not to any single match's figure.
- **`results_only`** publishes results and the frozen bands only. Use it if that aggregate is judged too
  close to republishing FotMob data. The page then shows points against the frozen bands and expected
  points, and marks the npxG check as "not published".

Switch by setting `tracker.publication` in `config.yaml` and rerunning the tracker. There is no setting that
publishes FotMob values; the earlier `publish_fotmob_values` switch was removed on 2026-11-20.
