# PROMPTS.md: Pochettino at United (method version 2)

Start every prompt with:
```
Read CLAUDE.md and docs/methods.md first. Plan briefly, wait for "go", then implement. Run pytest, give me the command to see it working, list anything skipped, and commit.
```

**Deadline:** the forecast (prompt 6) must be committed and pushed BEFORE Pochettino's first Premier League match. Check the fixture date first. If time is short, do 1, 2, 3 and 6 with the fallback options marked below, then the rest after.

## 0. Setup
```
Create pyproject.toml (package pochettino_fit, Python 3.12; deps pandas numpy scipy statsmodels pymc arviz
matplotlib soccerdata pyarrow; dev: pytest ruff), config.yaml, the layout in CLAUDE.md, .gitignore with
data/raw/ and .env, MIT licence, GitHub Actions running ruff and pytest. Copy methods.md into docs/.
```

## 1. Data
```
Check and report each source's terms before downloading. Build data/processed/matches.parquet with ONE row
per league match (home view): league, season, date, home, away, home/away npxG, goals, minutes 11 v 11 (cut at
first red card), home/away manager, each manager's tenure year at that date, each team's recent luck (goals minus
npxG over its previous 10 league matches, computed only from earlier matches), pre-season log squad value
(previous summer), league-average ClubElo at season start.
Sources: Understat via soccerdata (npxG, shots with minute and situation); football-data.co.uk (results backup);
transfermarkt-datasets (manager per match, squad values) if its licence allows, otherwise data/spells.csv from
Wikipedia with a URL per row; ClubElo via soccerdata (league averages only). Hand files: data/managers_2026_27.csv,
data/international_matches.csv, data/raw/united_2026_27_fotmob.csv (local only).
Also data/processed/team_matches.parquet (one row per team per match) for style metrics only.
Coverage report: Pochettino (Tottenham, PSG, Chelsea), Carrick, United 2025-26 and 2026-27.
Tests: no team appears twice for one match in matches.parquet; luck uses only past matches; name matching.
```

## 2. Synthetic data and recovery test (before touching real data)
```
Write a simulator that generates leagues from the exact model in docs/methods.md sections 1-2 with known
parameters: league offsets, home advantage by league-season, club effects, squad value effect, managers who
move clubs, managers whose quality drifts (OU with known gamma), tenure effects, luck term. Then build
pochettino_fit/manager_model.py (section 1-2: pairwise, one row per match, OU per half-season block with
exp(-gamma*dt), non-centred, international term) and show it recovers the true values: report coverage of
50% and 90% intervals for every parameter type, divergences, R-hat, ESS, run time.
Fallback if sampling is too slow: season blocks instead of half-seasons, then ADVI, and say which was used.
```

## 3. Fit on real data
```
Fit the manager model on matches.parquet plus the international matches. Report: Pochettino's mu and current
theta with 90% intervals; the contribution of each of his spells (refit leaving each out); with and without the
USA spell; Carrick's effect; United's club effect; gamma and the half-life ln2/gamma; tenure effects; beta;
home advantage by league-season; league offsets. Sensitivity: league bridge alpha at +/- 1 se. Write all results
to docs/data/model_summary.json and add a results section to docs/methods.md generated from that file.
```

## 4. Finishing and United profile
```
Finishing (methods section 3): career non-penalty goals and npxG for the current United squad, phi_p with
variance 0.85*npxG_p and prior sd 0.05, and the team multiplier f_U with an interval.
United profile: 2025-26 and 2026-27 league style metrics, recency-weighted, shot-based ones in level or one-goal
game states. Report each with its interval and league percentile, and mark which values come from local-only
FotMob data (not publishable).
```

## 5. Style fit
```
Fit the style model (methods section 4a) per metric on team_matches.parquet. Compute Pochettino's and Carrick's
style effects, the predicted change Delta = theta_P - theta_Carrick per metric, D^2 with the inverse correlation
matrix for every posterior draw, and D^2's percentile among all real managerial changes in the data. Imposition
strength. Weakness check on xG per shot against, big chances against, set-piece xG against. Plain-English summary
written to docs/data/fit.json.
```

## 6. Frozen forecast (before his first league match)
```
Fit the log-link attack/defence model (methods 5a). Strip Carrick's effect and tenure effect from United's
current attack and defence (5b). Simulate every remaining league fixture for every team with the match-level xG
shock and Poisson goals (5c), for two runs: baseline new manager and Pochettino. Outputs: United points for the
rest of the season (50% and 80% intervals), final position distribution, P(top half), P(bottom three), his
contribution with interval, and detectability (section 6).
Back-test the forecast method on past mid-season appointments: coverage, PIT, log score vs simpler models.
Write docs/forecasts/<YYYY-MM-DD>.json including model version, data cut-off, git commit and squad-freeze note.
The command must refuse to overwrite an existing forecast file.
Fallback if short of time: skip the full league simulation (United points only) and say so in the file.
```

## 7. Tracker
```
pochettino_fit/tracker.py: after I add a new United match to the data, refit (or normal update between
refits), and append to docs/data/tracker.json: forecast npxG difference and interval vs actual, cumulative
points vs the frozen band, chance quality conceded vs pre-appointment level, updated posterior for his
effect. Never touch files in docs/forecasts/. Test that it refuses to.
```

## 8. Page and publish
```
docs/index.html, dark theme, no CDN, no photos or logos, mobile friendly. Sections: 1) The fit (predicted
style change chart, weakness check), 2) The forecast (frozen date, points band, baseline vs Pochettino, his
contribution), 3) The tracker, 4) How much the data can tell us (detectability), 5) Method summary, limits,
data attribution, link to methods.md. README with the same structure and reproduction steps. Check nothing
in data/raw is committed.
```
