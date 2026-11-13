# CLAUDE.md: Pochettino at United (fit, forecast, tracker)

## What this is
Mauricio Pochettino was appointed Manchester United head coach on 13 November 2026 (starting 18 November, deal to summer 2029), after Michael Carrick was sacked following the club's worst start since 1986.

The project answers three questions, in this order:
1. **Fit:** how well do Pochettino's teams' habits match what this United squad does, and does his record address United's specific weaknesses (transition defending, chance quality conceded, set pieces)?
2. **Forecast:** a pre-registered prediction of United's underlying performance and points for the rest of the 2026-27 Premier League season, published BEFORE his first league match (commit timestamp is the proof).
3. **Tracker:** after each league match, update the estimate (Bayesian updating) and compare reality with the forecast, publicly, whatever it shows.

Inspired by the way Liverpool's research team judged managers on underlying performance relative to resources, not results or reputation (Ian Graham, *How to Win the Premier League*).

## The central statistical point
United have under-performed their xG by about 5 points this season. Some improvement is expected under ANY coach (regression to the mean). The forecast must separate:
- baseline: United's expected performance with an average replacement coach (Elo strength + regression of finishing luck + typical new-coach effect), and
- Pochettino's estimated effect on top of that, with its uncertainty.
Never credit Pochettino with improvement the baseline already predicts.

## Honesty rules
1. Every number on the page comes from a real run. No hand-typed figures.
2. Uncertainty on everything: intervals, not single numbers.
3. The forecast is frozen once published. Later changes go in a new, dated version; the original stays visible.
4. Report forecast misses as prominently as hits.
5. Call it a statistical model, not AI. No photos, crests, logos or broadcaster branding.
6. Respect data terms. FotMob data stays in `data/raw/` (git-ignored) and is never republished; publish only our own derived figures, with attribution. Check every other source's terms before downloading.
7. Limits on the page: public data only, league matches only, xG models differ by source, things the model cannot see (staff, injuries, dressing room).

## Method (full maths in docs/methods.md, version 2)
1. Unit: one LEAGUE match, ONE row per match (home team's view), never two mirrored rows. No cups or European games. Outcome y = home npxG minus away npxG per 90 at 11 v 11; truncate at the first red card and weight by minutes.
2. Single-stage pairwise model (no two-stage Elo residuals):
   y = h(league, season) + (team strength_home - team strength_away) + (manager effect_home - manager effect_away) + (tenure effect_home - tenure effect_away) + beta * (recent luck_home - recent luck_away) + noise.
   - team strength = league offset + club effect + b * lagged log squad value (pre-season value, so a manager's own success does not leak in);
   - league offsets: hierarchical, anchored by the league-average ClubElo per season (an exogenous bridge between leagues; never team-level Elo);
   - home advantage varies by league and season (2020-21 behind closed doors);
   - recent luck = goals minus xG over the previous 10 league matches, so new-manager bounces are measured net of regression to the mean.
3. Manager effect = long-run level mu_m + Ornstein-Uhlenbeck deviation with decay rate gamma > 0 (rho^dt = exp(-gamma dt)), discretised per half-season block, not per match. Partial pooling, non-centred PyMC. International spells enter as a separate noisier observation scaled by lambda (wide prior), for the effect only, never for style.
4. United profile: 2025-26 and 2026-27 league matches, recency-weighted, using fast-stabilising metrics; shot-based metrics in level or one-goal game states where possible.
5. Finishing: phi_p estimated with variance proportional to npxG_p (Bernoulli shot noise, about 0.85 * npxG_p), tight prior (sd about 0.05); team multiplier from the current squad's xG shares.
6. Style: predict United's style under Pochettino as United club level + his style effect, and compare with United's current style (club level + outgoing manager's effect). The quantity of interest is the predicted CHANGE (his effect minus the outgoing manager's), measured with the inverse correlation matrix and propagated over posterior draws. Do not rescale shrunk effects by their spread.
7. Forecast: log-link attack and defence model for npxG (no negative rates), with a match-level xG shock before the Poisson goal draw (overdispersion). United's baseline = current xG-based strength MINUS the outgoing manager's estimated effect and tenure effect; then add (a) an average new manager (baseline run) or (b) Pochettino's effect (Pochettino run), each with the year-one tenure effect. His contribution = run (b) minus run (a). Squad frozen at the freeze date.
8. Detectability: state how many matches are needed to separate his effect from zero.
9. Tracker: refit after each match; chance quality conceded panel; frozen forecast never edited.
10. Validation: synthetic recovery test (known effects, including drifting ones); back-test on past mid-season appointments with coverage, PIT and log score; keep a component only if it helps.

## Stack
- Python 3.12, pandas, numpy, scipy, statsmodels, PyMC, matplotlib. `pyproject.toml`, `ruff`, `pytest`.
- Front end: static `docs/index.html` with inline CSS and JS, reading `docs/data/*.json`. No CDN. Dark theme.
- GitHub Actions runs `pytest -q`. A second workflow may rebuild the tracker JSON when I push new match data (never auto-scrapes).

## Layout
```
data/raw/            downloaded or hand-collected source data (git-ignored)
data/                spells.csv, managers_2026_27.csv, international_matches.csv, small derived tables with sources
src/pochettino_fit/  fetch.py features.py strength_model.py manager_model.py profile.py fit.py forecast.py tracker.py backtest.py export.py
docs/                index.html, methods.md, data/*.json, forecasts/ (frozen, dated forecast files)
tests/
config.yaml
```

## Commands
```
python -m pochettino_fit.fetch
python -m pochettino_fit.manager_model
python -m pochettino_fit.fit
python -m pochettino_fit.forecast --freeze     # writes docs/forecasts/<date>.json; refuses to overwrite
python -m pochettino_fit.tracker               # after each new match
python -m pochettino_fit.backtest
python -m pochettino_fit.export
pytest -q
python -m http.server 8000 --directory docs
```

## Definition of done (every task)
pytest passes, the command to see it working is given, anything skipped or uncertain is listed, and the work is committed.
