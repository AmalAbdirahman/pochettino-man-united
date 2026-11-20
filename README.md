# Pochettino at United: fit, forecast, tracker

Mauricio Pochettino became Manchester United head coach in November 2026. This project asks three questions,
in this order, using public data and a statistical model (not AI):

1. **The fit.** How well do his teams' habits match what this United squad does, and does his record address
   United's weak points (chance quality conceded, set pieces)?
2. **The forecast.** United's underlying performance and points for the rest of the 2026-27 Premier League
   season, frozen **before his first league match**. The commit timestamp is the proof.
3. **The tracker.** After every league match, reality against the frozen forecast, whatever it shows.

The page reads every number from the JSON files in `docs/`. Nothing on it is typed by hand.

## In short

Static copy of the page's generated summary (2026-11-20). Sources: the frozen forecast
`docs/forecasts/2026-11-20.json` (points, position, detectability) and the style fit `docs/data/fit.json`
(style change). The page builds these sentences from those files; if they ever disagree, the files win.

- Compared with an average new coach, Pochettino adds an estimated +0.6 points over the rest of the season
  (80% range −5 to +6), which cannot be told apart from zero.
- The frozen forecast has United finishing on 50 points (80% range 39 to 61), with a median position of 11th.
- The clearest style change the model predicts is PPDA (lower = more pressing): from 11.6 now to 9.0 under
  him, the largest of 2 changes whose 80% interval excludes zero.
- Telling an effect of 0.10 npxG per 90 apart from zero would take about 760 matches; the rest of this season
  is 28.

Inspired by the way Liverpool's research team judged managers on underlying performance relative to
resources, not results or reputation (Ian Graham, *How to Win the Premier League*).

## What is on the page

1. **The fit:** the predicted change in United's style (his estimated effect minus Carrick's) for twelve
   metrics, with intervals; a weakness check; his estimated effect in every half-season he has managed.
2. **The forecast:** final points with an average new coach and with Pochettino, his contribution (the
   difference), final-position probabilities and every remaining fixture. The forecast file in
   `docs/forecasts/` is never edited. A later version would get a new date, and the original stays visible.
3. **The tracker:** cumulative points against the frozen 50% and 80% bands, each match against its frozen
   expectation (misses are labelled as misses), and an updated estimate of his effect.
4. **How much the data can tell us:** how many matches it would take to tell his effect from zero.
   This season will mostly show noise, and the page says so.
5. **Method, limits and data:** a summary, the back-test, the limits and attribution.

## The central statistical point

United had under-performed their xG this season, so some improvement is expected under any coach
(regression to the mean, plus the usual new-coach effect). The forecast separates a **baseline** (an average
replacement coach) from **Pochettino's estimated effect** on top, and never credits him with improvement the
baseline already predicts. Full maths: [docs/methods.md](docs/methods.md).

## Reproduce

Python 3.12. Install with `pip install -e ".[dev]"`. On Windows use the project virtual environment's Python
(`.venv/Scripts/python.exe`).

```
python -m pochettino_fit.fetch            # downloads public sources into data/raw/ (git-ignored)
python -m pochettino_fit.dataset          # match tables in data/processed/
python -m pochettino_fit.fd_stats         # football-data shots and corners
python -m pochettino_fit.fit              # manager model (PyMC); --summarise writes docs/data/model_summary.json
python -m pochettino_fit.profile          # United profile and finishing skill
python -m pochettino_fit.style --fit      # style fit -> docs/data/fit.json
python -m pochettino_fit.backtest         # back-test -> docs/data/backtest.json
python -m pochettino_fit.forecast         # preview only; --freeze refuses to overwrite or to run after the first match
python -m pochettino_fit.spread           # re-runs the frozen forecast in memory -> docs/data/forecast_spread.json
python -m pochettino_fit.tracker          # after each new match -> docs/data/tracker.json (see docs/tracker_plan.md)
python -m pochettino_fit.export           # manifest only; every step above that writes docs/data refreshes it
pytest -q
python -m http.server 8000 --directory docs   # then open http://localhost:8000
```

United's 2026-27 match statistics come from hand-collected FotMob notes in `data/raw/`, which are not in the
repository. The tracker needs them, so a fresh clone can run everything except the tracker update.

## Data and terms

See [docs/data_sources.md](docs/data_sources.md). In short: Understat (via an archive to April 2026),
transfermarkt-datasets (CC0), football-data.co.uk, Wikipedia (CC BY-SA 4.0) and FotMob. Raw
data stays in `data/raw/` and is never committed. ClubElo is listed in the sources table but **not used**:
no league rating enters the model. FotMob data is never republished. The page shows only
results and our own model outputs derived from it. All club and player names refer to real public match
data, and there are no photos, crests, logos or broadcaster branding.

## Limits

- Public data only, league matches only.
- xG providers are mixed without calibration: Understat to 18 April 2026, FotMob-based xG for 2026-27,
  no overlapping season (see [docs/deviations.md](docs/deviations.md)).
- Comparing managers across leagues assumes an effect in xG difference means the same in every league.
- The model cannot see staff, injuries, the dressing room or January transfers.
- The style metrics have no possession, field tilt or big chances.
- Whether style fit predicts results has not been back-tested, so the fit section is descriptive only.

## Roadmap (not done yet)

- Full refits of the manager model during the season. The tracker currently uses a normal update between
  refits.
- A back-test of the manager terms at past appointments. The back-test covers the xG simulation only.
- A back-test of whether style fit predicts results.

## Licence

See [LICENSE](LICENSE).
