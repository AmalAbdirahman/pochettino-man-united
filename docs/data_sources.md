# Data sources and terms

Checked on 2026-11-13. Raw files live in `data/raw/` (git-ignored) and are never republished.
Only our own derived figures are published, with attribution. `data/raw/MANIFEST.json` records the
URL, download time and SHA-256 of every raw file.

| Source | What we use | Coverage | Terms found | How we use it |
|---|---|---|---|---|
| Understat, via the frozen archive in [peteowen1/pannadata](https://github.com/peteowen1/pannadata) (release `understat-latest`) | Shot-level xG, minute, situation; match deep completions and PPDA | Big five, 2014-15 to 2026-04-18 | None stated by the archive or by Understat. understat.com has shown a maintenance page since about 2026-11-02. | npxG and style inputs. Raw shots stay local. |
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | League games, managers per game, card minutes, player valuations | 2012-13 to 2026-07-06 | Repository licence CC0-1.0. Built from Transfermarkt data; Transfermarkt's own terms could not be retrieved. | Manager spells, red-card truncation, pre-season squad value. Derived figures only. |
| [football-data.co.uk](https://www.football-data.co.uk) | Results; HxG/AxG in 2026-27 | Big five, 2014-15 to 2026-11-08 | No reuse terms published; disclaimer on accuracy only. | Team-name matching and 2026-27 results. The 2026-27 xG matches FotMob's to within 0.01 per team-match, so we treat it as FotMob data: derived figures only. |
| FotMob (hand-collected notes) | United 2026-27 match stats | 10 league matches | Not for republication. | United's 2026-27 npxG, local only. |
| [ClubElo](http://clubelo.com) | Not used | — | No terms found. API returned HTTP 502 on 2026-11-13. | **Not used.** Planned as a league-bridge prior; the league offset cancels in the within-league model, so no league rating enters it. |
| Wikipedia season articles (CC BY-SA 4.0) | 2026-27 head coaches and changes | To 2026-11-08 | CC BY-SA 4.0 | `data/managers_2026_27.csv`, one URL per row. Not yet verified by hand. |
| Wikipedia, 2026-27 Manchester United season (CC BY-SA 4.0) | United first-team squad as of 2026-11-08 | 33 players | CC BY-SA 4.0 | `data/united_squad_2026_27.csv` (finishing, prompt 4), Understat ids matched by name. Not yet verified by hand. |

Not used: FBref (Opta advanced data removed in January 2026).
