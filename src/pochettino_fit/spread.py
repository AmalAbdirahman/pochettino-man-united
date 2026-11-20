"""Spread of the frozen forecast's final points (page note on the bottom-three paradox).

python -m pochettino_fit.spread   -> docs/data/forecast_spread.json

The frozen file stores quantiles, not standard deviations. This re-runs the forecast simulation in memory
with the same code path, seed and inputs as forecast.run (it does not modify forecast.py or the frozen
file), refuses to write anything unless the re-run reproduces the frozen summary exactly, and then
records the standard deviation of final and rest-of-season points for both runs.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pochettino_fit import forecast as F
from pochettino_fit import season_sim as sim
from pochettino_fit.config import ROOT, load_config
from pochettino_fit.strength_model import fit_strength, recency
from pochettino_fit.united_effects import manager_draws

CHECKED = ("united_now", "baseline", "pochettino", "contribution")


def rerun(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Same steps and random-number order as forecast.run.

    Returns the summary, the raw runs and United's index.
    """
    fc = cfg["forecast"]
    rng = np.random.default_rng(fc["seed"])
    m, pen_rate, pen_xg = F.league_matches(cfg)
    cutoff = pd.Timestamp(cfg["features"]["data_cutoff"])
    played = m[(m.season == fc["season"]) & (m.date <= cutoff)]
    teams = sorted(set(played.home_team) | set(played.away_team))
    st = fit_strength(m[m.date <= cutoff], cutoff + pd.Timedelta(days=1), fc["half_life_days"])
    fix = sim.remaining_fixtures(played, teams)
    utd = cfg["profile"]["team_tm_id"]
    md = manager_draws(cfg, m, st, utd)
    n = len(md["baseline"])
    par = st.draws(n, rng)
    fu = F.cfg_finishing_draws(cfg, n, rng)
    noise = sim.Noise.make(n, len(fix), len(teams), rng)
    base = sim.current_table(played, teams)
    w = recency(m[m.date <= cutoff].date, cutoff + pd.Timedelta(days=1), fc["half_life_days"])
    k = sim.goal_calibration(m[m.date <= cutoff], w, pen_rate)
    sk = (st.shock_sd if fc["shock"] else 0.0, k if fc["calibrate"] else 1.0)
    runs = {r: F.simulate_run(st, par, md[r], fu, fix, teams, utd, noise, base, pen_rate, sk)
            for r in ("baseline", "pochettino")}
    out = F.assemble(cfg, runs, md, st, played, fix, teams, utd, pen_rate, pen_xg)
    return out, runs, teams.index(utd)


def mismatches(out: dict[str, Any], frozen: dict[str, Any]) -> list[str]:
    """Fields that differ after a JSON round trip (the frozen file has string keys, e.g. positions)."""
    same = json.loads(json.dumps({k: out[k] for k in CHECKED}))
    return [f"{k}.{f}" for k in CHECKED for f in set(same[k]) | set(frozen[k])
            if same[k].get(f) != frozen[k].get(f)]


def spread(runs: dict[str, Any], ui: int) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for r, v in runs.items():
        pts, rest, pos = v["table"]["pts"][:, ui], v["table"]["rest_pts"][:, ui], v["table"]["pos"][:, ui]
        res[r] = {"final_points_sd": float(np.std(pts)), "final_points_mean": float(np.mean(pts)),
                  "rest_points_sd": float(np.std(rest)), "position_sd": float(np.std(pos))}
    return res


def main() -> None:
    cfg = load_config()
    fpath = sorted((ROOT / cfg["paths"]["forecasts"]).glob("*.json"))[0]
    frozen = json.loads(fpath.read_text(encoding="utf-8"))
    out, runs, ui = rerun(cfg)
    bad = mismatches(out, frozen)
    if bad:
        raise SystemExit(f"re-run does not reproduce {fpath.name} in {bad}; nothing written")
    res = {"source": f"re-run of forecasts/{fpath.name}, reproduced exactly in {list(CHECKED)}",
           "draws": int(len(runs["baseline"]["table"]["pts"])), **spread(runs, ui)}
    path = ROOT / cfg["paths"]["docs_data"] / "forecast_spread.json"
    Path(path).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
