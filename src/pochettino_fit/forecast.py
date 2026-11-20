"""Pre-registered forecast of United's rest of season (prompt 6; methods sections 5 and 6).

python -m pochettino_fit.forecast            preview: data/processed/forecast_preview.json
python -m pochettino_fit.forecast --freeze   writes docs/forecasts/<YYYY-MM-DD>.json; refuses to overwrite

Two runs share every random number: (a) baseline = United with Carrick's (and Amorim's) effect and tenure
stage stripped out, plus an average new manager in tenure year 1; (b) the same with Pochettino's
projected effect. His contribution = (b) - (a), draw by draw.
"""

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pochettino_fit import season_sim as sim
from pochettino_fit.config import ROOT, load_config
from pochettino_fit.strength_model import Strength, fit_strength, recency
from pochettino_fit.united_effects import manager_draws, split_effect


def league_matches(cfg: dict[str, Any]) -> tuple[pd.DataFrame, float, float]:
    """League rows in the strength window, npxG filled, plus penalty goal and penalty xG rates."""
    fc = cfg["forecast"]
    m = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "matches.parquet")
    m = m[(m.league == fc["league"]) & (m.date >= pd.Timestamp(fc["window_start"]))]
    ref = m[m.season == fc["rate_season"]]
    pen_rate = float(pd.concat([ref.home_pen_goals, ref.away_pen_goals]).mean())
    return sim.fill_npxg(m, pen_rate), pen_rate, pen_rate   # penalty xG per team-match ~ penalty goals


def summarise_points(x: np.ndarray, levels: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"mean": float(x.mean()), "median": float(np.median(x))}
    for lv in levels:
        lo, hi = np.quantile(x, [(1 - lv) / 2, (1 + lv) / 2])
        out[f"{int(lv * 100)}"] = [float(lo), float(hi)]
    return out


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    fc = cfg["forecast"]
    rng = np.random.default_rng(fc["seed"])
    m, pen_rate, pen_xg = league_matches(cfg)
    cutoff = pd.Timestamp(cfg["features"]["data_cutoff"])
    played = m[(m.season == fc["season"]) & (m.date <= cutoff)]
    teams = sorted(set(played.home_team) | set(played.away_team))
    st = fit_strength(m[m.date <= cutoff], cutoff + pd.Timedelta(days=1), fc["half_life_days"])
    fix = sim.remaining_fixtures(played, teams)
    utd = cfg["profile"]["team_tm_id"]
    md = manager_draws(cfg, m, st, utd)
    n = len(md["baseline"])
    par = st.draws(n, rng)
    fu = cfg_finishing_draws(cfg, n, rng)
    noise = sim.Noise.make(n, len(fix), len(teams), rng)
    base = sim.current_table(played, teams)
    w = recency(m[m.date <= cutoff].date, cutoff + pd.Timedelta(days=1), fc["half_life_days"])
    k = sim.goal_calibration(m[m.date <= cutoff], w, pen_rate)
    sk = (st.shock_sd if fc["shock"] else 0.0, k if fc["calibrate"] else 1.0)
    runs = {r: simulate_run(st, par, md[r], fu, fix, teams, utd, noise, base, pen_rate, sk)
            for r in ("baseline", "pochettino")}
    out = assemble(cfg, runs, md, st, played, fix, teams, utd, pen_rate, pen_xg)
    out["simulation"].update({"shock_used": fc["shock"], "calibration_used": fc["calibrate"],
                              "goal_calibration_k_estimated": k, "model_choice": fc["model_choice"]})
    return out


def cfg_finishing_draws(cfg: dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    prof = json.loads((ROOT / cfg["paths"]["docs_data"] / "profile.json").read_text(encoding="utf-8"))
    f = prof["finishing"]["f_U"]
    return rng.normal(f["mean"], f["sd"], size=n)


def expected_npxg(st: Strength, par: dict[str, np.ndarray], eff: np.ndarray, fix: pd.DataFrame,
                  utd: int) -> tuple[np.ndarray, np.ndarray]:
    """(n, F) expected npxG for home and away sides; United's attack/defence get the manager shift."""
    hi = np.array([st.index(t) for t in fix.home_team])
    ai = np.array([st.index(t) for t in fix.away_team])
    a, d = par["a"].copy(), par["d"].copy()
    u = st.index(utd)
    ea, ed = split_effect(eff, st, u)
    a[:, u] += ea
    d[:, u] += ed
    lh = np.exp(par["c"][:, None] + par["h"][:, None] + a[:, hi] - d[:, ai])
    la = np.exp(par["c"][:, None] + a[:, ai] - d[:, hi])
    return lh, la


def simulate_run(st: Strength, par: dict[str, np.ndarray], eff: np.ndarray, fu: np.ndarray,
                 fix: pd.DataFrame, teams: list[int], utd: int, noise: sim.Noise,
                 base: dict[str, np.ndarray], pen_rate: float, sk: tuple[float, float]) -> dict[str, Any]:
    lh, la = expected_npxg(st, par, eff, fix, utd)
    s, k = sk
    f_h = np.where((fix.home_team == utd).to_numpy()[None, :], fu[:, None], 1.0)
    f_a = np.where((fix.away_team == utd).to_numpy()[None, :], fu[:, None], 1.0)
    gh = sim.goals(lh, noise.z_home, noise.u_home, s, f_h, pen_rate, k)
    ga = sim.goals(la, noise.z_away, noise.u_away, s, f_a, pen_rate, k)
    pos = {t: i for i, t in enumerate(teams)}
    tab = sim.season_table(gh, ga, fix.home_team.map(pos).to_numpy(), fix.away_team.map(pos).to_numpy(),
                           len(teams), base, noise.tie)
    return {"table": tab, "lh": lh, "la": la, "gh": gh, "ga": ga}


def run_summary(r: dict[str, Any], ui: int, levels: list[float]) -> dict[str, Any]:
    pos = r["table"]["pos"][:, ui]
    n_teams = r["table"]["pos"].shape[1]
    return {"rest_of_season_points": summarise_points(r["table"]["rest_pts"][:, ui], levels),
            "final_points": summarise_points(r["table"]["pts"][:, ui], levels),
            "position_probabilities": {int(p): float(np.mean(pos == p)) for p in range(1, n_teams + 1)},
            "p_top_half": float(np.mean(pos <= n_teams // 2)),
            "p_bottom_three": float(np.mean(pos > n_teams - 3)),
            "p_top_four": float(np.mean(pos <= 4)),
            "median_position": float(np.median(pos))}


def united_fixtures(runs: dict[str, Any], fix: pd.DataFrame, utd: int, names: dict[int, str],
                    lv: float, n_keep: int) -> list[dict[str, Any]]:
    """Per remaining United fixture: expected npxG difference (United minus opponent) in both runs, and
    United's points in the first n_keep simulation draws as a digit string (for the tracker's
    cumulative band; draws are shared across fixtures, so the string keeps their correlation)."""
    r, b = runs["pochettino"], runs["baseline"]
    out = []
    for k, row in enumerate(fix.itertuples()):
        if utd not in (row.home_team, row.away_team):
            continue
        home = row.home_team == utd
        sgn = 1 if home else -1
        diff = (r["lh"][:, k] - r["la"][:, k]) * sgn
        base = (b["lh"][:, k] - b["la"][:, k]) * sgn
        lo, hi = np.quantile(diff, [(1 - lv) / 2, (1 + lv) / 2])
        gf, ga = (r["gh"][:, k], r["ga"][:, k]) if home else (r["ga"][:, k], r["gh"][:, k])
        pts = sim.points(gf, ga).astype(int)
        opp = row.away_team if home else row.home_team
        out.append({"opponent": names.get(opp, str(opp)), "opponent_tm_id": int(opp),
                    "venue": "home" if home else "away", "npxg_diff_mean": float(diff.mean()),
                    "npxg_diff_interval": [float(lo), float(hi)], "interval": lv,
                    "baseline_npxg_diff_mean": float(base.mean()),
                    "p_win_draw_loss": [float(np.mean(pts == v)) for v in (3, 1, 0)],
                    "expected_points": float(pts.mean()),
                    "points_draws": "".join(map(str, pts[:n_keep]))})
    return out


def detectability(cfg: dict[str, Any], md: dict[str, np.ndarray], n_remaining: int) -> dict[str, Any]:
    """Methods section 6: matches needed to tell his effect from zero."""
    d = cfg["detectability"]
    sigma, theta = float(np.mean(md["sigma"])), md["theta_p"]
    k = (d["alpha_z"] + d["power_z"]) ** 2 * sigma**2
    sd0 = float(np.std(theta))
    post_sd = {int(n): float((1 / sd0**2 + n / sigma**2) ** -0.5) for n in (n_remaining, 38, 76, 114)}
    return {"sigma": sigma, "effect_mean": float(np.mean(theta)), "effect_sd_now": sd0,
            "matches_needed": {f"{e:.2f}": float(k / e**2) for e in (abs(float(np.mean(theta))), 0.1, 0.2)},
            "posterior_sd_after_n_matches": post_sd,
            "note": "n = (z_a + z_b)^2 sigma^2 / effect^2 at full weight per match; posterior sd assumes "
                    "a normal update with w = 1 per match."}


def assemble(cfg: dict[str, Any], runs: dict[str, Any], md: dict[str, np.ndarray], st: Strength,
             played: pd.DataFrame, fix: pd.DataFrame, teams: list[int], utd: int, pen_rate: float,
             pen_xg: float) -> dict[str, Any]:
    fc, lv = cfg["forecast"], cfg["forecast"]["interval_levels"]
    ui = teams.index(utd)
    names = team_names(played)
    contrib = runs["pochettino"]["table"]["rest_pts"][:, ui] - runs["baseline"]["table"]["rest_pts"][:, ui]
    return {
        "united_now": {"played": int(((played.home_team == utd) | (played.away_team == utd)).sum()),
                       "points": float(sim.current_table(played, teams)["pts"][ui]),
                       "remaining": int(((fix.home_team == utd) | (fix.away_team == utd)).sum())},
        "baseline": run_summary(runs["baseline"], ui, lv),
        "pochettino": run_summary(runs["pochettino"], ui, lv),
        "contribution": {"points": summarise_points(contrib, lv), "p_positive": float(np.mean(contrib > 0)),
                         "p_negative": float(np.mean(contrib < 0)),
                         "npxg_diff_per90": summarise_points(md["theta_p"], lv)},
        "manager_terms": {k: summarise_points(md[k], lv) for k in ("stripped", "baseline", "pochettino")},
        "united_fixtures": united_fixtures(runs, fix, utd, names, lv[-1], fc["fixture_draws_kept"]),
        "detectability": detectability(cfg, md, int(((fix.home_team == utd) | (fix.away_team == utd)).sum())),
        "simulation": {"draws": len(contrib), "fixtures_simulated": len(fix), "teams": len(teams),
                       "shock_sd": st.shock_sd, "penalty_goals_per_team_match": pen_rate,
                       "penalty_xg_subtracted_where_npxg_missing": pen_xg,
                       "strength_half_life_days": fc["half_life_days"],
                       "window_start": str(fc["window_start"])},
    }


def team_names(m: pd.DataFrame) -> dict[int, str]:
    pairs = pd.concat([m[["home_team", "home_name"]].set_axis(["t", "n"], axis=1),
                       m[["away_team", "away_name"]].set_axis(["t", "n"], axis=1)]).drop_duplicates("t")
    return dict(zip(pairs.t.astype(int), pairs.n, strict=True))


def git_state() -> dict[str, Any]:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def header(cfg: dict[str, Any], today: dt.date) -> dict[str, Any]:
    fc = cfg["forecast"]
    bt = ROOT / cfg["paths"]["docs_data"] / "backtest.json"
    return {"title": "Pre-registered forecast: Manchester United 2026-27 Premier League, rest of season",
            "frozen_on": str(today), "model_version": cfg["model_version"], "method": "docs/methods.md",
            "data_cutoff": str(cfg["features"]["data_cutoff"]),
            "first_league_match": str(fc["first_league_match"]),
            "git": git_state(), "squad_freeze": fc["squad_freeze_note"], "deviations": fc["deviations"],
            "backtest": json.loads(bt.read_text(encoding="utf-8"))["summary"] if bt.exists() else None,
            "units": "points; npxG difference per 90 (United minus opponent)"}


def freeze_path(cfg: dict[str, Any], today: dt.date) -> Path:
    return ROOT / cfg["paths"]["forecasts"] / f"{today.isoformat()}.json"


def write_frozen(cfg: dict[str, Any], out: dict[str, Any], today: dt.date) -> Path:
    """Write the frozen forecast; never overwrite an existing file or freeze after the first match."""
    path = freeze_path(cfg, today)
    if path.exists():
        raise FileExistsError(f"{path} exists: frozen forecasts are never overwritten. "
                              "Publish a new dated version.")
    if today >= pd.Timestamp(cfg["forecast"]["first_league_match"]).date():
        raise RuntimeError("Too late: the first league match has started; a frozen forecast must predate it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    return path


def print_summary(out: dict[str, Any]) -> None:
    u = out["united_now"]
    print(f"United now: {u['points']:.0f} pts from {u['played']}; {u['remaining']} to play")
    for k in ("baseline", "pochettino"):
        r = out[k]
        rp = r["rest_of_season_points"]
        print(f"{k:>10}: rest {rp['median']:.0f} pts (50% {rp['50']}, "
              f"80% {rp['80']}); final {r['final_points']['median']:.0f}; "
              f"top half {r['p_top_half']:.2f}; bottom three {r['p_bottom_three']:.3f}; "
              f"median pos {r['median_position']:.0f}")
    c = out["contribution"]["points"]
    pc = out["contribution"]
    print(f"contribution: {c['mean']:+.2f} pts (80% {c['80']}), P(>0) {pc['p_positive']:.2f}, "
          f"P(<0) {pc['p_negative']:.2f}")
    print("detectability:", json.dumps(out["detectability"]["matches_needed"]),
          json.dumps(out["detectability"]["posterior_sd_after_n_matches"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", action="store_true")
    a = p.parse_args()
    cfg = load_config()
    today = dt.date.today()
    out = {**header(cfg, today), **run(cfg)}
    print_summary(out)
    if a.freeze:
        print("written", write_frozen(cfg, out, today))
    else:
        prev = ROOT / cfg["paths"]["processed"] / "forecast_preview.json"
        prev.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("preview written", prev)


if __name__ == "__main__":
    main()
