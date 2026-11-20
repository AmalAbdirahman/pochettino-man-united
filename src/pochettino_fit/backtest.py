"""Back-test of the forecast machinery on past mid-season appointments (methods section 8.2).

python -m pochettino_fit.backtest   -> docs/data/backtest.json

For every in-season manager change in the big five (new manager's first league match with at least
min_before matches played and min_after to play), cut all data the day before, fit the attack/defence
model on the league's recent matches, simulate the club's actual remaining fixtures and score its
rest-of-season points. Compared: the xG simulation with and without the match-level shock and with and
without the goal calibration k, and a naive results-only model (the club's W/D/L rates so far).

Not back-tested: the manager terms (stripping the outgoing manager, adding the incoming one). Doing that
without leakage needs a refit of the manager model with data cut at every appointment (hours each).
"""

from typing import Any

import numpy as np
import pandas as pd

from pochettino_fit import season_sim as sim
from pochettino_fit.config import ROOT, load_config
from pochettino_fit.strength_model import fit_strength, recency

MODELS = {"xg_shock_cal": (True, True), "xg_cal": (False, True), "xg_shock_raw": (True, False),
          "xg_raw": (False, False)}


def appointments(m: pd.DataFrame, min_before: int, min_after: int) -> pd.DataFrame:
    """One row per in-season manager change: league, season, team, cutoff date, outgoing, incoming."""
    long = pd.concat([
        m[["league", "season", "date", "home_team", "home_manager"]].set_axis(
            ["league", "season", "date", "team", "manager"], axis=1),
        m[["league", "season", "date", "away_team", "away_manager"]].set_axis(
            ["league", "season", "date", "team", "manager"], axis=1)]).sort_values("date")
    rows = []
    for (lg, s, t), g in long.groupby(["league", "season", "team"]):
        mg = g.manager.to_numpy()
        for i in range(min_before, len(g) - min_after + 1):
            if isinstance(mg[i], str) and isinstance(mg[i - 1], str) and mg[i] != mg[i - 1]:
                rows.append({"league": lg, "season": s, "team": t, "cutoff": g.date.iloc[i],
                             "outgoing": mg[i - 1], "incoming": mg[i]})
    return pd.DataFrame(rows)


def interval_covers(x: np.ndarray, y: float, level: float) -> bool:
    lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2], method="inverted_cdf")
    return bool(lo <= y <= hi)


def pit(x: np.ndarray, y: float, u: float) -> float:
    """Randomised PIT for a discrete predictive: P(X < y) + u P(X = y)."""
    return float(np.mean(x < y) + u * np.mean(x == y))


def log_score(x: np.ndarray, y: float, support: int) -> float:
    """log P(X = y) from simulated counts, with add-half smoothing over 0..support."""
    return float(np.log((np.sum(x == y) + 0.5) / (len(x) + 0.5 * (support + 1))))


def crps(x: np.ndarray, y: float) -> float:
    """Sample CRPS: E|X - y| - E|X - X'| / 2 (sorted-sample formula for the second term)."""
    xs = np.sort(x.astype(float))
    n = len(xs)
    e_xx = 2 * np.sum(xs * (2 * np.arange(1, n + 1) - n - 1)) / n**2
    return float(np.mean(np.abs(xs - y)) - e_xx / 2)


def naive_points(past: pd.DataFrame, team: int, n_fix: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Results-only model: each remaining match is W/D/L with the club's smoothed rates so far."""
    home = past.home_team == team
    gf = np.where(home, past.home_goals, past.away_goals)
    ga = np.where(home, past.away_goals, past.home_goals)
    counts = np.array([np.sum(gf > ga), np.sum(gf == ga), np.sum(gf < ga)]) + 1.0
    res = rng.multinomial(n_fix, counts / counts.sum(), size=n)
    return res[:, 0] * 3 + res[:, 1]


def case_samples(lg: pd.DataFrame, case: Any, bc: dict[str, Any],
                 rng: np.random.Generator) -> dict[str, Any] | None:
    cut = pd.Timestamp(case.cutoff)
    win = lg[(lg.date < cut) & (lg.date >= cut - pd.Timedelta(days=bc["window_days"]))]
    season = lg[lg.season == case.season]
    rest = season[(season.date >= cut) & ((season.home_team == case.team) | (season.away_team == case.team))]
    past = season[(season.date < cut) & ((season.home_team == case.team) | (season.away_team == case.team))]
    st = fit_strength(win, cut, bc["half_life_days"])
    if not set(rest.home_team) | set(rest.away_team) <= set(st.teams):
        return None
    pen = float(pd.concat([win.home_pen_goals, win.away_pen_goals]).mean())
    k = sim.goal_calibration(win, recency(win.date, cut, bc["half_life_days"]), pen)
    n = bc["draws"]
    par = st.draws(n, rng)
    hi, ai = [np.array([st.index(t) for t in rest[c]]) for c in ("home_team", "away_team")]
    lh = np.exp(par["c"][:, None] + par["h"][:, None] + par["a"][:, hi] - par["d"][:, ai])
    la = np.exp(par["c"][:, None] + par["a"][:, ai] - par["d"][:, hi])
    noise = sim.Noise.make(n, len(rest), 1, rng)
    home = (rest.home_team == case.team).to_numpy()
    out = {"naive": naive_points(past, case.team, len(rest), n, rng)}
    for name, (shock, cal) in MODELS.items():
        s, kk = (st.shock_sd if shock else 0.0), (k if cal else 1.0)
        gh = sim.goals(lh, noise.z_home, noise.u_home, s, 1.0, pen, kk)
        ga = sim.goals(la, noise.z_away, noise.u_away, s, 1.0, pen, kk)
        out[name] = np.where(home, sim.points(gh, ga), sim.points(ga, gh)).sum(1)
    actual = np.where(home, sim.points(rest.home_goals, rest.away_goals),
                      sim.points(rest.away_goals, rest.home_goals)).sum()
    return {"samples": out, "actual": float(actual), "n_fix": len(rest), "k": k, "s": st.shock_sd}


def score_case(r: dict[str, Any], u: float) -> dict[str, dict[str, float]]:
    y, sup = r["actual"], 3 * r["n_fix"]
    return {m: {"cover50": interval_covers(x, y, 0.5), "cover80": interval_covers(x, y, 0.8),
                "pit": pit(x, y, u), "log_score": log_score(x, y, sup), "crps": crps(x, y)}
            for m, x in r["samples"].items()}


def aggregate(scores: list[dict[str, dict[str, float]]], bins: int = 10) -> dict[str, Any]:
    out = {}
    for m in scores[0]:
        s = pd.DataFrame([c[m] for c in scores])
        hist = np.histogram(s.pit, bins=bins, range=(0, 1))[0]
        out[m] = {"coverage50": float(s.cover50.mean()), "coverage80": float(s.cover80.mean()),
                  "mean_log_score": float(s.log_score.mean()), "mean_crps": float(s.crps.mean()),
                  "pit_histogram": hist.tolist()}
    return out


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    bc = cfg["backtest"]
    m = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "matches.parquet")
    m = m[m.home_npxg.notna() & m.away_npxg.notna() & m.home_manager.notna() & m.away_manager.notna()]
    cases = appointments(m[m.season.isin(bc["seasons"])], bc["min_before"], bc["min_after"])
    rng = np.random.default_rng(bc["seed"])
    rows, scores, ks, ss = [], [], [], []
    for case in cases.itertuples():
        r = case_samples(m[m.league == case.league], case, bc, rng)
        if r is None:
            continue
        sc = score_case(r, float(rng.random()))
        scores.append(sc)
        ks.append(r["k"])
        ss.append(r["s"])
        rows.append({"league": case.league, "season": case.season, "eng": case.league == "ENG"})
    meta = pd.DataFrame(rows)
    eng = [s for s, e in zip(scores, meta.eng, strict=True) if e]
    return finish(cfg, scores, eng, meta, ks, ss)


def finish(cfg: dict[str, Any], scores: list[Any], eng: list[Any], meta: pd.DataFrame, ks: list[float],
           ss: list[float]) -> dict[str, Any]:
    allm, engm = aggregate(scores), aggregate(eng)
    best = max(allm, key=lambda k: allm[k]["mean_log_score"])
    return {"cases": len(scores), "cases_eng": len(eng), "by_league": meta.league.value_counts().to_dict(),
            "seasons": cfg["backtest"]["seasons"], "settings": {k: v for k, v in cfg["backtest"].items()},
            "k_mean": float(np.mean(ks)), "shock_sd_mean": float(np.mean(ss)),
            "all_leagues": allm, "premier_league": engm,
            "summary": {"cases": len(scores), "best_by_log_score": best,
                        "best": {k: allm[best][k] for k in ("coverage50", "coverage80", "mean_log_score",
                                                             "mean_crps")},
                        "naive": {k: allm["naive"][k] for k in ("coverage50", "coverage80", "mean_log_score",
                                                                 "mean_crps")},
                        "not_tested": "manager terms (needs a manager-model refit at every appointment)"}}


def print_table(out: dict[str, Any]) -> None:
    print(f"{out['cases']} cases ({out['cases_eng']} Premier League); mean k {out['k_mean']:.3f}, "
          f"mean shock sd {out['shock_sd_mean']:.3f}")
    for scope in ("all_leagues", "premier_league"):
        print(f"\n{scope}: model  cover50  cover80  log score  CRPS   PIT histogram")
        for m, r in out[scope].items():
            print(f"  {m:<14} {r['coverage50']:.3f}   {r['coverage80']:.3f}   {r['mean_log_score']:.3f}"
                  f"   {r['mean_crps']:.3f}  {r['pit_histogram']}")


def main() -> None:
    import json
    cfg = load_config()
    out = run(cfg)
    path = ROOT / cfg["paths"]["docs_data"] / "backtest.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_table(out)
    print("written", path)
    from pochettino_fit.export import refresh_manifest
    refresh_manifest(cfg)


if __name__ == "__main__":
    main()
