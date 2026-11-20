"""Style fit (prompt 5; methods section 4): predicted style change under Pochettino and the weakness check.

python -m pochettino_fit.style --fit   one PyMC fit per metric -> data/processed/style/<metric>.npz (local)
python -m pochettino_fit.style         summary -> docs/data/fit.json

Delta = theta_Pochettino - theta_outgoing per metric (team-match sd units, draw by draw),
D^2 = Delta' R^-1 Delta
with R the correlation of the metrics over team-matches, and D^2's percentile among all real managerial
changes in the data (posterior means, so shrinkage is treated alike for every change).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pochettino_fit.config import ROOT, load_config
from pochettino_fit.export import refresh_manifest
from pochettino_fit.style_model import BY_NAME, METRICS, StyleMetric, prepare, raw_values, sample


def out_dir(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["paths"]["processed"] / "style"


SOURCES = {"understat": "team_matches.parquet", "fd": "fd_team_matches.parquet"}


def load_rows(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {k: pd.read_parquet(ROOT / cfg["paths"]["processed"] / f) for k, f in SOURCES.items()}


def fit_metric(t: pd.DataFrame, m: StyleMetric, cfg: dict[str, Any], seed: int) -> None:
    sc, fc, club = cfg["style"], cfg["fit"], cfg["profile"]["team_tm_id"]
    d = prepare(t, m)
    p = sample(d, sc["sampling"], seed)
    keep = [d.managers.index(fc["manager"]), d.managers.index(fc["outgoing"])]
    ls_labels = list(pd.unique(d.rows.league + " " + d.rows.season))
    counts = d.rows.manager.value_counts()
    np.savez_compressed(
        out_dir(cfg) / f"{m.name}.npz", managers=np.array(d.managers),
        n_matches=counts[d.managers].to_numpy(), theta_mean=p["theta"].mean(0),
        theta_keep=p["theta"][:, keep], kappa_utd=p["kappa"][:, d.clubs.index(club)],
        lam_ref=p["lam"][:, ls_labels.index(sc["reference_league_season"])],
        sigma_theta=p["sigma_theta"], sigma_kappa=p["sigma_kappa"], sigma=p["sigma"], max_rhat=p["max_rhat"],
        centre=d.centre, scale=d.scale, n_rows=len(d.z))
    print(f"{m.name}: {len(d.z)} rows, max R-hat {float(p['max_rhat']):.3f}", flush=True)


def fit_all(cfg: dict[str, Any]) -> None:
    out_dir(cfg).mkdir(parents=True, exist_ok=True)
    t = load_rows(cfg)
    for i, m in enumerate(METRICS):
        fit_metric(t[m.source], m, cfg, cfg["style"]["seed"] + i)


# ---------- summary ----------

def interval(x: np.ndarray, lv: float) -> list[float]:
    return [float(q) for q in np.quantile(x, [(1 - lv) / 2, (1 + lv) / 2])]


def natural(f: Any, m: StyleMetric, theta: np.ndarray) -> np.ndarray:
    """United's level in natural units (reference league-season) with a given manager effect."""
    v = f["centre"] + f["scale"] * (f["lam_ref"] + f["kappa_utd"] + theta)
    return np.exp(v) if m.log else v


def metric_summary(f: Any, m: StyleMetric, lv: float, min_n: int) -> dict[str, Any]:
    tp, tc = f["theta_keep"][:, 0], f["theta_keep"][:, 1]
    now, new = natural(f, m, tc), natural(f, m, tp)
    sth, ska = f["sigma_theta"], f["sigma_kappa"]
    est = f["theta_mean"][f["n_matches"] >= min_n]
    out = {"label": m.label, "units_note": "natural units at United's club level, reference league-season",
           "united_now": {"mean": float(now.mean()), "interval": interval(now, lv)},
           "united_under_pochettino": {"mean": float(new.mean()), "interval": interval(new, lv)},
           "change": {"mean": float((new - now).mean()), "interval": interval(new - now, lv)},
           "delta_sd_units": {"mean": float((tp - tc).mean()), "interval": interval(tp - tc, lv),
                              "p_positive": float(np.mean(tp > tc))},
           "pochettino_effect_sd_units": {"mean": float(tp.mean()), "interval": interval(tp, lv)},
           "outgoing_effect_sd_units": {"mean": float(tc.mean()), "interval": interval(tc, lv)},
           "manager_share_of_style_variation": float(np.mean(sth**2 / (sth**2 + ska**2))),
           "pochettino_abs_effect_percentile": float(np.mean(np.abs(est) < abs(tp.mean()))),
           "max_rhat": float(f["max_rhat"]), "rows": int(f["n_rows"])}
    if m.better:
        good = (new < now) if m.better == "lower" else (new > now)
        out["p_improves"] = float(np.mean(good))
    return out


def correlation(t: dict[str, pd.DataFrame], names: list[str]) -> np.ndarray:
    """Pairwise correlation over team-matches, joining the two sources on match and team."""
    cols = {n: raw_values(t[BY_NAME[n].source], BY_NAME[n])[0].set_axis(
        pd.MultiIndex.from_frame(t[BY_NAME[n].source][["match_id", "team"]])) for n in names}
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).corr().to_numpy()


def real_changes(t: pd.DataFrame) -> pd.DataFrame:
    s = t[t.manager.notna()].sort_values(["team", "date"])
    prev = s.groupby("team").manager.shift()
    ch = s[prev.notna() & (prev != s.manager)]
    return pd.DataFrame({"team": ch.team, "date": ch.date, "outgoing": prev[ch.index],
                         "incoming": ch.manager})


def mean_delta(fs: dict[str, Any], names: list[str], out: str, inc: str) -> np.ndarray | None:
    v = []
    for n in names:
        idx = {mg: i for i, mg in enumerate(fs[n]["managers"])}
        if out not in idx or inc not in idx:
            return None
        v.append(fs[n]["theta_mean"][idx[inc]] - fs[n]["theta_mean"][idx[out]])
    return np.array(v)


def distance(fs: dict[str, Any], t: dict[str, pd.DataFrame], names: list[str], cfg: dict[str, Any],
             lv: float) -> dict[str, Any]:
    rinv = np.linalg.inv(correlation(t, names))
    delta = np.column_stack([fs[n]["theta_keep"][:, 0] - fs[n]["theta_keep"][:, 1] for n in names])
    d2 = np.einsum("ij,jk,ik->i", delta, rinv, delta)
    ch = real_changes(t["fd"])
    ds = [mean_delta(fs, names, r.outgoing, r.incoming) for r in ch.itertuples()]
    ref = np.array([d @ rinv @ d for d in ds if d is not None])
    dm = delta.mean(0)
    d2_mean_effects = float(dm @ rinv @ dm)
    return {"metrics": names, "d2_draws": {"mean": float(d2.mean()), "median": float(np.median(d2)),
                                           "interval": interval(d2, lv)},
            "d2_posterior_means": d2_mean_effects,
            "percentile_among_real_changes": float(np.mean(ref < d2_mean_effects)),
            "real_changes": int(len(ref)), "real_changes_median_d2": float(np.median(ref)),
            "note": "D^2 over draws includes posterior noise and is larger than D^2 of the posterior means; "
                    "the percentile compares posterior means with posterior means."}


def plain_english(res: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    fc, mets, dist = cfg["fit"], res["metrics"], res["distance"]
    out = [f"Predicted style change from {fc['outgoing']} to {fc['manager']}: "
           "the distance D² between their estimated styles is "
           f"{dist['d2_posterior_means']:.2f}, larger than "
           f"{100 * dist['percentile_among_real_changes']:.0f}% "
           f"of {dist['real_changes']} real managerial changes in the data."]
    big = sorted(mets.items(), key=lambda kv: -abs(kv[1]["delta_sd_units"]["mean"]))[:3]
    for _, s in big:
        c, lo, hi = s["change"], *s["change"]["interval"]
        out.append(f"{s['label']}: {s['united_now']['mean']:.2f} now, "
                   f"{s['united_under_pochettino']['mean']:.2f} predicted "
                   f"(change {c['mean']:+.2f}, 80% {lo:+.2f} to {hi:+.2f}).")
    for n in cfg["style"]["weakness"]:
        s = mets[n]
        out.append(f"Weakness check, {s['label']}: P(improves under him) = {s['p_improves']:.2f}.")
    n = {k: v[fc["outgoing"]] for k, v in res["matches_in_style_data"].items()}
    out.append(f"{fc['outgoing']} has {n['fd']} matches for shots and corners (football-data) but only "
               f"{n['understat']} for the Understat metrics, so his effect there is shrunk towards an "
               "average manager, and the predicted change mostly reflects Pochettino's own habits.")
    return out


def summarise(cfg: dict[str, Any]) -> dict[str, Any]:
    sc, fc = cfg["style"], cfg["fit"]
    lv = sc["interval"]
    fs = {m.name: np.load(out_dir(cfg) / f"{m.name}.npz") for m in METRICS}
    t = load_rows(cfg)
    counts = {k: v.manager.value_counts() for k, v in t.items()}
    res: dict[str, Any] = {
        "model": "methods 4a; one partially pooled fit per metric on Understat team-matches", "interval": lv,
        "reference_league_season": sc["reference_league_season"],
        "matches_in_style_data": {src: {mg: int(c.get(mg, 0)) for mg in (fc["manager"], fc["outgoing"])}
                                  for src, c in counts.items()},
        "sources": {"understat": "Understat archive, big five, 2014-15 to 2026-04-18",
                    "fd": f"football-data.co.uk, big five, 2014-15 to {cfg['features']['data_cutoff']}"},
        "metrics": {m.name: metric_summary(fs[m.name], m, lv, sc["min_matches_percentile"]) for m in METRICS}}
    res["distance"] = distance(fs, t, sc["distance_metrics"], cfg, lv)
    res["weakness_check"] = {n: res["metrics"][n] for n in sc["weakness"]}
    res["limits"] = sc["limits"]
    res["plain_english"] = plain_english(res, cfg)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", action="store_true")
    a = p.parse_args()
    cfg = load_config()
    if a.fit:
        fit_all(cfg)
    res = summarise(cfg)
    path = ROOT / cfg["paths"]["docs_data"] / "fit.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(res["plain_english"]))
    print("written", path)
    refresh_manifest(cfg)


if __name__ == "__main__":
    main()
