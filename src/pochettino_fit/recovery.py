"""Synthetic recovery test for the manager model (prompt 2).

Run: python -m pochettino_fit.recovery --size small --seed 1 [--blocks season] [--advi]
Simulates leagues with known parameters, fits the model and reports interval coverage,
divergences, R-hat, ESS and run time to docs/data/recovery_<size>_seed<seed>.json.
"""

import argparse
import json
import time
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from pochettino_fit.config import ROOT, load_config
from pochettino_fit.manager_model import ModelData, build_model, prepare
from pochettino_fit.simulate import simulate

SCALARS = ["sigma", "tau", "gamma", "beta", "b_value", "sigma_mu", "sigma_kappa", "lambda_int", "sigma_int"]


def to_season_blocks(m: pd.DataFrame, intl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback: one state per season instead of per half-season (block -> even block)."""
    return m.assign(block=m.block // 2 * 2), intl.assign(block=intl.block // 2 * 2)


def draws_of(idata: Any, var: str) -> np.ndarray:
    """Posterior draws flattened to (n_draws, ...)."""
    x = idata.posterior[var].values
    return x.reshape(-1, *x.shape[2:])


def coverage(draws: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    out = {}
    for p in (0.5, 0.9):
        lo, hi = np.quantile(draws, [(1 - p) / 2, (1 + p) / 2], axis=0)
        out[f"cov{int(p * 100)}"] = float(np.mean((truth >= lo) & (truth <= hi)))
    out["n"] = int(np.size(truth))
    out["corr_mean_truth"] = float(np.corrcoef(draws.mean(0), truth)[0, 1]) if np.size(truth) > 2 else None
    return out


def _centre_by(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract the group mean along the last axis (works for truth vectors and draw matrices)."""
    out = np.array(x, dtype=float, copy=True)
    for g in np.unique(groups):
        sel = groups == g
        out[..., sel] -= out[..., sel].mean(axis=-1, keepdims=True)
    return out


def strength_truth(d: ModelData, m: pd.DataFrame, truth: dict, b_true: float) -> np.ndarray:
    """True club-season strength kappa + b * logV, matched to the model's club-season rows."""
    logv = pd.concat([
        m[["season", "home_team", "home_log_squad_value"]].set_axis(["s", "c", "v"], axis=1),
        m[["season", "away_team", "away_log_squad_value"]].set_axis(["s", "c", "v"], axis=1),
    ]).drop_duplicates(["s", "c"]).set_index(["s", "c"]).v
    cs = d.cs_table
    v = logv.reindex(pd.MultiIndex.from_arrays([cs.season, cs.club])).to_numpy()
    return truth["kappa"].reindex(cs.club).to_numpy() + b_true * v


def vector_results(d: ModelData, m: pd.DataFrame, truth: dict, idata: Any) -> dict[str, dict]:
    st = d.states
    theta_true = truth["theta"].reindex(pd.MultiIndex.from_arrays([st.manager, st.block])).to_numpy()
    h_true = truth["h"].reindex(pd.MultiIndex.from_tuples(d.ls_labels)).to_numpy()
    ls = (d.cs_table.league + " " + d.cs_table.season).to_numpy()
    s_true = _centre_by(strength_truth(d, m, truth, truth["scalars"]["b_value"]), ls)
    s_draws = _centre_by(draws_of(idata, "strength"), ls)
    return {
        "mu": coverage(draws_of(idata, "mu"), truth["mu"].reindex(d.managers).to_numpy()),
        "theta": coverage(draws_of(idata, "theta"), theta_true),
        "strength": coverage(s_draws, s_true),
        "h": coverage(draws_of(idata, "h"), h_true),
    }


def scalar_rows(idata: Any, truth: dict) -> dict[str, dict[str, float]]:
    s = dict(truth["scalars"])
    out = {}
    for name in SCALARS + ["delta_1", "delta_2"]:
        if name.startswith("delta"):
            dr = draws_of(idata, "delta")[:, int(name[-1]) - 1]
        elif name in idata.posterior:
            dr = draws_of(idata, name)
        else:
            continue
        lo50, hi50, lo90, hi90 = np.quantile(dr, [0.25, 0.75, 0.05, 0.95])
        t = s[name]
        out[name] = {"truth": t, "mean": float(dr.mean()), "q05": float(lo90), "q95": float(hi90),
                     "in50": bool(lo50 <= t <= hi50), "in90": bool(lo90 <= t <= hi90)}
    if "sigma_kappa" in out:  # model's kappa is net of the club-mean value term
        out["sigma_kappa"]["comparable"] = False
    return out


def realised_noise_sd(m: pd.DataFrame) -> float:
    return float(((m.y - m.true_mean) * np.sqrt(m.w)).std())


def diagnostics(idata: Any) -> dict[str, float]:
    keep = [v for v in idata.posterior.data_vars if not v.endswith("_z") and v != "logv_missing"]
    summ = az.summary(idata, var_names=keep, kind="diagnostics")
    div = int(idata.sample_stats["diverging"].values.sum())
    return {"divergences": div, "max_rhat": float(summ["r_hat"].max()),
            "min_ess_bulk": float(summ["ess_bulk"].min()), "min_ess_tail": float(summ["ess_tail"].min())}


def fit(d: ModelData, sampling: dict[str, Any], seed: int, advi: bool) -> tuple[Any, str]:
    model = build_model(d)
    with model:
        if advi:
            approx = pm.fit(30000, method="advi", random_seed=seed,
                            compile_kwargs={"mode": sampling["mode"]}, progressbar=False)
            return approx.sample(sampling["draws"] * sampling["chains"]), "ADVI"
        idata = pm.sample(draws=sampling["draws"], tune=sampling["tune"], chains=sampling["chains"],
                          cores=sampling["cores"], target_accept=sampling["target_accept"],
                          random_seed=seed, compile_kwargs={"mode": sampling["mode"]}, progressbar=False)
    return idata, "NUTS"


def run(size: str, seed: int, blocks: str, advi: bool, target_accept: float | None = None) -> dict[str, Any]:
    cfg = load_config()["simulation"]
    if target_accept:
        cfg["sampling"]["target_accept"] = target_accept
    m, intl, truth = simulate(cfg["sizes"][size], cfg["truth"], seed)
    if blocks == "season":
        m, intl = to_season_blocks(m, intl)
        st = truth["theta"].reset_index()
        st = st[st.block % 2 == 0]  # season state = truth in the first half (approximation)
        truth["theta"] = st.set_index(["manager", "block"]).theta
    d = prepare(m, intl)
    t0 = time.time()
    idata, method = fit(d, cfg["sampling"], seed, advi)
    runtime = time.time() - t0
    report = {
        "size": size, "seed": seed, "blocks": blocks, "method": method, "backend": cfg["sampling"]["mode"],
        "n_matches": len(d.y), "n_managers": len(d.managers), "n_states": len(d.states),
        "n_clubs": len(d.clubs), "n_international": len(d.intl_r), "runtime_s": round(runtime, 1),
        "sampling": cfg["sampling"],
        "vectors": vector_results(d, m, truth, idata),
        "scalars": scalar_rows(idata, truth),
        "realised_noise_sd": realised_noise_sd(m),
        "note": ("lambda[l,s] is simulated but cancels in y, so it is not estimated. kappa alone is "
                 "not identified separately from b_between; club-season strength (centred within "
                 "league-season) is scored instead."),
    }
    if method == "NUTS":
        report["diagnostics"] = diagnostics(idata)
    return report


def summarise() -> str:
    """Markdown table of every recovery run in docs/data (generated, never hand-edited)."""
    runs = [json.loads(f.read_text()) for f in sorted((ROOT / "docs" / "data").glob("recovery_*.json"))]
    runs = [r for r in runs if r["size"] != "tiny"]
    lines = ["# Synthetic recovery results", "",
             "Generated by `python -m pochettino_fit.recovery --summary` from docs/data/recovery_*.json.",
             "", "## Runs", "",
             "| size | seed | blocks | method | target_accept | matches | managers | states | divergences "
             "| max R-hat | min ESS bulk | run time (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        dg = r.get("diagnostics", {})
        lines.append(f"| {r['size']} | {r['seed']} | {r['blocks']} | {r['method']} | "
                     f"{r['sampling']['target_accept']} | {r['n_matches']} | "
                     f"{r['n_managers']} | {r['n_states']} | {dg.get('divergences', '-')} | "
                     f"{dg.get('max_rhat', float('nan')):.3f} | {dg.get('min_ess_bulk', float('nan')):.0f} | "
                     f"{r['runtime_s']:.0f} |")
    lines += ["", "## Interval coverage of vector parameters (share of true values inside)", "",
              "| size | seed | parameter | n | 50% | 90% | corr(mean, truth) |",
              "|---|---|---|---|---|---|---|"]
    for r in runs:
        for k, v in r["vectors"].items():
            corr = "-" if v["corr_mean_truth"] is None else f"{v['corr_mean_truth']:.2f}"
            lines.append(f"| {r['size']} | {r['seed']} | {k} | {v['n']} | {v['cov50']:.2f} | "
                         f"{v['cov90']:.2f} | {corr} |")
    lines += ["", "## Scalars: truth inside the 90% interval, pooled over runs", "",
              "| parameter | runs | inside 50% | inside 90% | note |", "|---|---|---|---|---|"]
    names = sorted({k for r in runs for k in r["scalars"]})
    for k in names:
        rows = [r["scalars"][k] for r in runs if k in r["scalars"]]
        note = "not comparable (net of club-mean value term)" if any(
            row.get("comparable") is False for row in rows) else ""
        lines.append(f"| {k} | {len(rows)} | {sum(x['in50'] for x in rows)} | "
                     f"{sum(x['in90'] for x in rows)} | {note} |")
    lines += ["", "Realised noise sd per run (sigma is judged against this, not only the nominal 1.2): "
              + ", ".join(f"{r['size']}/{r['seed']}: {r['realised_noise_sd']:.3f}" for r in runs
                          if "realised_noise_sd" in r), "",
              "lambda[l,s] is simulated but cancels exactly in y (both teams share it), so it is not "
              "estimated. kappa alone is not identified separately from the club-mean value term; "
              "club-season strength (centred within league-season) is scored instead.", ""]
    text = "\n".join(lines)
    (ROOT / "docs" / "recovery.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", default="small", choices=["tiny", "small", "full"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--blocks", default="half", choices=["half", "season"])
    p.add_argument("--advi", action="store_true")
    p.add_argument("--summary", action="store_true", help="only rebuild docs/recovery.md")
    p.add_argument("--target-accept", type=float, default=None)
    a = p.parse_args()
    if a.summary:
        print(summarise())
        return
    report = run(a.size, a.seed, a.blocks, a.advi, a.target_accept)
    tag = f"_ta{a.target_accept}" if a.target_accept else ""
    out = ROOT / "docs" / "data" / f"recovery_{a.size}_seed{a.seed}_{a.blocks}{tag}.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
