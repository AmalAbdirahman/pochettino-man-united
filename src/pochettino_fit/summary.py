"""Summarise real-data fits into docs/data/model_summary.json and the methods.md results block.

Every number written here comes from the posterior files in data/processed/fit_<variant>.nc.
"""

import json
import re
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from pochettino_fit.config import ROOT
from pochettino_fit.fit import VARIANTS, theta_now
from pochettino_fit.simulate import block_of

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def interval(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(np.mean(x)), "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]), "p_positive": float(np.mean(x > 0))}


def load(path: Path) -> Any:
    return xr.open_datatree(path, engine="h5netcdf")


def flat(post: Any, var: str) -> np.ndarray:
    v = post[var].values
    return v.reshape(-1, *v.shape[2:])


def diagnostics(tree: Any) -> dict[str, float]:
    keep = [v for v in tree["posterior"].data_vars
            if not v.endswith("_z") and v not in ("logv_missing", "strength", "theta")]
    summ = az.summary(tree, var_names=keep, kind="diagnostics")
    return {"divergences": int(tree["sample_stats"]["diverging"].values.sum()),
            "max_rhat": float(summ["r_hat"].max()), "min_ess_bulk": float(summ["ess_bulk"].min())}


def state_labels(cfg: dict[str, Any], manager: str) -> dict[int, str]:
    """Club (or national team) for each of a manager's blocks."""
    m = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "matches.parquet")
    m["block"] = m.date.map(block_of)
    lab = {}
    for side in ("home", "away"):
        mine = m[m[f"{side}_manager"] == manager]
        lab.update(dict(zip(mine.block, mine[f"{side}_name"], strict=True)))
    intl = pd.read_csv(ROOT / "data" / "international_matches.csv", comment="#", parse_dates=["date"])
    for d, team in zip(intl[intl.manager == manager].date, intl[intl.manager == manager].team, strict=True):
        lab.setdefault(block_of(d), team)
    return lab


def manager_block(post: Any, states: pd.DataFrame, manager: str, now: int) -> dict[str, Any]:
    mu = post["mu"].sel(manager=manager).values.ravel()
    return {"mu": interval(mu), "theta_now": interval(theta_now(post, states, manager, now)),
            "mu_rank_pct": float(rank_pct(post, manager))}


def rank_pct(post: Any, manager: str) -> float:
    """Share of managers whose posterior mean mu is below this manager's."""
    means = post["mu"].mean(("chain", "draw"))
    return float((means < means.sel(manager=manager)).mean())


def base_section(cfg: dict[str, Any], tree: Any) -> dict[str, Any]:
    fc, post = cfg["fit"], tree["posterior"]
    path = ROOT / cfg["paths"]["processed"] / "fit_base.nc"
    states = pd.read_parquet(path.with_suffix(".states.parquet"))
    cs = pd.read_parquet(path.with_suffix(".cs.parquet"))
    ls = pd.read_parquet(path.with_suffix(".ls.parquet")).ls.tolist()
    now = block_of(pd.Timestamp(fc["now"]))
    labels = state_labels(cfg, fc["manager"])
    th = flat(post, "theta")
    mine = states[states.manager == fc["manager"]]
    gamma = flat(post, "gamma")
    return {
        "pochettino": {**manager_block(post, states, fc["manager"], now),
                       "by_block": [{"block": int(b), "label": labels.get(int(b), "?"), **interval(th[:, i])}
                                    for i, b in zip(mine.index, mine.block, strict=True)]},
        "carrick": manager_block(post, states, fc["outgoing"], now),
        "united_strength": united_strength(cfg, post, cs),
        "gamma": interval(gamma), "half_life_years": interval(np.log(2) / gamma),
        "tau": interval(flat(post, "tau")), "sigma_mu": interval(flat(post, "sigma_mu")),
        "sigma_kappa_residual": interval(flat(post, "sigma_kappa")), "sigma": interval(flat(post, "sigma")),
        "tenure": {"year1_vs_3plus": interval(flat(post, "delta")[:, 0]),
                   "year2_vs_3plus": interval(flat(post, "delta")[:, 1])},
        "beta_luck": interval(flat(post, "beta")), "b_value_within": interval(flat(post, "b_value")),
        "b_value_between": interval(flat(post, "b_between")), "h0": interval(flat(post, "h0")),
        "home_advantage": [{"league_season": s, **interval(flat(post, "h")[:, i])} for i, s in enumerate(ls)],
        "lambda_int": interval(flat(post, "lambda_int")) if "lambda_int" in post else None,
        "n_states": int(len(states)), "n_managers": int(post["mu"].sizes["manager"]),
    }


def united_strength(cfg: dict[str, Any], post: Any, cs: pd.DataFrame) -> list[dict[str, Any]]:
    """United's club-season strength relative to the league-season average."""
    m = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "matches.parquet")
    names = dict(zip(m.home_team, m.home_name, strict=False))
    s = flat(post, "strength")
    out = []
    key = cs.league + " " + cs.season
    full = key.map(key.value_counts()) >= 18  # skip league-seasons with only part of the league in the fit
    for i in np.flatnonzero((cs.club.map(names).eq(cfg["fit"]["club_tm_name"]) & full).to_numpy()):
        peers = np.flatnonzero((key == key.iloc[i]).to_numpy())
        rel = s[:, i] - s[:, peers].mean(axis=1)
        out.append({"season": cs.season.iloc[i], **interval(rel)})
    return sorted(out, key=lambda r: r["season"])


def variant_effects(cfg: dict[str, Any]) -> dict[str, Any]:
    fc, now, out = cfg["fit"], block_of(pd.Timestamp(cfg["fit"]["now"])), {}
    for v in VARIANTS:
        path = ROOT / cfg["paths"]["processed"] / f"fit_{v}.nc"
        if not path.exists():
            out[v] = None
            continue
        tree = load(path)
        states = pd.read_parquet(path.with_suffix(".states.parquet"))
        out[v] = {**manager_block(tree["posterior"], states, fc["manager"], now),
                  "diagnostics": diagnostics(tree), "runtime_s": tree.attrs.get("runtime_s")}
    return out


def write_summary(cfg: dict[str, Any]) -> str:
    path = ROOT / cfg["paths"]["processed"] / "fit_base.nc"
    tree = load(path)
    summary = {
        "model": "methods.md version 2, implementation 2.1 (no league offset in y; Mundlak squad value)",
        "data_cutoff": str(cfg["features"]["data_cutoff"]), "evaluated_at": str(cfg["fit"]["now"]),
        "units": "non-penalty xG difference per 90 minutes (11 v 11); positive = better",
        "base": base_section(cfg, tree),
        "variants": variant_effects(cfg),
        "league_offsets": "not estimated: they cancel exactly in a within-league xG difference",
        "league_bridge_sensitivity": "not applicable for the same reason",
    }
    out = ROOT / "docs" / "data" / "model_summary.json"
    out.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    update_methods(summary)
    return results_markdown(summary)


def fmt(iv: dict[str, float] | None, d: int = 2) -> str:
    if iv is None:
        return "n/a"
    return f"{iv['mean']:+.{d}f} (90% {iv['q05']:+.{d}f} to {iv['q95']:+.{d}f})"


def results_markdown(s: dict[str, Any]) -> str:
    b, v = s["base"], s["variants"]
    p = b["pochettino"]
    lines = [
        "## 10. Results (generated from docs/data/model_summary.json; do not edit by hand)", "",
        f"Data to {s['data_cutoff']}; effects evaluated at {s['evaluated_at']}. Units: {s['units']}.", "",
        "### Implementation notes (version 2.1)",
        "- The league offset lambda cancels exactly in a within-league xG difference, so it is not "
        "estimated; cross-league comparisons rest on managers who moved between leagues and on the priors.",
        "- Squad value enters as a club mean (b_between) plus the within-club change (b_value), so b_value "
        "is not biased by persistent club differences; kappa alone is not identified, only club strength.",
        "- Synthetic recovery (docs/recovery.md): 90% intervals covered about 85-87% of true manager and "
        "club values at full size, so treat intervals as slightly too narrow.", "",
        "### Pochettino", "",
        f"- Long-run level mu: {fmt(p['mu'])}; P(above average) {p['mu']['p_positive']:.0%}; "
        f"posterior mean above {p['mu_rank_pct']:.0%} of the {b['n_managers']} managers in the data.",
        f"- Current effect theta (now): {fmt(p['theta_now'])}.",
        "", "| block | team | theta (mean) | 90% interval |", "|---|---|---|---|",
    ]
    for r in p["by_block"]:
        lines.append(f"| {r['block'] // 2}-{'H1' if r['block'] % 2 == 0 else 'H2'} | {r['label']} | "
                     f"{r['mean']:+.2f} | {r['q05']:+.2f} to {r['q95']:+.2f} |")
    lines += ["", "### Spell contributions and the USA spell", "",
              "| fit | mu | theta now | divergences | max R-hat |", "|---|---|---|---|---|"]
    for k, r in v.items():
        if r is None:
            lines.append(f"| {k} | not run | | | |")
            continue
        dg = r["diagnostics"]
        lines.append(f"| {k} | {fmt(r['mu'])} | {fmt(r['theta_now'])} | {dg['divergences']} | "
                     f"{dg['max_rhat']:.3f} |")
    us = b["united_strength"]
    lines += ["", "### Carrick and United", "",
              f"- Carrick mu: {fmt(b['carrick']['mu'])}; current theta: {fmt(b['carrick']['theta_now'])}.",
              "- United club strength relative to league average: "
              + "; ".join(f"{r['season']} {fmt(r)}" for r in us[-3:]) + ".", "",
              "### Shared parameters", "",
              f"- Drift: gamma {fmt(b['gamma'])}; half-life {fmt(b['half_life_years'], 1)} years; "
              f"tau {fmt(b['tau'])}; sigma_mu {fmt(b['sigma_mu'])}.",
              f"- Tenure (vs year 3+): year 1 {fmt(b['tenure']['year1_vs_3plus'])}; "
              f"year 2 {fmt(b['tenure']['year2_vs_3plus'])}.",
              f"- Recent luck beta: {fmt(b['beta_luck'], 3)}.",
              f"- Squad value: within-club {fmt(b['b_value_within'])}; between clubs "
              f"{fmt(b['b_value_between'])} per unit log value.",
              f"- Home advantage (global mean) {fmt(b['h0'])}; match noise sigma {fmt(b['sigma'])}.",
              f"- International scale lambda_int: {fmt(b['lambda_int'])} (only Pochettino's USA matches; "
              "mostly prior).", "",
              f"League offsets: {s['league_offsets']}.", ""]
    return "\n".join(lines)


def update_methods(summary: dict[str, Any]) -> None:
    path = ROOT / "docs" / "methods.md"
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{results_markdown(summary)}\n{END}"
    if START in text:
        text = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n---\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")
