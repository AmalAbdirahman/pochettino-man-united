"""Tracker (prompt 7; methods section 7): compare each new United league match with the frozen forecast.

python -m pochettino_fit.tracker

Reads the ORIGINAL frozen forecast (the earliest file in docs/forecasts/, never written to) and the local
FotMob notes for United. Every gameweek after the freeze is matched to its frozen fixture by opponent and
venue. Writes data/processed/tracker_local.json (everything) and docs/data/tracker.json (publishable
fields only: FotMob values always stay local). tracker.publication picks what else is public:
derived_only (results, frozen bands and our model outputs) or results_only (results and frozen bands;
nothing computed from FotMob xG). See docs/tracker_plan.md.

His effect is updated with a normal update between full refits (methods 7): each match gives one noisy
observation of his effect, actual npxG difference minus the frozen baseline-run expectation, with the
manager model's match noise sd sigma at full weight (no red-card truncation in the FotMob notes).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from pochettino_fit import sources as S
from pochettino_fit.config import ROOT, load_config
from pochettino_fit.features import name_key

FOTMOB_FIELDS = ("actual_npxg_diff", "observed_effect", "chance_quality_conceded")
# our model outputs that are computed from FotMob xG (published under derived_only, dropped under results_only)
MODEL_FIELDS = ("effect", "inside_interval", "inside_interval_share")
MODES = ("derived_only", "results_only")
NOTES = {
    "derived_only": "FotMob xG and chance-quality values are kept local (not for republication). "
                    "This file holds results, the frozen bands and our own model outputs only.",
    "results_only": "Results and the frozen bands only. Nothing computed from FotMob xG is published.",
}
STRIP = ("afc", "fc", "&", "and", "hove", "albion")


def club_key(name: str) -> str:
    """'Brighton & Hove Albion' and 'Brighton' both become 'brighton'; 'AFC Bournemouth' -> 'bournemouth'."""
    return " ".join(w for w in name_key(name).split() if w not in STRIP)


def guarded_write(path: Path, obj: Any, forecasts_dir: Path) -> None:
    """Write JSON, refusing any path inside the frozen forecasts folder."""
    if path.resolve().is_relative_to(forecasts_dir.resolve()):
        raise PermissionError(f"refusing to write {path}: frozen forecasts are never touched")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def original_forecast(forecasts_dir: Path) -> tuple[Path, dict[str, Any]]:
    files = sorted(forecasts_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no frozen forecast in {forecasts_dir}: run forecast --freeze first")
    return files[0], json.loads(files[0].read_text(encoding="utf-8"))


def match_fixture(fixtures: list[dict[str, Any]], opponent: str, venue: str) -> int:
    hits = [i for i, f in enumerate(fixtures) if f["venue"] == venue
            and club_key(f["opponent"]) == club_key(opponent)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} frozen fixtures match {opponent} ({venue})")
    return hits[0]


def predictive_interval(fx: dict[str, Any], sigma: float) -> list[float]:
    """Interval for the match's ACTUAL npxG difference: the frozen interval covers the expected value only
    (parameter uncertainty), so add the match noise sigma; normal approximation at the same level."""
    z = float(norm.ppf((1 + fx["interval"]) / 2))
    lo, hi = fx["npxg_diff_interval"]
    sd = float(np.hypot((hi - lo) / (2 * z), sigma))
    return [fx["npxg_diff_mean"] - z * sd, fx["npxg_diff_mean"] + z * sd]


def per_match(row: pd.Series, fx: dict[str, Any], sigma: float) -> dict[str, Any]:
    actual = float(row.npxg_for - row.npxg_against)
    lo, hi = predictive_interval(fx, sigma)
    return {"gameweek": int(row.gameweek), "opponent": fx["opponent"], "venue": fx["venue"],
            "goals_for": int(row.goals_for), "goals_against": int(row.goals_against),
            "points": int(row.points), "expected_points": fx["expected_points"],
            "p_win_draw_loss": fx["p_win_draw_loss"],
            "forecast_npxg_diff_mean": fx["npxg_diff_mean"],
            "frozen_expected_npxg_diff_interval": fx["npxg_diff_interval"],
            "predictive_npxg_diff_interval": [lo, hi], "interval": fx["interval"],
            "inside_interval": bool(lo <= actual <= hi),
            "actual_npxg_diff": actual, "observed_effect": actual - fx["baseline_npxg_diff_mean"]}


def points_band(fixtures: list[dict[str, Any]], idx: list[int], actual: int,
                levels: list[float]) -> dict[str, Any]:
    """Frozen-forecast distribution of United's points over the fixtures played so far."""
    digits = [np.frombuffer(fixtures[i]["points_draws"].encode(), np.uint8) - 48 for i in idx]
    draws = np.sum(digits, axis=0)
    out: dict[str, Any] = {"matches": len(idx), "actual": actual, "forecast_median": float(np.median(draws)),
                           "forecast_mean": float(draws.mean()),
                           "pit": float(np.mean(draws < actual) + 0.5 * np.mean(draws == actual))}
    for lv in levels:
        out[f"{int(lv * 100)}"] = [float(q) for q in np.quantile(draws, [(1 - lv) / 2, (1 + lv) / 2])]
    out["inside_80"] = bool(out["80"][0] <= actual <= out["80"][1]) if "80" in out else None
    return out


def effect_update(prior_mean: float, prior_sd: float, sigma: float, obs: list[float]) -> dict[str, Any]:
    """Normal-normal update: precision adds 1/sigma^2 per match at full weight."""
    prec = 1 / prior_sd**2 + len(obs) / sigma**2
    mean = (prior_mean / prior_sd**2 + sum(obs) / sigma**2) / prec
    sd = prec**-0.5
    k = float(norm.ppf(0.9))
    return {"matches": len(obs), "mean": mean, "sd": sd, "80": [mean - k * sd, mean + k * sd],
            "p_positive": float(norm.cdf(mean / sd)), "prior_mean": prior_mean, "prior_sd": prior_sd,
            "sigma": sigma, "method": "normal update between refits (methods section 7)"}


def chance_quality(fm: pd.DataFrame) -> dict[str, float]:
    return {"matches": len(fm),
            "npxg_per_shot_against": float(fm.npxg_against.sum() / fm.shots_against.sum()),
            "big_chances_against_per_match": float(fm.big_chances_against.mean()),
            "npxg_against_per_match": float(fm.npxg_against.mean())}


def track(frozen: dict[str, Any], fm: pd.DataFrame, levels: list[float]) -> dict[str, Any]:
    fixtures, first = frozen["united_fixtures"], frozen["united_now"]["played"] + 1
    fm = fm.sort_values("gameweek")
    before, after = fm[fm.gameweek < first], fm[fm.gameweek >= first]
    idx = [match_fixture(fixtures, r.opponent, r.venue) for r in after.itertuples()]
    if len(set(idx)) != len(idx):
        raise ValueError("the same frozen fixture matched twice")
    d = frozen["detectability"]
    rows = [per_match(r, fixtures[i], d["sigma"]) for (_, r), i in zip(after.iterrows(), idx, strict=True)]
    out: dict[str, Any] = {"matches": rows, "effect": effect_update(
        d["effect_mean"], d["effect_sd_now"], d["sigma"], [r["observed_effect"] for r in rows])}
    if rows:
        out["cumulative_points"] = points_band(fixtures, idx, sum(r["points"] for r in rows), levels)
        pts = np.cumsum([r["points"] for r in rows])
        out["cumulative_by_match"] = [points_band(fixtures, idx[:k], int(pts[k - 1]), levels)
                                      for k in range(1, len(rows) + 1)]
        out["inside_interval_share"] = float(np.mean([r["inside_interval"] for r in rows]))
        out["chance_quality_conceded"] = {"before": chance_quality(before), "since": chance_quality(after)}
    return out


def publishable(out: dict[str, Any], mode: str) -> dict[str, Any]:
    """The public tracker file. FotMob values are dropped in every mode; there is no mode that keeps them."""
    if mode not in MODES:
        raise ValueError(f"tracker.publication must be one of {MODES}, not {mode!r}")
    drop = FOTMOB_FIELDS + (MODEL_FIELDS if mode == "results_only" else ())
    pub = {k: v for k, v in out.items() if k not in drop}
    pub["matches"] = [{k: v for k, v in r.items() if k not in drop} for r in out["matches"]]
    pub["publication"], pub["note"] = mode, NOTES[mode]
    return pub


def main() -> None:
    cfg = load_config()
    fdir = ROOT / cfg["paths"]["forecasts"]
    path, frozen = original_forecast(fdir)
    fm = S.load_fotmob_united(ROOT / cfg["paths"]["raw"])
    out = {"frozen_forecast": path.name, "frozen_on": frozen["frozen_on"],
           **track(frozen, fm, cfg["forecast"]["interval_levels"])}
    guarded_write(ROOT / cfg["paths"]["processed"] / "tracker_local.json", out, fdir)
    pub = publishable(out, cfg["tracker"]["publication"])
    guarded_write(ROOT / cfg["paths"]["docs_data"] / "tracker.json", pub, fdir)
    e = out["effect"]
    print(f"{len(out['matches'])} matches since the freeze; effect {e['mean']:+.3f} "
          f"(80% {e['80'][0]:+.3f} to {e['80'][1]:+.3f}), P(>0) {e['p_positive']:.2f}")
    if "cumulative_points" in out:
        c = out["cumulative_points"]
        print(f"points {c['actual']} vs frozen 80% band {c['80']} (PIT {c['pit']:.2f})")


if __name__ == "__main__":
    main()
