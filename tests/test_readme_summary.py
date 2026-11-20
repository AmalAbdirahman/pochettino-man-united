"""The README's static summary must match the JSON it quotes."""

import json

from pochettino_fit.config import ROOT


def test_readme_summary_matches_json() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    fc = json.loads((ROOT / "docs/forecasts/2026-11-20.json").read_text(encoding="utf-8"))
    fit = json.loads((ROOT / "docs/data/fit.json").read_text(encoding="utf-8"))
    c, p = fc["contribution"]["points"], fc["pochettino"]
    lo, hi = (int(v) for v in p["final_points"]["80"])
    ppda = fit["metrics"]["ppda"]
    expected = [
        f"{c['mean']:+.1f} points", f"(80% range −{-int(c['80'][0])} to +{int(c['80'][1])})",
        f"on {int(p['final_points']['median'])} points (80% range {lo} to {hi})",
        f"median position of {int(p['median_position'])}th",
        f"from {ppda['united_now']['mean']:.1f} now to {ppda['united_under_pochettino']['mean']:.1f}",
        f"about {round(fc['detectability']['matches_needed']['0.10'])} matches",
        f"rest of this season is {fc['united_now']['remaining']}.",
    ]
    missing = [e for e in expected if e not in readme]
    assert not missing, missing
