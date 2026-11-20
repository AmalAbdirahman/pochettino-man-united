"""Export (prompt 8): the manifest the static page reads.

python -m pochettino_fit.export   -> docs/data/manifest.json

Lists the frozen forecasts (the earliest is the original; later files are dated revisions) and the data
files the page loads. Never writes inside docs/forecasts/.

Every step that writes page data (style, fit --summarise, spread, backtest, tracker) calls
refresh_manifest() at the end, so the footer's commit cannot go stale. A commit cannot contain its own hash,
so the footer names the commit the data was built from and says if there were uncommitted changes.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

from pochettino_fit.config import ROOT, load_config

PAGE_DATA = ["fit.json", "tracker.json", "backtest.json", "model_summary.json", "forecast_spread.json"]


def manifest(forecasts: Path, data: Path) -> dict[str, Any]:
    files = sorted(p.name for p in forecasts.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no frozen forecast in {forecasts}")
    missing = [f for f in PAGE_DATA if not (data / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing page data: {missing}")
    return {"original_forecast": f"forecasts/{files[0]}", "revisions": [f"forecasts/{f}" for f in files[1:]],
            "data": {f.removesuffix(".json"): f"data/{f}" for f in PAGE_DATA}}


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip()


def refresh_manifest(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    data = ROOT / cfg["paths"]["docs_data"]
    changed = git("status", "--porcelain").splitlines()
    dirty = any(not line.endswith("docs/data/manifest.json") for line in changed)
    out = {**manifest(ROOT / cfg["paths"]["forecasts"], data),
           "built_from_commit": git("rev-parse", "--short", "HEAD"), "uncommitted_changes": dirty}
    (data / "manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    print(json.dumps(refresh_manifest(), indent=2))


if __name__ == "__main__":
    main()
