"""Export (prompt 8): the manifest the static page reads.

python -m pochettino_fit.export   -> docs/data/manifest.json

Lists the frozen forecasts (the earliest is the original; later files are dated revisions) and the data
files the page loads. Never writes inside docs/forecasts/.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

from pochettino_fit.config import ROOT, load_config

PAGE_DATA = ["fit.json", "tracker.json", "backtest.json", "model_summary.json"]


def manifest(forecasts: Path, data: Path) -> dict[str, Any]:
    files = sorted(p.name for p in forecasts.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no frozen forecast in {forecasts}")
    missing = [f for f in PAGE_DATA if not (data / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing page data: {missing}")
    return {"original_forecast": f"forecasts/{files[0]}", "revisions": [f"forecasts/{f}" for f in files[1:]],
            "data": {f.removesuffix(".json"): f"data/{f}" for f in PAGE_DATA}}


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip()


def main() -> None:
    cfg = load_config()
    data = ROOT / cfg["paths"]["docs_data"]
    out = {**manifest(ROOT / cfg["paths"]["forecasts"], data), "built_from_commit": git_commit()}
    (data / "manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
