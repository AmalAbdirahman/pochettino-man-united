"""Download raw source files into data/raw/<source>/ and log them in data/raw/MANIFEST.json.

Run: python -m pochettino_fit.fetch [--only SOURCE ...] [--force]
Raw files are git-ignored and never republished.
"""

import argparse
import datetime as dt
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from pochettino_fit.config import ROOT, load_config

USER_AGENT = "pochettino-fit-research/0.1"


def football_data_season_code(season: str) -> str:
    """'2014-15' -> '1415'."""
    start = int(season[:4])
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def seasons_from(first: str, last: str) -> list[str]:
    """Inclusive list of seasons like '2014-15' .. '2026-27'."""
    a, b = int(first[:4]), int(last[:4])
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(a, b + 1)]


def planned_downloads(cfg: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(source, url, local filename) for every file to fetch."""
    src = cfg["sources"]
    out: list[tuple[str, str, str]] = []
    for name in ("understat_archive", "transfermarkt"):
        for f in src[name]["files"]:
            out.append((name, f"{src[name]['base']}/{f}", f))
    fd = src["football_data"]
    for season in seasons_from(cfg["scope"]["first_season"], cfg["appointment"]["season"]):
        code = football_data_season_code(season)
        for lg, div in fd["leagues"].items():
            out.append(("football_data", f"{fd['base']}/{code}/{div}.csv", f"{lg}_{code}.csv"))
        ce = src["clubelo"]
        day = f"{season[:4]}-{ce['snapshot_day']}"
        out.append(("clubelo", f"{ce['base']}/{day}", f"{day}.csv"))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def run(only: list[str] | None, force: bool) -> None:
    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw"]
    manifest_path = raw / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for source, url, fname in planned_downloads(cfg):
        if only and source not in only:
            continue
        dest = raw / source / fname
        if dest.exists() and not force:
            continue
        try:
            download(url, dest)
        except OSError as e:
            print(f"FAILED {url}: {e}")
            continue
        manifest[f"{source}/{fname}"] = {
            "url": url,
            "downloaded": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "sha256": sha256(dest),
            "bytes": dest.stat().st_size,
        }
        print(f"ok {source}/{fname}")
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        time.sleep(0.5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", help="sources to fetch")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    a = p.parse_args()
    run(a.only, a.force)


if __name__ == "__main__":
    main()
