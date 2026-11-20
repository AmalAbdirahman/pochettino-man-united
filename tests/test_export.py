"""Export manifest: the original forecast is the earliest file; missing inputs fail loudly."""

from pathlib import Path

import pytest

from pochettino_fit.export import PAGE_DATA, manifest


def test_manifest_lists_original_and_revisions(tmp_path: Path) -> None:
    f, d = tmp_path / "forecasts", tmp_path / "data"
    f.mkdir()
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        manifest(f, d)
    for name in ("2026-12-30.json", "2026-11-20.json"):
        (f / name).write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        manifest(f, d)
    for name in PAGE_DATA:
        (d / name).write_text("{}", encoding="utf-8")
    m = manifest(f, d)
    assert m["original_forecast"] == "forecasts/2026-11-20.json"
    assert m["revisions"] == ["forecasts/2026-12-30.json"] and m["data"]["fit"] == "data/fit.json"


def test_page_loads_only_listed_data() -> None:
    """Every manifest.data.<name> the page fetches must be in PAGE_DATA, and the reverse."""
    import re

    from pochettino_fit.config import ROOT

    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    used = set(re.findall(r"manifest\.data\.(\w+)", html))
    assert used == {f.removesuffix(".json") for f in PAGE_DATA}


def test_build_steps_refresh_the_manifest() -> None:
    from pochettino_fit.config import ROOT

    for step in ("style", "summary", "spread", "backtest", "tracker"):
        src = (ROOT / "src" / "pochettino_fit" / f"{step}.py").read_text(encoding="utf-8")
        assert "refresh_manifest(" in src, step
