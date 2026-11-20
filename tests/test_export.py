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
