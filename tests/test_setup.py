"""Setup checks: package imports, config loads, raw data is never tracked."""

import datetime as dt
import subprocess

import pochettino_fit
from pochettino_fit.config import ROOT, load_config


def test_package_imports() -> None:
    assert pochettino_fit.__version__


def test_config_has_expected_keys() -> None:
    cfg = load_config()
    for key in ["model_version", "appointment", "scope", "outcome", "luck", "paths"]:
        assert key in cfg
    assert cfg["model_version"] == 2
    assert cfg["outcome"]["rows_per_match"] == 1
    assert cfg["appointment"]["starts"] == dt.date(2026, 11, 18)


def test_raw_data_not_tracked() -> None:
    out = subprocess.run(
        ["git", "ls-files", "data/raw"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert out.stdout.strip() == ""


def test_raw_data_is_gitignored() -> None:
    out = subprocess.run(
        ["git", "check-ignore", "-q", "data/raw/anything.csv"], cwd=ROOT, check=False
    )
    assert out.returncode == 0
