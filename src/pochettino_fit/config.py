"""Load config.yaml from the repository root."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Return the parsed config as a dict."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
