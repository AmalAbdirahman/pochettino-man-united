"""The pre-registered forecast must never change.

The hash is of the file as committed in 66d190d (LF line endings). Windows checkouts with
core.autocrlf write CRLF, so line endings are normalised before hashing; any other change fails.
"""

import hashlib
import json

from pochettino_fit.config import ROOT

FROZEN = ROOT / "docs" / "forecasts" / "2026-11-20.json"
SHA256 = "46bcbc45d303c4c5957a78974b78cca4b349dde7f6429ecb5b33e20c9a22984e"


def test_frozen_forecast_is_unchanged() -> None:
    data = FROZEN.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == SHA256, "docs/forecasts/2026-11-20.json has changed"


def test_frozen_forecast_header() -> None:
    f = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert f["frozen_on"] == "2026-11-20" and f["first_league_match"] == "2026-11-22"
    assert f["git"]["commit"].startswith("bb91860") and f["git"]["dirty"] is False
