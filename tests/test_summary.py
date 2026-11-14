"""Summary helpers on synthetic posteriors."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")
xr = pytest.importorskip("xarray")

from pochettino_fit import summary as S  # noqa: E402
from pochettino_fit.fit import theta_now  # noqa: E402


def fake_post(gamma: float, mu: float, theta_last: float, tau: float = 0.1) -> xr.Dataset:
    n = (2, 500)
    return xr.Dataset({
        "mu": (("chain", "draw", "manager"), np.full((*n, 1), mu)),
        "theta": (("chain", "draw", "state"), np.full((*n, 2), theta_last)),
        "gamma": (("chain", "draw"), np.full(n, gamma)),
        "tau": (("chain", "draw"), np.full(n, tau)),
    }, coords={"manager": ["P"]})


STATES = pd.DataFrame({"manager": ["P", "P"], "block": [4040, 4046]})


def test_theta_now_no_gap_returns_last_state() -> None:
    out = theta_now(fake_post(0.7, 0.1, 0.4), STATES, "P", now_block=4046)
    assert np.allclose(out, 0.4)


def test_theta_now_long_gap_reverts_to_mu() -> None:
    out = theta_now(fake_post(5.0, 0.1, 0.4), STATES, "P", now_block=4066)
    assert abs(out.mean() - 0.1) < 0.02
    assert abs(out.std() - 0.1) < 0.02  # stationary sd tau


def test_interval_quantiles() -> None:
    iv = S.interval(np.arange(1001) / 1000 - 0.5)
    assert iv["q05"] == pytest.approx(-0.45) and iv["q95"] == pytest.approx(0.45)
    assert iv["p_positive"] == pytest.approx(0.5, abs=0.001)


def test_update_methods_replaces_block(tmp_path, monkeypatch) -> None:
    doc = tmp_path / "docs"
    doc.mkdir()
    (doc / "methods.md").write_text("# Methods\n\ntext\n", encoding="utf-8")
    monkeypatch.setattr(S, "ROOT", tmp_path)
    monkeypatch.setattr(S, "results_markdown", lambda s: f"RESULT {s['n']}")
    S.update_methods({"n": 1})
    S.update_methods({"n": 2})
    text = (doc / "methods.md").read_text(encoding="utf-8")
    assert text.count(S.START) == 1 and "RESULT 2" in text and "RESULT 1" not in text
    assert text.startswith("# Methods")
