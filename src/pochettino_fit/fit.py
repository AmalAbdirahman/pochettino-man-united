"""Fit the manager model on real data (prompt 3).

Run: python -m pochettino_fit.fit [--variant base|no_usa|drop_spurs|drop_psg|drop_chelsea|all]
Each variant writes data/processed/fit_<variant>.nc (local only). Then
python -m pochettino_fit.fit --summarise writes docs/data/model_summary.json.
"""

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm

from pochettino_fit.config import ROOT, load_config
from pochettino_fit.manager_model import ModelData, build_model, prepare
from pochettino_fit.simulate import BLOCK_YEARS, block_of

VARIANTS = ["base", "no_usa", "drop_spurs", "drop_psg", "drop_chelsea"]


def load_international(manager_names: set[str]) -> pd.DataFrame:
    intl = pd.read_csv(ROOT / "data" / "international_matches.csv", comment="#", parse_dates=["date"])
    intl = intl[intl.manager.isin(manager_names)]
    return pd.DataFrame({"manager": intl.manager, "block": intl.date.map(block_of),
                         "w": intl.weight, "r": intl.residual, "date": intl.date})


def variant_data(variant: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matches and international rows for one variant (leave one spell or the USA out)."""
    fc = cfg["fit"]
    m = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "matches.parquet")
    m = m[m.y.notna()].copy()
    m["block"] = m.date.map(block_of)
    managers = set(m.home_manager.dropna()) | set(m.away_manager.dropna())
    intl = load_international(managers)
    if variant == "no_usa":
        intl = intl.iloc[0:0]
    if variant.startswith("drop_"):
        club = fc["spells"][variant[5:]]
        his = ((m.home_manager == fc["manager"]) & (m.home_name == club)) | (
            (m.away_manager == fc["manager"]) & (m.away_name == club))
        m = m[~his]
    return m, intl


def sample(d: ModelData, s: dict[str, Any], seed: int) -> Any:
    with build_model(d):
        return pm.sample(draws=s["draws"], tune=s["tune"], chains=s["chains"], cores=s["cores"],
                         target_accept=s["target_accept"], random_seed=seed,
                         compile_kwargs={"mode": s["mode"]}, progressbar=False)


def run_variant(variant: str, cfg: dict[str, Any], seed: int = 20261118) -> None:
    m, intl = variant_data(variant, cfg)
    d = prepare(m, intl)
    t0 = time.time()
    idata = sample(d, cfg["fit"]["sampling"], seed)
    idata.attrs["runtime_s"] = round(time.time() - t0, 1)
    idata.attrs["variant"] = variant
    out = ROOT / cfg["paths"]["processed"] / f"fit_{variant}.nc"
    idata.to_netcdf(out, engine="h5netcdf")
    d.states.to_parquet(out.with_suffix(".states.parquet"))
    pd.DataFrame({"club": d.clubs}).to_parquet(out.with_suffix(".clubs.parquet"))
    d.cs_table.to_parquet(out.with_suffix(".cs.parquet"))
    pd.DataFrame({"ls": [f"{a} {b}" for a, b in d.ls_labels]}).to_parquet(out.with_suffix(".ls.parquet"))
    print(f"{variant}: {len(d.y)} matches, {len(d.managers)} managers, {len(d.states)} states, "
          f"{len(d.intl_r)} international rows, {idata.attrs['runtime_s']} s; dropped {d.dropped}")


def theta_now(post: Any, states: pd.DataFrame, manager: str, now_block: int) -> np.ndarray:
    """Draws of a manager's effect at now_block: OU projection from his last observed state."""
    st = states[states.manager == manager]
    last = st.loc[st.block.idxmax()]
    th = post["theta"].values.reshape(-1, post["theta"].shape[-1])[:, last.name]
    mu = post["mu"].sel(manager=manager).values.ravel()
    gamma, tau = post["gamma"].values.ravel(), post["tau"].values.ravel()
    rho = np.exp(-gamma * (now_block - last.block) * BLOCK_YEARS)
    z = np.random.default_rng(0).normal(size=len(mu))
    return mu + rho * (th - mu) + np.sqrt(1 - rho**2) * tau * z


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="base", choices=VARIANTS + ["all"])
    p.add_argument("--summarise", action="store_true")
    a = p.parse_args()
    cfg = load_config()
    if a.summarise:
        from pochettino_fit.summary import write_summary
        print(write_summary(cfg))
        return
    for v in VARIANTS if a.variant == "all" else [a.variant]:
        run_variant(v, cfg)


if __name__ == "__main__":
    main()
