"""Simulate leagues from the exact model in docs/methods.md sections 1-2 (known parameters).

Output tables use the same columns as data/processed/matches.parquet, so the manager model
runs unchanged on synthetic and real data. Truth is returned alongside.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

BLOCK_YEARS = 0.5


def block_of(date: pd.Timestamp) -> int:
    """Half-season index: 2 * season start year (+1 for January-June)."""
    start = date.year if date.month >= 7 else date.year - 1
    return 2 * start + (0 if date.month >= 7 else 1)


def round_robin(n: int) -> list[list[tuple[int, int]]]:
    """Double round robin by the circle method: list of matchdays of (home, away)."""
    idx = list(range(n))
    days = []
    for r in range(n - 1):
        pairs = [(idx[i], idx[n - 1 - i]) for i in range(n // 2)]
        days.append([(a, b) if r % 2 == 0 else (b, a) for a, b in pairs])
        idx = [idx[0], idx[-1], *idx[1:-1]]
    return days + [[(b, a) for a, b in d] for d in days]


def ou_paths(n: int, n_blocks: int, tau: float, gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary OU deviations, exact discretisation per half-season block."""
    rho = np.exp(-gamma * BLOCK_YEARS)
    eta = np.empty((n, n_blocks))
    eta[:, 0] = rng.normal(0, tau, n)
    for b in range(1, n_blocks):
        eta[:, b] = rho * eta[:, b - 1] + np.sqrt(1 - rho**2) * tau * rng.normal(size=n)
    return eta


@dataclass
class World:
    t: dict[str, Any]
    rng: np.random.Generator
    first_year: int
    n_blocks: int
    mu: np.ndarray
    eta: np.ndarray
    kappa: dict[str, float] = field(default_factory=dict)
    value_level: dict[str, float] = field(default_factory=dict)
    club_league: dict[str, int] = field(default_factory=dict)
    manager_of: dict[str, tuple[int, pd.Timestamp]] = field(default_factory=dict)
    unemployed: list[int] = field(default_factory=list)
    next_manager: int = 0
    luck_hist: dict[str, list[float]] = field(default_factory=dict)
    employed_blocks: dict[int, set[int]] = field(default_factory=dict)

    def theta(self, m: int, block: int) -> float:
        return self.mu[m] + self.eta[m, block - 2 * self.first_year]

    def new_manager(self) -> int:
        self.next_manager += 1
        return self.next_manager - 1

    def hire(self, club: str, date: pd.Timestamp) -> None:
        use_pool = self.unemployed and self.rng.random() < self.t["rehire_prob"]
        m = self.unemployed.pop(self.rng.integers(len(self.unemployed))) if use_pool else self.new_manager()
        self.manager_of[club] = (m, date)

    def fire(self, club: str) -> None:
        self.unemployed.append(self.manager_of.pop(club)[0])


def make_world(size: dict[str, int], t: dict[str, Any], seed: int) -> World:
    rng = np.random.default_rng(seed)
    n_blocks = 2 * size["seasons"]
    n_mgr = 20 * size["leagues"] * size["clubs"] + 100
    w = World(t=t, rng=rng, first_year=2014, n_blocks=n_blocks,
              mu=rng.normal(0, t["sigma_mu"], n_mgr),
              eta=ou_paths(n_mgr, n_blocks, t["tau"], t["gamma"], rng))
    for lg in range(size["leagues"]):
        names = [f"L{lg}C{c:02d}" for c in range(size["pool_clubs"])]
        k = rng.normal(0, t["sigma_kappa"], len(names))
        k -= k.mean()  # sum to zero within league
        for name, kc in zip(names, k, strict=True):
            w.kappa[name], w.club_league[name] = kc, lg
            w.value_level[name] = 19.5 + 0.6 * kc / t["sigma_kappa"] + rng.normal(0, 0.3)
    return w


def season_params(w: World, size: dict[str, int], s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Home advantage, league offsets and league Elo for season index s."""
    t, rng = w.t, w.rng
    h = t["h0"] + rng.normal(0, t["sigma_h"], size["leagues"])
    if s == t["no_crowd_season"]:
        h = np.full(size["leagues"], t["no_crowd_h"]) + rng.normal(0, 0.02, size["leagues"])
    elo = 1600 + 60 * np.arange(size["leagues"])[::-1] + rng.normal(0, 15, size["leagues"])
    elo_z = (elo - elo.mean()) / 100
    lam = t["alpha"] * elo_z + rng.normal(0, t["sigma_lambda"], size["leagues"])
    return h, lam, elo


def luck_of(w: World, club: str) -> tuple[float, int]:
    hist = w.luck_hist.get(club, [])[-10:]
    return (float(np.mean(hist)), len(hist)) if hist else (np.nan, 0)


def tenure_of(start: pd.Timestamp, date: pd.Timestamp) -> int:
    return int(min(3, np.floor((date - start).days / 365.25) + 1))


def play(w: World, home: str, away: str, date: pd.Timestamp, ctx: dict[str, Any]) -> dict[str, Any]:
    """Simulate one match from the methods.md equation and update luck histories."""
    t, rng, blk = w.t, w.rng, block_of(date)
    row: dict[str, Any] = {"date": date, "block": blk, "home_team": home, "away_team": away}
    mean = ctx["h"]
    for side, club, sign in (("home", home, 1), ("away", away, -1)):
        m, start = w.manager_of[club]
        w.employed_blocks.setdefault(m, set()).add(blk)
        ten = tenure_of(start, date)
        luck, n = luck_of(w, club)
        logv = ctx["logv"][club]
        strength = ctx["lam"] + w.kappa[club] + t["b_value"] * logv
        delta = t["delta"][ten - 1] if ten < 3 else 0.0
        mean += sign * (strength + w.theta(m, blk) + delta + t["beta"] * (0.0 if n == 0 else luck))
        row.update({f"{side}_manager": f"M{m:04d}", f"{side}_tenure": ten, f"{side}_luck": luck,
                    f"{side}_luck_n": n, f"{side}_log_squad_value": logv})
    minutes = rng.uniform(10, 90) if rng.random() < t["red_card_prob"] else 90.0
    wgt = minutes / 90
    y = rng.normal(mean, t["sigma"] / np.sqrt(wgt))
    base = 1.35 + abs(y) / 2
    xg = {"home": base + y / 2, "away": base - y / 2}
    goals = {s: rng.poisson(xg[s]) for s in xg}
    for side, club, op in (("home", home, "away"), ("away", away, "home")):
        net = (goals[side] - xg[side]) - (goals[op] - xg[op])
        w.luck_hist.setdefault(club, []).append(net)
    row.update({"minutes_11v11": minutes, "w": wgt, "y": y, "home_goals": goals["home"],
                "away_goals": goals["away"], "true_mean": mean})
    return row


def manager_moves(w: World, clubs: list[str], points: dict[str, float], date: pd.Timestamp,
                  season_end: bool) -> None:
    """Sack bottom-3 managers with probability sack_prob; at season end some managers leave."""
    for club in sorted(clubs, key=lambda c: points.get(c, 0))[:3]:
        if w.rng.random() < w.t["sack_prob"]:
            w.fire(club)
            w.hire(club, date)
    if season_end:
        for club in clubs:
            if w.rng.random() < w.t["leave_prob"]:
                w.fire(club)
                w.hire(club, date)


def promote_relegate(w: World, active: list[str], table: dict[str, float], next_year: int) -> list[str]:
    lg = w.club_league[active[0]]
    inactive = [c for c, v in w.club_league.items() if v == lg and c not in active]
    n_swap = min(3, len(inactive))
    down = sorted(active, key=lambda c: table[c])[:n_swap]
    up = list(w.rng.choice(inactive, size=n_swap, replace=False))
    for c in up:
        if c not in w.manager_of:
            w.hire(c, pd.Timestamp(f"{next_year}-07-01"))
    return [c for c in active if c not in down] + up


def simulate_league_season(w: World, lg: int, s: int, active: list[str], ctx: dict[str, Any]) -> list[dict]:
    year = w.first_year + s
    rows, half_pts = [], {c: 0.0 for c in active}
    days = round_robin(len(active))
    for d, day in enumerate(days):
        date = pd.Timestamp(f"{year}-08-15") + pd.Timedelta(days=7 * d + (0 if d < 19 else 14))
        if d == 19:  # January: half-season decisions
            manager_moves(w, active, half_pts, date - pd.Timedelta(days=1), season_end=False)
            half_pts = {c: 0.0 for c in active}
        for i, j in day:
            r = play(w, active[i], active[j], date, ctx)
            hg, ag = r["home_goals"], r["away_goals"]
            half_pts[active[i]] += 3 if hg > ag else 1 if hg == ag else 0
            half_pts[active[j]] += 3 if ag > hg else 1 if hg == ag else 0
            ctx["table"][active[i]] = ctx["table"].get(active[i], 0) + (3 if hg > ag else hg == ag)
            ctx["table"][active[j]] = ctx["table"].get(active[j], 0) + (3 if ag > hg else hg == ag)
            rows.append({**r, "league": f"L{lg}", "season": f"{year}-{(year + 1) % 100:02d}",
                         "league_elo": ctx["elo"]})
    manager_moves(w, active, half_pts, pd.Timestamp(f"{year + 1}-06-30"), season_end=True)
    return rows


def international(w: World, n_mgr: int) -> pd.DataFrame:
    """Separate noisier observations for a few managers in blocks when they had no club job."""
    t, rng, rows = w.t, w.rng, []
    candidates = [m for m in range(n_mgr) if m in w.employed_blocks]
    for m in rng.choice(candidates, size=min(t["n_international"], len(candidates)), replace=False):
        free = [b for b in range(2 * w.first_year, 2 * w.first_year + w.n_blocks)
                if b not in w.employed_blocks[m]]
        for blk in rng.choice(free, size=min(2, len(free)), replace=False) if free else []:
            for _ in range(t["intl_matches_per_block"]):
                wq = 1.0 if rng.random() < 0.6 else 0.5
                r = rng.normal(t["lambda_int"] * w.theta(m, blk), t["sigma_int"] / np.sqrt(wq))
                rows.append({"manager": f"M{m:04d}", "block": int(blk), "w": wq, "r": r})
    return pd.DataFrame(rows, columns=["manager", "block", "w", "r"])


def simulate(size: dict[str, int], t: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    w = make_world(size, t, seed)
    actives = {}
    for lg in range(size["leagues"]):
        actives[lg] = [f"L{lg}C{c:02d}" for c in range(size["clubs"])]
        for c in actives[lg]:
            w.manager_of[c] = (w.new_manager(), pd.Timestamp("2014-07-01") - pd.Timedelta(
                days=int(w.rng.integers(0, 1500))))
    rows, h_truth, lam_truth = [], {}, {}
    for s in range(size["seasons"]):
        h, lam, elo = season_params(w, size, s)
        for lg in range(size["leagues"]):
            season = f"{w.first_year + s}-{(w.first_year + s + 1) % 100:02d}"
            h_truth[(f"L{lg}", season)], lam_truth[(f"L{lg}", season)] = h[lg], lam[lg]
            logv = {c: w.value_level[c] + w.rng.normal(0, 0.3) for c in actives[lg]}
            ctx = {"h": h[lg], "lam": lam[lg], "elo": elo[lg], "logv": logv, "table": {}}
            rows += simulate_league_season(w, lg, s, actives[lg], ctx)
            actives[lg] = promote_relegate(w, actives[lg], ctx["table"], w.first_year + s + 1)
    m = pd.DataFrame(rows).sort_values(["date", "league"]).reset_index(drop=True)
    m["match_id"] = "s" + m.index.astype(str)
    intl = international(w, w.next_manager)
    truth = truth_tables(w, m, intl, h_truth, lam_truth)
    return m, intl, truth


def truth_tables(w: World, m: pd.DataFrame, intl: pd.DataFrame, h: dict, lam: dict) -> dict:
    t = w.t
    managers = sorted(set(m.home_manager) | set(m.away_manager) | set(intl.manager))
    ids = [int(x[1:]) for x in managers]
    blocks = pd.concat([m[["home_manager", "block"]].set_axis(["manager", "block"], axis=1),
                        m[["away_manager", "block"]].set_axis(["manager", "block"], axis=1),
                        intl[["manager", "block"]]]).drop_duplicates()
    blocks["theta"] = [w.theta(int(x[1:]), b) for x, b in zip(blocks.manager, blocks.block, strict=True)]
    clubs = sorted(set(m.home_team) | set(m.away_team))
    kappa = pd.Series([w.kappa[c] for c in clubs], index=clubs)
    kappa -= kappa.groupby(kappa.index.str[:2]).transform("mean")  # same constraint as the model
    return {
        "scalars": {"sigma": t["sigma"], "tau": t["tau"], "gamma": t["gamma"], "beta": t["beta"],
                    "b_value": t["b_value"], "delta_1": t["delta"][0], "delta_2": t["delta"][1],
                    "sigma_mu": t["sigma_mu"], "sigma_kappa": t["sigma_kappa"],
                    "lambda_int": t["lambda_int"], "sigma_int": t["sigma_int"]},
        "mu": pd.Series(w.mu[ids], index=managers),
        "theta": blocks.set_index(["manager", "block"]).theta,
        "kappa": kappa,
        "h": pd.Series(h), "lambda": pd.Series(lam),
    }
