# Methods (version 2): Pochettino at United

Every part of the model: the equation, the assumptions, why it is there, and where it can break.
Version 2 replaces the two-stage Elo design of version 1 with a single-stage pairwise model, after review.

---

## 0. Data and notation

| Symbol | Meaning |
|---|---|
| k | one league match, ONE row per match, from the home team's view |
| H(k), A(k) | home and away team |
| l, s | league and season |
| c | club; m | manager; b | half-season block (time unit for manager drift) |
| w_k | minutes played 11 v 11 / 90 (match cut at the first red card) |
| y_k | (home npxG − away npxG) per 90 over the 11 v 11 minutes |

Scope: league matches in the big five leagues, 2014-15 to now. No cups and no European matches, for every team and manager alike.

**Why one row per match.** With one row per team, each match appears twice with mirrored outcomes, so the errors are perfectly negatively correlated. That doubles the apparent sample and makes intervals about 1.4 times too narrow.

**Why npxG difference.** Goals and points carry much more luck. xG difference predicts future results better. Penalties are rare and weakly linked to how a team plays.

**Weak points**
- xG only sees shots.
- Providers' xG models differ. Calibrate one source on another with a season both cover before mixing them.
- Truncating at red cards removes slightly more data from aggressive teams.

---

## 1. The single-stage pairwise model

y_k = h_{l,s} + (S_{H,k} − S_{A,k}) + (θ_{m(H),b} − θ_{m(A),b}) + (δ_{τ(H)} − δ_{τ(A)}) + β (L_{H,k} − L_{A,k}) + ε_k

ε_k ~ N(0, σ² / w_k)

| Term | Meaning |
|---|---|
| h_{l,s} | Home advantage for league l in season s. Hierarchical around a global mean, so 2020-21 (no crowds) can drop toward zero. |
| S_{c,k} | Team strength, section 1a |
| θ_{m,b} | Manager effect in time block b, section 2 |
| δ_τ | Tenure effect, shared by all managers: τ = year 1, 2, 3+ of a spell |
| L_{c,k} | Recent luck: goals minus npxG over the team's previous 10 league matches |
| β | How much recent luck predicts the next match's xG difference. Expected near zero; its job is to stop regression to the mean being credited to a new manager. |

### 1a. Team strength

S_{c,k} = λ_{l,s} + κ_c + b_V · log V_{c,s−1}

- **λ_{l,s}: league-season offset.** Within a league, teams only play each other, so the data cannot say how strong one league is compared with another. The bridge is a prior: λ_{l,s} ~ N(α · Ē_{l,s}, σ_λ²), where Ē_{l,s} is that league's average ClubElo at season start. A league average is not driven by any single manager, so it does not have the feedback problem of team-level Elo (below).
- **κ_c: club effect.** Structure, recruitment, facilities. Optional extension: let κ drift slowly, with the same process as the manager effect, since clubs change (United's structure since 2024).
- **log V_{c,s−1}: pre-season squad value** from the previous summer. Lagged, so a manager's own success raising player values does not leak in. The log reflects diminishing returns.

**Why not team-level Elo any more.** ClubElo is a smoothed record of results.
1. It already contains the club effect, so subtracting it leaves κ_c measuring only the gap between a club's xG and its results.
2. It catches up with a good manager during his spell, so his residuals shrink in years two and three even if performance stays the same. The tenure "fade" would then be fake.
Estimating strength inside one model avoids both.

### 1b. Identification

- Manager vs club: identified because managers move between clubs (the data form connected networks of managers and clubs). A manager who only ever coached one club is identified only through priors.
- Level constraints: κ_c sums to zero within each league; λ carries the league level.
- League vs league: identified only through the Elo-anchored prior on λ. State this on the page.

**Weak points**
- The league bridge is the weakest link. If the league-average Elo is off, every cross-league comparison shifts, including Pochettino's PSG spell against his Premier League spells. Test: refit with α fixed at plus and minus one standard error and report how much his estimate moves.
- Squad value is a market estimate, not a wage bill.
- The model is additive and linear in xG difference. Very lopsided matches may not fit.

---

## 2. Manager effect over time

θ_{m,b} = μ_m + η_{m,b}

η_{m,b} = e^{−γ Δ_b} η_{m,b−1} + ν_{m,b},  ν_{m,b} ~ N(0, τ² (1 − e^{−2γ Δ_b}))

- μ_m: long-run level. η: current deviation, which decays toward zero.
- Δ_b: years between the manager's consecutive blocks with data. A long gap lets η fade and widens uncertainty.
- γ > 0: decay rate, estimated across all managers. Half-life of a deviation = ln 2 / γ years.
- This is the exact discretisation of an Ornstein-Uhlenbeck process, so it stays stationary (variance τ²) whatever the gaps.
- **Blocks, not matches.** One latent state per half-season. A state per match (hundreds per manager, each almost identical to the last) makes the sampler grind.

### 2a. Partial pooling, non-centred

μ_m = σ_μ z_m, z_m ~ N(0, 1);  κ_c = σ_κ z_c;  η built from standard-normal innovations.
Without this, the sampler struggles when σ_μ is small (the "funnel").

For intuition, with no time drift a manager's estimate is pulled toward zero by:
μ̂_m ≈ B · r̄_m,  B = σ_μ² / (σ_μ² + σ² / n_eff)

### 2b. International spells

For each national-team match q under manager m:

r_q = λ_int · θ_{m,b(q)} + ε_q,  ε_q ~ N(0, σ_int² / w_q)

- r_q = result minus World Football Elo expected result (as in the USA script); w_q = 1 competitive, about 0.5 friendly.
- λ_int ~ N(0.5, 0.5²): wide, because international management is a different job.
- These matches inform the effect only, never the style signature.

**Weak points**
- Selection: managers are sacked after bad runs, which cuts their worst data short. Hiring after bad luck is handled by β, not removed entirely.
- The manager is credited with everything around him: signings made for him, assistants, injuries.
- "Average" (zero) means the average manager in the big five leagues, already a high bar.
- With few managers having both club and international data, λ_int is weakly identified. Report Pochettino's estimate with and without the USA spell.

---

## 3. Finishing skill

For player p, career non-penalty goals G_p and npxG X_p:

G_p − X_p = φ_p X_p + e_p,  Var(e_p) ≈ 0.85 X_p

- The variance comes from shot outcomes being roughly independent Bernoulli trials: Σ p_s(1 − p_s) ≈ (1 − average shot xG) × X_p.
- So the estimate φ̂_p = (G_p − X_p) / X_p has variance ≈ 0.85 / X_p: precision grows in proportion to xG.
- Prior φ_p ~ N(0, σ_φ²), σ_φ ≈ 0.05. Tight, because skill is small and career totals overstate it (players who finish badly early leave big clubs and take fewer shots).
- Team multiplier: f_U = 1 + Σ_p π_p φ_p, with π_p each player's expected share of United's npxG.

**Weak points:** career data cross teams and xG providers; opposition goalkeeping is ignored.

---

## 4. Style and fit

### 4a. Style model

For each style metric d (possession, field tilt, shots per 90, xG per shot for and against, set-piece share of xG, pressing proxy), the same structure as section 1, applied to team-match values x_{k,d}:

x_{c,k,d} = λ_{l,s,d} + κ_{c,d} + θ_{m,d} + ε

(one row per team per match is fine here, because each team's own style metric is not the mirror of the opponent's. Possession is the exception: use the home team only.)
Shot-based metrics are computed in level or one-goal game states where shot timings allow.

### 4b. The quantity that matters: predicted change

- United's style now ≈ κ_U + θ_Carrick
- United's predicted style under Pochettino ≈ κ_U + θ_P
- **Predicted change** Δ = θ_P − θ_Carrick (one value per style dimension)

Distance: D² = Δᵀ R⁻¹ Δ, with R the correlation matrix of style metrics, so correlated metrics (possession and field tilt) are not counted twice. Compute D² for every posterior draw and report its distribution, and its percentile among all real managerial changes in the data.

**Why not compare his effect with United's profile directly.** His effect is a shift relative to a club; United's profile is a level. They are different quantities. Rescaling his shrunk effects to match would also undo the shrinkage and inflate noise.

**Imposition strength:** share of a manager's teams' style variation explained by θ_m rather than κ_c. High means he imposes his system wherever he goes.

**Weakness check:** his effect on xG per shot against, big chances against and set-piece xG against (the dimensions where United are weak).

**Weak points**
- Style fit predicting results is a hypothesis. Test it in the back-test; if it does not predict, report it as description only.
- Carrick's effect is estimated from a short spell, so Δ is uncertain.
- Possession and field tilt cannot be filtered by game state with match-level data.

---

## 5. Forecast

### 5a. Attack and defence model for the forecast

From the same data, a log-link model for npxG for each side:

log E[npxG_{i vs j}] = μ + h·home + a_i − d_j + θ^a_{m(i)} − θ^d_{m(j)} + δ^a_{τ(i)} − δ^d_{τ(j)}

The log link keeps rates positive (an additive model can predict negative xG for a weak attack against a strong defence).

### 5b. Replacing the outgoing manager

United's current attack and defence estimates were formed under Carrick, so they contain his effect and his tenure stage. Strip them out first:

a_U⁰ = a_U − θ^a_Carrick − δ^a_{τ(Carrick)}  (same for defence)

Then two runs:
- **Baseline:** a_U⁰ + an average manager (θ = 0) + δ^a_1
- **Pochettino:** a_U⁰ + θ^a_P + δ^a_1

His contribution = Pochettino run − baseline run. The uncertainty in Carrick's effect carries through.

### 5c. Simulating a match

For each posterior draw and each remaining fixture:
1. Expected npxG for each side from 5a.
2. **Match-level xG shock:** actual npxG = expected × e^{ζ}, ζ ~ N(−s²/2, s²) (mean-preserving), with s fitted from the spread of match xG around its expectation. Without this step, simulated scorelines have too little variance (too few thrashings and shock results).
3. Goals ~ Poisson(f_i × actual npxG + penalty rate). Add the Dixon–Coles low-score correction if the back-test shows draws are under-predicted.
4. Points, then the league table (simulate every remaining fixture in the league, other teams from their current estimates).

Outputs: points for the rest of the season with an interval; final-position distribution; P(top half); P(bottom three). The squad is frozen at the freeze date.

**Weak points:** other teams' strengths are fixed; January transfers and injuries are not modelled; points intervals will be wide (say so).

---

## 6. Detectability

Rough frequentist version, with match noise σ and effect Δ:
n ≈ (1.96 + 0.84)² σ² / Δ². With σ ≈ 1.2 and Δ ≈ 0.2, n ≈ 280 matches.

Bayesian version: posterior sd after n matches ≈ (1/σ_post,0² + n·w̄/σ²)^(−1/2).

Use the fitted σ and Pochettino's posterior sd, and state the answer on the page. The tracker will mostly show noise this season, and that is expected.

---

## 7. Tracker

After each United league match: add it, refit (or approximate with a normal update between refits: precision adds w_k/σ², mean is the precision-weighted average), and append:
- forecast npxG difference and interval vs actual;
- cumulative points vs the frozen forecast band;
- chance quality conceded (xG per shot against, big chances against) vs United's level before the appointment;
- updated posterior for his effect at United.

The frozen forecast is never edited.

---

## 8. Validation

1. **Synthetic recovery:** simulate leagues with known κ, μ, γ, δ, β and home advantage (including managers whose quality drifts, and managers who change club). The model must recover them within its intervals about as often as the intervals claim.
2. **Back-test** on past mid-season appointments: cut all data at the appointment date, forecast, compare.
   - Coverage of 50% and 80% intervals.
   - PIT histogram (flat = calibrated).
   - Log score and CRPS against simpler models: no manager effect; no time drift; no luck term.
3. **Ablation:** keep a component only if it improves the back-test.
4. **Sensitivity:** league bridge (α ± 1 se), with and without the USA spell, half-season vs season blocks.

---

## 9. Biggest remaining risks

1. The league bridge: cross-league comparisons rest on the Elo-anchored prior.
2. Little information: three club spells plus the USA; most of his forecast effect comes from the prior and older spells.
3. Constant club effect, unless the drifting version is used.
4. Style fit is untested until the back-test says otherwise.
5. Mixing xG providers without calibration.
