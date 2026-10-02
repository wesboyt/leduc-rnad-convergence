# Timing: why the sampled rule does not converge, and what the literature says about it

Notation from `docs/ALGORITHM.md` (§ numbers refer to it). Claims C1–C12 are defined and testable in `docs/CLAIMS.md`
/ `reproduce.py`.

## 0. Summary

Regularised learning in zero-sum games runs on **two clocks**. The *inner* clock solves the regularised game around
the current magnet $\rho_k$. The *outer* clock moves the magnet. Theory guarantees last-iterate convergence to a Nash
equilibrium only when the inner problem is (approximately) solved before the magnet moves, the inner update is a
gradient/mirror step small relative to the regularisation, and the residual regularisation toward a *fixed* policy
vanishes. On 2-player Leduc with exact NashConv we find:

* With exact values and a mirror step, the scheme converges: NashConv 0.0006 (C3).
* The sampled rule breaks all three conditions:
  - its optimizer does not take mirror-like steps (C6, C7);
  - it keeps fixed pulls toward uniform (a floor of 0.109, C4);
  - its inner loop, a per-information-set stochastic approximation, is ~10× slower than the refresh interval and
    carries ~30× more magnet error than the outer loop tolerates (C8, C12).

None of this contradicts the theory. Every failure is outside a hypothesis of the theorems. What is new is the
quantification for this sampled rule: the magnet-error budget per refresh, and the inner-loop time of the sampled step.

## 1. Two time scales

**Inner problem.** For a fixed magnet $\rho_k$ the target is the regularised equilibrium $\pi^\star(\rho_k)$ (§2).
Write $\pi_{k,t}$ for the iterate $t$ steps after the $k$-th refresh, and measure the inner error
$e_{k,t} = D(\pi_{k,t}, \pi^\star(\rho_k))$. In the experiments $D$ is the reach-weighted RMS of the log-probability
error per round (and total variation per row).

**Exact inner rate.** For the closed-form MMD step (§3) with $\eta$ small relative to $\alpha$, Sokota et al. (2023)
prove linear convergence to $\pi^\star(\rho)$, with per-step contraction of order $\eta\alpha/(1+\eta\alpha)$ in a
Bregman divergence. Measured (C2): at $\eta\alpha = 0.006$ the residual falls ~10× per 1000 steps
(≈ 0.0023 per step), so the inner time to 1% is ~1500–2000 steps (C12).

**Outer problem.** $\rho_{k+1} \leftarrow \pi_{k,K}$ is one step of a proximal-point method on the game's monotone
variational inequality, with the dilated KL as Bregman divergence and step $1/\alpha$. Perolat et al. (2021) show that
iterating the regularised fixed point converges to a Nash equilibrium. For proximal-point methods with *inexact*
steps, the classical condition (Rockafellar, 1976) is summable errors, $\sum_k \varepsilon_k < \infty$. With an error
that does not shrink, the iterates stall in a neighbourhood whose size grows with the error and shrinks with the outer
contraction. In bilinear (zero-sum) games the outer contraction per refresh is weak: the exact reference needs tens of
refreshes to go from the QRE floor (0.19) to 0.01 (C3).

**Consequence: the error budget.** Because the outer contraction is weak, a small per-refresh error is amplified.
Measured on the exact reference with Gaussian noise of standard deviation $\sigma$ on the refreshed magnet's
log-probabilities (C8; 40k steps, K = 500):

| $\sigma$ per refresh | 0 | 0.01 | 0.03 | 0.1 | 0.3 |
|---|---|---|---|---|---|
| NashConv | 0.0045 | 0.0099 | 0.044 | 0.077 | 0.35 |

To reach NashConv ~0.01, the inner solve must hand the magnet an error of roughly 0.01–0.03 in log-probability.

## 2. The timing issues

### T1 — The magnet moves before the inner problem is solved (the "ratchet")

*Definition.* The refresh period $K$ (or the plateau test) is shorter than the inner time $T_{\text{in}}(\varepsilon)$
needed to reach the error budget. Every refresh then writes the current inner error into the magnet. The next inner
problem is centred on that error and does not remove it.

*Evidence.*
- The original plateau rule fired at its minimum period in every arm (≈ every 500 steps). With exact Q and a strong
  magnet, the last iterate reached ~0.3 and then degraded to 1.1–1.5 under ~15 refreshes; with refreshes every 2000
  it did not degrade.
- In the sampled rule the iterate is far from $\pi^\star(\rho_k)$ at every refresh (`experiments/refresh_error.py`):
  round-2 RMS log-probability error 1.5–2.6, some rows with total variation > 0.9.

*Literature.* DeepNash / R-NaD (Perolat et al. 2022) moves the regularisation policy only after the dynamics have
been run toward the fixed point. It interpolates the reward transformation between the two most recent regularisation
policies to avoid a discontinuous target. Perolat et al. (2021) analyse the iteration of exact fixed points.

### T2 — The optimizer is not a mirror/gradient step

*Definition.* The theorems are for (mirror-)gradient steps whose size scales with the gradient. Adam normalises each
coordinate, so near the fixed point its step does not shrink with the gradient. Momentum (and schedule-free
interpolation between the iterate and its running average) adds delayed feedback to a rotational vector field.

*Evidence (C6, C7, C9).*
- On the identical exact objective, schedule-free AdamW ends at NashConv 4.4–6.0 (worse than uniform play, 4.75) at
  every learning rate tried. The same objective reaches 0.006 with the mirror step and 0.02–0.06 with SGD.
- With Adam, error grows with $\beta_1$: 0.04–0.07 at $\beta_1 = 0$, 0.13–1.15 at 0.9.
- Momentum-free Adam still stalls at ~0.04: the step does not shrink near the fixed point.
- With sampled values, schedule-free AdamW degrades to ~4. SGD is the only stable optimizer, at ~0.3.
- Large $\epsilon$ in Adam does not rescue it (0.15–4.7).

*Literature.*
- Simultaneous gradient dynamics need not converge in min-max problems. Daskalakis & Panageas (2018) characterise the
  stable limit points of gradient descent-ascent and its optimistic variant.
- Gidel et al. (ICLR 2019) analyse games as variational inequalities. They show how extrapolation and averaging
  restore convergence where simultaneous steps fail.
- Optimistic and look-ahead methods give last-iterate convergence in extensive-form games (Lee, Kroer & Luo 2021).
  With entropy regularisation they converge linearly to the regularised equilibrium (Cen et al. 2023).
- None of these covers Adam's per-coordinate normalisation. The negative effect of positive momentum on game
  dynamics is studied in Gidel et al., "Negative Momentum for Improved Game Dynamics" (AISTATS 2019; not in the
  original reading list, added because it addresses momentum directly).

Our result is consistent with this literature. It quantifies how far outside the hypotheses this particular
optimizer lands.

### T3 — The sampled inner loop is a per-information-set stochastic approximation

*Definition.* With one pivot per hand and a fixed quota, the expected step at information set $s$ is proportional to
$\gamma f_s \bar w_s$: the learning rate times the sampling frequency times the mean importance weight. That makes it
proportional to the counterfactual reach (§5.1). The step's variance is proportional to $\gamma$. Hence:

* the per-row inner time is $\propto 1/(\gamma f_s \bar w_s \alpha)$, so rarely reached rows (most of round 2) set
  the inner time;
* the per-row error floor is $\propto \gamma$, so frequently reached rows (round 1) sit in a noise ball whose size
  grows with $\gamma$.

There is no single $\gamma$ that makes the slow rows fast and the noisy rows quiet.

*Evidence (C10, C12).*
- With a FIXED magnet and EXACT Q, after 16k steps (32 refresh periods), the sampled rule's round-2 RMS log-prob
  error is still 0.9 (falling from 1.9). Round-1 error sits at 0.25–0.45 at lr 10 and 0.6–0.7 at lr 30. NashConv is
  0.19–0.25 against the exact QRE's 0.140. The exact reference solves the same game to 1% in < 2000 steps.
- Exact and sampled Q give the same ~0.3 plateau, so the noise is in which rows get updated, not in the values.
- 4× batch, lr 3–30, refresh 2000, an averaged magnet and an inner learning-rate decay leave it at 0.31–0.47.

*Literature.* Abe et al. (APMD, ICML 2024) prove last-iterate convergence under noisy feedback for mirror descent
perturbed toward a slingshot (magnet) that is updated at an interval. Our reading of their analysis: the learning
rate and the slingshot interval must be chosen with the noise level in mind. The sampled rule uses a constant step
and an interval chosen without reference to the noise, and its noise is heterogeneous across information sets
(through $f_s$). The tabular sampled harness has no generalisation between information sets, so T3 is at its worst
here. A function approximator shares statistical strength across information sets and changes this trade-off; that is
untested in this repository.

*Update (C18):* APMD's prescription transplanted into the sampled rule (learning rate restarted at each magnet update
and decayed as $\propto 1/(1 + s/500)$; interval 4000 ≈ $T^{4/5}$ for $T$ = 32k) lowers the floor to 0.225 from 0.377 with
the same interval and a constant step. This is constructive evidence for T1/T3.

### T4 — Fixed regularisation toward uniform sets a bias floor

*Definition.* The fixed pulls $\eta_{\text{unif}}$ and $\alpha_{\text{ent}}$ add a temperature $\tau$ toward the uniform
policy that never moves. However the magnet moves, the limit is at best the QRE of temperature $\tau$ around the moving
magnet, and no better than the QRE of $\tau$ itself.

*Evidence (C4).* The QRE of the baseline's $\tau \approx (0.072, 0.21)$ chips has NashConv 0.1087. With the refreshing
magnet and $\tau$ on, the exact reference pins at exactly 0.1087.

*Literature.* MMD with a fixed magnet converges to the QRE of its temperature (Sokota et al. 2023). Approaching Nash
needs the magnet to move or the temperature to be annealed. Cen et al. (2023) bound the regularisation bias by the
temperature.

### T5 — Refreshing the magnet to the iterate compounds lost support

*Definition.* Each refresh multiplies a dominated action's probability by roughly $e^{-\Delta Q/\alpha}$. After tens of
refreshes, probabilities underflow to exactly 0. An action that becomes good when the opponent shifts cannot come back.

*Evidence (C3).* Without a floor, min π = 0 from early on and NashConv spikes late (0.02 → 0.6 at α = 0.5). A
log-probability floor (−20 or −40) removes the large late spikes and reaches 0.0006.

*Literature.* MMD's analysis assumes interior iterates. DeepNash thresholds NeuRD's logits and discards
low-probability actions in post-processing for related reasons.

## 3. What was tested and is NOT the cause

| hypothesis | test | result |
|---|---|---|
| node-only anchor (raw continuation values) instead of trajectory regularisation | C5 | converges as well (0.0022 vs 0.0028) |
| the magnet is too weak at $\eta_{\text{reg}} = 0.2$ | reference at the baseline's temperature in chips | 0.2 is in the best range; 1.0 is too strong (0.058 at 60k) |
| noise in the action values | C10 | exact Q plateaus like sampled Q |
| the expected step has the wrong fixed point | C11 | with clips off, the expected gradient at the exact regularised equilibrium is ~3% of its value at uniform; the clips add some bias |
| per-row step ∝ counterfactual reach | `mmd_grid.py --grid cfw` | ~2× slower in the exact reference, not a floor |
| lost support in the sampled rule | `ladder.py --grid floor` | logit floor 20 changes nothing; 10 changes little |
| the magnet should be an average / the inner lr should decay | `ladder.py --grid avg` | 0.31–0.38, unchanged |

## 4. Relation to the reading list

| work | what it establishes (as used here) | key hypotheses | our observation |
|---|---|---|---|
| Perolat et al. 2021, *From Poincaré recurrence to convergence in imperfect information games* | reward regularisation toward a magnet turns recurrent (cycling) FoReL dynamics into convergent ones; iterating the regularised fixed point converges to Nash | exact (continuous-time) dynamics; fixed point reached before the magnet moves | consistent: exact reference converges (C3); violations of "reached before" are T1/T3 |
| Perolat et al. 2022, *DeepNash / R-NaD* | the practical recipe: reward transform, dynamics run toward the fixed point, then a new regularisation policy, with interpolation between the two most recent; NeuRD with logit thresholding | enough inner steps per iteration; large batches | the sampled rule refreshes on a short clock (T1) without the support safeguards (T5) |
| Sokota et al. 2023, *MMD* | linear last-iterate convergence to the QRE when the step is small relative to the regularisation; moving magnet / annealing for Nash | mirror step; exact or unbiased low-variance values | reproduced (C2, C3); the fixed pulls are a QRE floor (C4) |
| Abe et al. 2024, *APMD* | last-iterate convergence under noisy feedback with a slingshot updated at an interval | step sizes and update interval chosen for the noise | sampled per-row noise is heterogeneous and the step constant (T3) |
| Daskalakis & Panageas 2018 | stable limit points of GDA / optimistic GDA in min-max; plain dynamics need not converge | smooth min-max, gradient steps | consistent with T2; Adam is outside the analysis |
| Lee, Kroer & Luo 2021 | last-iterate convergence of optimistic dilated mirror-descent methods in extensive-form games | optimism; (for rates) a unique equilibrium | an optimistic variant of the sampled rule is untested here (open) |
| Cen et al. 2023 | entropy-regularised optimistic policy updates converge linearly (last iterate) to the regularised equilibrium of zero-sum Markov games | regularisation; exact or controlled-error values | T4 bias floor; the linear rate matches C2 |
| Gidel et al. 2019, *A variational inequality perspective on GANs* | extrapolation and averaging fix the non-convergence of simultaneous gradient steps on games | monotone VI | T2; averaging the magnet did not fix T3 (`--grid avg`) |
| Rockafellar 1976 (added) | inexact proximal point converges when errors are summable | monotone operator | the error budget (C8) and the measured sampled error (C12) |

## 5. What would change these conclusions (how to challenge them)

* A sampled configuration of schedule-free AdamW or Adam with $\beta_1 = 0.9$ that reaches NashConv < 0.3 on the
  same harness would refute T2's practical claim (C6/C9).
* A refresh schedule of the sampled rule that improves on ~0.3 without changing the optimizer or the per-row sampling
  would weaken T1/T3 as the binding constraint.
* A demonstration that the error budget (C8) is an artefact of isotropic Gaussian noise. Real inner errors are
  structured: concentrated on round 2 and on rarely reached rows.
* The function-approximation question: whether shared parameters remove T3 (speeding up rare rows) or worsen it
  (coupling the noise) is not answered here.
