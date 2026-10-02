# The papers, their claims, and what this repository shows about each

Each section states (1) the paper's result **as used here**, a paraphrase, so check the paper for the exact
statement; (2) its hypotheses; (3) the claims of this repository that bear on it; (4) the relation, as one of:

* **CONFIRMED**: the paper's prediction (or prescription) reproduces on 2-player Leduc with exact NashConv.
* **BOUNDARY**: the paper's guarantee does not carry over when a named hypothesis is dropped. This refutes an
  *extrapolation* of the paper (an assumption common in practice), **not the paper**. The papers do not claim these
  cases.
* **NOT CONFIRMED HERE**: a prescription was tested in a specific form and did not deliver the predicted effect.
  This is evidence about that form on this game, not a refutation of the paper's theorem.
* **UNTESTED**: still not run.

No experiment here contradicts a theorem within its hypotheses.

## Summary matrix

| paper | CONFIRMED | BOUNDARY (extrapolation refuted) | NOT CONFIRMED HERE / UNTESTED |
|---|---|---|---|
| Perolat et al. 2021 | C2, C3 | C8, C12 (fixed point not reached before the magnet moves) | continuous-time dynamics (untested) |
| Perolat et al. 2022 (DeepNash) | C3 (support safeguard), C5, C19 (recipe with Adam b1 = 0 works: 0.41) | C9, C16, C19 (schedule-free inside the recipe: 2.30) | — |
| Sokota et al. 2023 (MMD) | C2, C3, C4 | C6, C7, C13, C14 (non-mirror optimisers) | C20 annealing inconclusive (0.251 vs 0.25) |
| Abe et al. 2024 (APMD) | **C18: the noisy-feedback prescription lowers the sampled floor 0.38 → 0.22** | C10, C12 (constant step, short interval) | — |
| Daskalakis & Panageas 2018 | C6, C7, C22 consistent (simultaneous dynamics need not converge) | — | C17: gradient-level optimism gives only 0.38 → 0.32 |
| Lee, Kroer & Luo 2021 | — | — | C17: optimism, gradient-level form, not confirmed; dilated optimistic methods untested |
| Cen et al. 2023 | C2, C4 | — | C17 (optimism) not confirmed in this form |
| Gidel et al. 2019 (VI perspective) | reach-weighted average < last iterate in every sampled arm | averaging the magnet does not fix T3 | C17: plain extragradient does not stabilise softmax-PG SGD at lr 0.3 |
| Gidel et al. 2019 (negative momentum, added) | C7, C15, C22 (positive momentum degrades game dynamics) | — | negative momentum untested |
| Daskalakis et al. 2018 (Optimistic Adam, added) | — | — | C17(b): optimism does not rescue schedule-free |
| Defazio et al. 2024 (schedule-free, added) | C22 (converges at ω = 0, its designed regime) | **C13, C14, C16, C19, C22, C23: fails in rotation-dominated game dynamics, via its interpolation β; its averaged iterate fails too** | — |
| Feng, Ou & Wang 2026 (Adam ODE in zero-sum games, added) | C7, C15, C22 consistent with momentum's role being reversed in games (abstract-level reading) | — | — |
| Rockafellar 1976 (added) | C8 | — | — |

## Perolat et al. 2021 — *From Poincaré recurrence to convergence in imperfect information games*

* **Result.** Follow-the-regularised-leader dynamics in zero-sum imperfect-information games are recurrent (they
  cycle). Reward regularisation toward a fixed policy makes them converge to the regularised fixed point, and
  iterating that fixed point converges to a Nash equilibrium.
* **Hypotheses.** Exact (continuous-time) dynamics; each regularised problem solved before the regulariser moves.
* **Here.** CONFIRMED: linear inner convergence (C2); iterating the magnet reaches 0.0006 (C3). BOUNDARY: with the
  magnet moving on a clock shorter than the inner solve, convergence is lost (C8: the error budget; C12: the
  sampled iterate carries ~0.9 log-prob error after 16k steps).

## Perolat et al. 2022 — *DeepNash / R-NaD*

* **Recipe** (read from the paper's methods): reward transformation toward $\pi_{\text{reg}}$ with $\eta = 0.2$; the
  regularisation policy is replaced every $\Delta_m$ learner steps (10k–100k), with the transformation interpolated as
  $\alpha_n = \min(1, 2n/\Delta_m)$ between the two most recent policies; NeuRD with threshold $\beta = 2$; **Adam with
  $b_1 = 0.0$, $b_2 = 0.999$**, lr 5e-5.
* **Here.**
  - CONFIRMED: the recipe works inside our sampled rule (C19: 0.41 with Adam $b_1 = 0$); the node-only transformation
    suffices (C5); a support safeguard is necessary (C3). The momentum-free optimiser is the right choice (C7, C15).
  - BOUNDARY: the same recipe with schedule-free AdamW is 5.6× worse (C19: 2.30); R-NaD realised with
    schedule-free AdamW fails, exact or sampled (C6, C9).

## Sokota et al. 2023 — *Magnetic mirror descent (MMD)*

* **Result.** MMD converges linearly in the last iterate to the regularised equilibrium (QRE) when the step is small
  relative to the regularisation; moving the magnet or annealing the temperature approaches Nash.
* **Hypotheses.** Mirror-descent step; values exact or with controlled error.
* **Here.**
  - CONFIRMED: the linear rate (C2), the QRE limit and its floor (C4), Nash with a moving magnet (C3).
  - BOUNDARY: the objective cannot be optimised by an arbitrary first-order optimiser (C6, C7, C13, C14).
  - Inconclusive: temperature annealing inside the sampled rule reaches 0.251 against a pre-stated 0.25 (C20), with
    reach-average 0.12. It is about as good as the moving magnet with the APMD schedule (0.225, C18).

## Abe et al. 2024 — *Adaptively perturbed mirror descent (APMD)*

* **Result** (read from the paper): mirror descent perturbed toward a slingshot that is re-initialised to the current
  iterate every $T_\sigma$ steps. Under noisy feedback the learning rate restarts at each slingshot update and decays
  as $\eta_t = 1/(\kappa s + 2\theta)$ ($s$ = steps since the update), with $T_\sigma = \Theta(T^{4/5})$.
* **Here.**
  - BOUNDARY: a constant step with a short interval (the baseline) plateaus at ~0.3–0.38 (C10, C12).
  - **CONFIRMED: the prescription transplanted into the sampled rule** (decay $\propto 1/(1 + s/500)$, interval 4000
    for 32k steps) **lowers the floor to 0.225 from 0.377** with the same interval and a constant step (C18). That is
    the best sampled result in this repository and direct support for the timing account (T1, T3).

## Daskalakis & Panageas 2018; Lee, Kroer & Luo 2021; Cen et al. 2023 — optimism

* **Results.** Plain simultaneous dynamics need not converge in min-max problems; optimistic methods give
  last-iterate convergence (in extensive-form games with dilated regularisers; linearly to the regularised
  equilibrium with entropy regularisation).
* **Here (C17, REFUTED as pre-stated).** Gradient-level optimism ($2g_t - g_{t-1}$) improves Adam $\beta_1 = 0.9$
  only from 0.384 to 0.320 (the claim was > 2×). Optimism does not rescue schedule-free AdamW (4.80 → 4.80).
  NOT CONFIRMED HERE, for the specific forms tested: constant steps, gradient-level optimism, not dilated optimistic
  mirror descent. The papers' own algorithms remain untested.

## Gidel et al. 2019 — *A variational inequality perspective on GANs* (and *Negative momentum*, AISTATS 2019)

* **Results.** Extrapolation and iterate averaging fix simultaneous gradient steps on monotone games; positive
  momentum hurts game dynamics, and negative momentum can help.
* **Here.** Positive momentum hurts monotonically (C7, C15, C22). The reach-weighted average beats the last
  iterate in every sampled arm. Plain extragradient did not stabilise softmax-PG SGD at lr 0.3 (C17c; that baseline
  was not unstable to begin with: 0.271). Schedule-free's own averaged iterate does not converge (C23).

## Defazio et al. 2024 — *The Road Less Scheduled* (schedule-free optimisers; added)

* **Result.** Schedule-free SGD/AdamW replace learning-rate schedules by iterate averaging: a base sequence $z$, a
  weighted average $x$ of $z$, and gradients evaluated at the interpolation $y = (1-\beta)z + \beta x$. They match
  or beat tuned schedules in convex and deep-learning **minimisation**, with guarantees for convex minimisation.
* **Hypotheses.** Minimisation.
* **Here.**
  - CONFIRMED in the designed regime: on the linearised field with $\omega = 0$ both variants converge (C22).
  - BOUNDARY, the main finding of `docs/WHITEPAPER.md`:
    - with any rotation ($\omega \ge 3$, $\lambda = 1$) both diverge at a step where SGD converges (C22);
    - on Leduc both fail on the exact objective where SGD and the mirror step converge (C13, C6);
    - the failure turns on with $\beta$ (C14);
    - it persists with sampled values (C16), inside the DeepNash recipe (C19), and, more weakly, with a
      shared-weight network (C15);
    - the averaged iterate fails as badly as the acting point (C23).

## Feng, Ou & Wang 2026 — *Understanding dynamics of Adam in zero-sum games: an ODE approach* (added)

* **Result (from the abstract).** In zero-sum games the roles of Adam's first- and second-moment parameters are the
  opposite of their roles in minimisation.
* **Here.** Consistent with C7, C15 and C22 (larger $\beta_1$ is worse in games, harmless in minimisation at
  $\omega = 0$). This reading rests on the abstract; check the paper before citing it as agreement.

## Rockafellar 1976 — *Monotone operators and the proximal point algorithm* (added)

* **Result.** The inexact proximal-point algorithm converges when the per-step errors are summable.
* **Here.** CONFIRMED: with a constant per-refresh error the magnet iteration stalls at a level that grows with the
  error (C8). The sampled rule's per-refresh error is constant and large (C12).
