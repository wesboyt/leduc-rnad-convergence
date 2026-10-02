# The papers, their claims, and what this repository shows about each

Each section states (1) the paper's result **as used here**, a paraphrase, so check the paper for the exact
statement; (2) its hypotheses; (3) the claims of this repository that bear on it; (4) the relation, as one of:

* **CONFIRMED**: the paper's prediction reproduces on 2-player Leduc with exact NashConv.
* **BOUNDARY**: the paper's guarantee does not carry over when a named hypothesis is dropped. This refutes an
  *extrapolation* of the paper (an assumption common in practice), **not the paper**. The papers do not claim these
  cases.
* **UNTESTED**: the paper prescribes something this repository has not run yet. Listed so that it is not mistaken for
  evidence.

No experiment here contradicts a theorem within its hypotheses. A claim that one did would be a bug in the instrument
or in the reading of the paper, and C1 (the audits) and this document exist to catch both.

## Summary matrix

| paper | CONFIRMED | BOUNDARY (extrapolation refuted) | UNTESTED |
|---|---|---|---|
| Perolat et al. 2021 | C2, C3 | C12 + C8 (fixed point not reached before the magnet moves) | continuous-time dynamics |
| Perolat et al. 2022 (DeepNash / R-NaD) | C3 (support safeguard needed), C5 | C9 (optimizer), C12 (refresh clock) | magnet interpolation; NeuRD thresholding in the sampled rule |
| Sokota et al. 2023 (MMD) | C2 (linear rate), C3, C4 (QRE floor) | C6, C7 (non-mirror optimizer) | annealed temperature in the sampled rule |
| Abe et al. 2024 (APMD) | — | C10, C12 (constant step, heterogeneous per-information-set noise) | APMD's step-size and slingshot-interval prescription inside the sampled rule |
| Daskalakis & Panageas 2018 | C6, C7 consistent (plain/accelerated simultaneous dynamics need not converge) | — | optimistic GDA |
| Lee, Kroer & Luo 2021 | — | — | optimistic dilated methods (OMWU / OGDA) |
| Cen et al. 2023 | C2 (linear rate to the regularised equilibrium), C4 (regularisation bias) | — | optimism |
| Gidel et al. 2019 (VI perspective) | reach-weighted average < last iterate in every sampled arm | averaging the MAGNET does not fix T3 | extragradient |
| Rockafellar 1976 (added) | C8 (non-summable refresh error stalls the outer loop) | — | — |

## Perolat et al. 2021 — *From Poincaré recurrence to convergence in imperfect information games*

* **Result.** Follow-the-regularised-leader dynamics in zero-sum imperfect-information games are recurrent (they
  cycle). Adding a reward regularisation toward a fixed policy makes them converge to the fixed point of the
  regularised game, and iterating that fixed point (magnet ← fixed point) converges to a Nash equilibrium.
* **Hypotheses.** Exact (continuous-time) dynamics; each regularised problem solved before the regulariser moves.
* **Here.**
  - CONFIRMED: with exact values and a mirror step the regularised problem is solved linearly (C2), and iterating
    the magnet reaches NashConv 0.0006 (C3).
  - BOUNDARY: when the magnet moves on a clock shorter than the inner solve, convergence is lost. The sampled rule's
    iterate is far from the fixed point at every refresh (round-2 RMS log-prob error ~1, C12), and the outer loop
    tolerates only ~0.01–0.03 (C8). The extrapolation refuted: "refreshing the regulariser on a fixed short period is
    enough".

## Perolat et al. 2022 — *DeepNash / R-NaD*

* **Result (as a recipe).** Reward transformation toward a regularisation policy; run the learning dynamics (NeuRD)
  toward the fixed point; replace the regularisation policy; interpolate the transformation between the two most
  recent regularisation policies; threshold NeuRD's logits.
* **Hypotheses (implicit in the recipe).** Enough learning steps per iteration for the dynamics to approach the fixed
  point; an optimizer that realises the NeuRD dynamics.
* **Here.**
  - CONFIRMED: the node-only form of the transformation (raw continuation values) works as well as the trajectory form
    (C5). A support safeguard is necessary: without a floor, refreshing compounds lost support (C3; DeepNash's
    thresholding addresses the same failure).
  - BOUNDARY: realised with schedule-free AdamW, the dynamics do not converge, exact or sampled (C6, C9). With a short
    refresh clock and sampled per-information-set updates the inner problem is not solved (C12).
  - UNTESTED: magnet interpolation; logit thresholding inside the sampled rule.

## Sokota et al. 2023 — *Magnetic mirror descent (MMD)*

* **Result.** MMD converges linearly in the last iterate to the QRE (regularised equilibrium) when the step size is
  small relative to the regularisation. Moving the magnet or annealing the temperature approaches Nash.
* **Hypotheses.** Mirror-descent step; values exact or with controlled error.
* **Here.**
  - CONFIRMED: the linear rate (residual ↓ ~10× per 1000 steps, C2); the QRE as the limit with a fixed magnet,
    including the floor imposed by fixed pulls toward uniform (C4: 0.1087); approaching Nash with a moving magnet (C3).
  - BOUNDARY: replacing the mirror step with Adam or schedule-free AdamW on the same objective breaks convergence
    (C6: 4.4–6.0; C7: grows with β1). The extrapolation refuted: "the MMD objective can be optimised by any
    first-order optimizer".
  - UNTESTED: annealing the temperature inside the sampled rule (only the fixed pulls were annealed).

## Abe et al. 2024 — *Adaptively perturbed mirror descent (APMD)*

* **Result.** Mirror descent perturbed toward a slingshot (magnet) that is updated at an interval converges in the
  last iterate, also under noisy feedback.
* **Hypotheses.** Mirror-descent steps; step sizes and slingshot interval chosen for the feedback noise.
* **Here.**
  - BOUNDARY: the sampled rule (constant step, fixed interval, noise heterogeneous across information sets through
    their sampling frequency) plateaus at ~0.3 with exact or sampled Q (C10). The inner loop's slow rows and noisy rows
    cannot both be served by one step size (C12). The extrapolation refuted: "a periodically updated magnet makes a
    sampled rule robust to its noise".
  - UNTESTED: APMD's own prescription (decreasing steps, adaptive slingshot interval) inside the sampled rule. **This
    is the most direct next test of the timing account.**

## Daskalakis & Panageas 2018 — *Limit points of (optimistic) gradient descent in min-max optimization*

* **Result.** Characterises the stable limit points of gradient descent-ascent and optimistic GDA; simultaneous
  gradient dynamics need not converge to min-max points.
* **Here.** CONFIRMED in spirit: simultaneous dynamics realised with momentum / per-coordinate normalisation do not
  converge on Leduc even when regularised and given exact values (C6, C7). UNTESTED: optimistic GDA.

## Lee, Kroer & Luo 2021 — *Last-iterate convergence in extensive-form games*

* **Result.** Optimistic regret-minimisation methods with dilated regularisers converge in the last iterate in
  extensive-form games (with rates under a uniqueness assumption).
* **Here.** UNTESTED. Optimistic variants of the reference and of the sampled rule are a natural extension of
  `lrc/reference.py:grad_run`.

## Cen et al. 2023 — *Faster last-iterate convergence of policy optimization in zero-sum Markov games*

* **Result.** Entropy-regularised optimistic policy updates converge linearly in the last iterate to the regularised
  (QRE) equilibrium; the gap to Nash is controlled by the regularisation.
* **Here.** CONFIRMED: linear convergence to the regularised equilibrium (C2) and a regularisation bias floor (C4).
  UNTESTED: optimism.

## Gidel et al. 2019 — *A variational inequality perspective on GANs*

* **Result.** Viewing games as variational inequalities, simultaneous gradient steps can fail where extrapolation
  (extragradient) and iterate averaging converge.
* **Here.**
  - CONFIRMED in spirit: in every sampled arm the reach-weighted average strategy is better than the last iterate
    (e.g. SGD: 0.18 vs 0.33).
  - BOUNDARY: averaging the MAGNET (`ladder.py --grid avg`) does not remove the sampled floor.
  - UNTESTED: extragradient.

## Rockafellar 1976 — *Monotone operators and the proximal point algorithm* (added)

* **Result.** The inexact proximal-point algorithm converges when the per-step errors are summable.
* **Here.** CONFIRMED: with a constant per-refresh error the magnet iteration stalls at a level that grows with the
  error (C8). The sampled rule's per-refresh error is constant and large (C12).
