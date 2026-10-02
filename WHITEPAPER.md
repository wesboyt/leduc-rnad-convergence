# Schedule-Free Shortcomings in Relation to Relative-Regret RL Algorithms

### Evidence from exactly-solvable poker, a mechanism in the linearised game, and what it implies for KL-regularised policy optimisation of large models

*"Relative-regret RL algorithms" here means updates that move a softmax policy by an action's value relative to a
baseline over its alternatives, regularised toward a reference policy: counterfactual-regret methods (CFR, NeuRD,
R-NaD, MMD) and the policy-optimisation methods with the same structure (PPO, GRPO, Nash-MD, SPPO).*

*Working paper, 2026-10-02. Every number below is produced by `reproduce.py` in this repository; claim ids (C1–C23)
refer to `docs/CLAIMS.md` and `RESULTS.md`, which record each claim's pre-stated refutation criterion and verdict,
including the claims that were refuted.*

---

## Abstract

Regularised learning dynamics are the modern route to Nash equilibria in zero-sum games: regret-based and
policy-gradient updates pulled toward a slowly moving reference policy (the "magnet": R-NaD/DeepNash, magnetic mirror
descent, APMD). The same structure has an advantage relative to a baseline, a softmax policy, a KL penalty toward a
reference, and periodic reference updates. It underlies the policy optimisation used to train large language models
(PPO, GRPO), and in particular its game-structured variants (Nash learning from human feedback, self-play preference
optimisation, self-play on zero-sum games). Their convergence theory assumes mirror or gradient steps. Practice uses
AdamW and, increasingly, **schedule-free** optimisers.

On two-player Leduc hold'em, where exploitability (NashConv) is computed exactly, we isolate the optimiser:

1. On an identical exact objective, **schedule-free SGD and schedule-free AdamW fail where plain SGD and a
   mirror step converge**: NashConv 3.0–7.2 vs 0.06 and 0.011 (C13, C6). Uniform play is 4.75.
2. The failure is caused by schedule-free's **interpolation** $y = (1-\beta)z + \beta x$, not by Adam's
   normalisation or by averaging: at $\beta \to 0$ both schedule-free variants converge (0.097, 0.073); at the
   default $\beta = 0.9$ they do not (3.03, 4.80) (C14).
3. The averaged iterate $x$, which schedule-free's guarantee is about, fails as badly as the acting point (C23,
   refuting our own hypothesis that it would converge).
4. Momentum alone degrades game dynamics monotonically (C7). DeepNash, the one large-scale success of this family,
   used Adam with $\beta_1 = 0$.
5. The failure persists:
   - with sampled values (schedule-free SGD 2.8–4.0 vs SGD 0.33–0.37, C16);
   - inside the DeepNash recipe (2.30 vs 0.41 with Adam $\beta_1 = 0$, C19);
   - more weakly, with a shared-weight network (1.92 vs 0.91, worse on all three seeds, C15).
6. In the linearised regularised game $F(w) = (\lambda I + \omega J)w$, both schedule-free variants converge in pure
   minimisation ($\omega = 0$) and **diverge as soon as rotation is present** ($\omega \ge 3\lambda$), at a step where
   plain SGD converges for every $\omega$ (C22). This isolates the mechanism: schedule-free's interpolation is
   stable in gradient fields and unstable in rotational ones.
7. The timing account is confirmed constructively: APMD's noisy-feedback prescription (learning rate restarted at
   every magnet update and decayed, long magnet interval) lowers the sampled rule's floor from 0.38 to 0.22 (C18).
   Optimism and extragradient in the forms tested did not help (C17, refuted).

Schedule-free methods were designed and proven for minimisation. We argue that the property that makes them work
there (acting at an interpolation toward the long-run average) is what breaks a rotation-dominated field. The
practical implication is concrete and testable: **game-structured large-model training should not assume that
optimiser choices validated in minimisation (pretraining, single-agent RLHF) carry over.**

---

## 1. Two literatures, one update

**Regret side.** Counterfactual regret minimisation accumulates instantaneous regrets
$r_t(s,a) = \mathrm{cfv}(s,a) - \sum_b \pi_t(b|s)\,\mathrm{cfv}(s,b)$ and plays regret-matching or Hedge policies.
Neural Replicator Dynamics (NeuRD; Hennes et al. 2020) is a one-line change to policy gradient: it bypasses the
softmax Jacobian and adds the advantage directly to the logits. It reduces to Hedge in the single-state case and is
equivalent to softmax CFR in the tabular sequential case. R-NaD (Perolat et al. 2021, 2022) adds a reward
transformation toward a magnet $\rho$ and moves $\rho$ only after the inner dynamics reach their fixed point. MMD
(Sokota et al. 2023) writes the same idea as a proximal mirror step with linear last-iterate convergence to the
regularised equilibrium.

**Policy-optimisation side.** PPO maximises a clipped importance-weighted advantage with a KL penalty or trust region.
GRPO (DeepSeekMath, 2024) removes the critic: for each prompt it samples a group of $G$ completions and uses the
**group-relative advantage** $A_i = (r_i - \mathrm{mean}(r))/\mathrm{std}(r)$, with a KL term toward a reference
policy in the loss. Game-structured variants make the analogy exact:
- Nash-MD (Munos et al., ICML 2024) runs mirror descent against a geometric mixture of the current and reference
  policy and proves last-iterate convergence to the *regularised* Nash equilibrium of a preference game.
- SPPO (Wu et al. 2024) runs multiplicative weights in self-play rounds against the previous round's policy.
- Self-play on zero-sum language games (e.g. SPIRAL, 2025) trains on genuinely adversarial objectives.

**The correspondence** (our sampled rule, `docs/ALGORITHM.md` §5, against GRPO-style training):

| regret / poker side | policy-optimisation / LLM side |
|---|---|
| information set $s$ (public history + private card) | prompt / context |
| legal actions $A(s)$, all evaluated at the pivot | the $G$ sampled completions of a group |
| $Q(s,a)$ by Monte-Carlo rollouts | completion reward (outcome or process) |
| instantaneous regret $Q(s,a) - \text{baseline}$ | advantage $r_i - \text{baseline}$ |
| sampled rule: $(\hat q_a - \text{mean}_{A(s)}\hat q)/\sigma_{\text{round}}$ | GRPO: $(r_i - \text{mean}_G r)/\text{std}_G r$ |
| NeuRD / Hedge: advantage added to logits | policy gradient on the token logits |
| magnet $\rho$, strength $\eta_{\text{reg}}$ | reference policy $\pi_{\text{ref}}$, KL coefficient $\beta$ |
| magnet refresh (R-NaD outer loop) | reference update: iterative/online DPO rounds, SPPO rounds, Nash-MD's mixture |
| PPO clip (inactive at ratio 1, §5.1) | PPO / GRPO clip |
| optimiser on $\theta$ | AdamW, schedule-free AdamW |

**Where the analogy holds and where it does not.**
- Single-agent RLHF against a *fixed* reward model is a (non-stationary) minimisation: its vector field has no
  rotational part.
- Rotation enters whenever the target depends on another learner: self-play, preference games (NLHF, SPPO),
  adversarial co-training of a policy and its reward model, debate and red-teaming. Our results speak to that class.
- Group sampling (many completions per prompt) and token-level credit assignment change the noise structure; scale
  (billions of weights) changes the curvature. Neither is tested here.

## 2. Setting and instruments

Two-player Leduc hold'em: 288 information sets, 672 action entries, value −0.0856. NashConv is exact (all 120 deals
enumerated; `lrc/game.py`), validated against an independent native implementation to 1e-14 (`docs/VALIDATION.md`)
and by 15 audits with negative controls (C1). We study three learners (`docs/ALGORITHM.md`):

* the **reference**: closed-form magnetic mirror descent with exact values and a magnet refreshed every $K$ steps;
* **exact-gradient learners**: the same objective with logits moved by an optimiser, either as a table (adaptive
  optimisers normalise per information-set action) or as a **shared-weight network** (normalisation per weight, the
  large-model regime);
* the **sampled rule**: on-policy pivots, Monte-Carlo action values, importance weights, sigma-normalised advantages,
  anchors, clipping and a uniform PPO surrogate whose logit gradient is NeuRD (§5.1 of ALGORITHM.md).

## 3. The baseline facts

* The game converges under a correct update: the reference reaches NashConv 0.0006 (C3).
* Fixed pulls toward uniform impose a QRE floor (0.1087, C4).
* The refresh clock must be slower than the inner solve: the outer loop tolerates ~0.01–0.03 log-probability of
  magnet error per refresh (C8). The sampled inner loop carries ~0.9 after 16k steps, where the reference needs
  1000 (C12). This is the timing account of `docs/TIMING.md`.

## 4. Schedule-free fails in game dynamics

### 4.1 The optimiser decides convergence on an identical objective (C6, C7, C13)

Same exact values, same node-only magnet at the baseline's strength, same refresh, same support floor; 30k steps;
mean NashConv of the last 5 evaluations:

| update on the logits (table) | NashConv |
|---|---|
| mirror step (reference) | 0.011 |
| NeuRD, SGD | 0.019 |
| softmax PG, SGD (lr 0.1) | 0.062 |
| softmax PG, Adam $\beta_1 = 0$ | 0.044 |
| softmax PG, Adam $\beta_1 = 0.5$ | 0.073 |
| softmax PG, Adam $\beta_1 = 0.9$ | 0.38 |
| softmax PG, **schedule-free SGD** (lr 0.1 / 0.03) | **3.03 / 7.19** |
| softmax PG, **schedule-free AdamW** (lr 0.003 / 0.01 / 0.03) | **4.71 / 4.80 / 5.91** |

Schedule-free SGD fails without any adaptive normalisation (C13).

### 4.2 The interpolation is the cause (C14)

Schedule-free maintains a base sequence $z$, a weighted average $x$ of $z$, and evaluates the gradient at
$y = (1-\beta) z + \beta x$. At $\beta \to 0$ it takes base-optimiser steps at $z$:

| $\beta$ | schedule-free SGD | schedule-free AdamW |
|---|---|---|
| ≈ 0 | 0.097 | 0.073 |
| 0.5 | 3.11 | 0.10 |
| 0.9 | 3.03 | 4.80 |

### 4.3 The averaged iterate does not rescue it (C23, a refuted hypothesis)

We pre-registered that the averaged iterate $x$ (eval mode) would converge, which would have made the failure a
deployment detail. It does not: $x$ scores 3.034 and 4.8007 against $y$'s 3.034 and 4.8008. Exploratory traces
(not pre-registered) show two failure modes, neither of them an orbit around the equilibrium:
- with the softmax gradient, the iterate wanders and then locks into a saturated, nearly pure policy (41% of
  entries > 0.99; NashConv frozen at 3.036 from step 10k on);
- with NeuRD, NashConv drifts upward through 12k steps (1.7 → 5.3; 1.3 → 6.2).

In both, $x$ tracks $y$: the long-run average of the base sequence is itself displaced from equilibrium.

## 5. Generality: network policy, sampled rule, the papers' prescriptions, and the linear mechanism

### 5.1 Sampled values (C16, C19)

The sampled rule (Monte-Carlo action values, on-policy pivots, importance weights, sigma-normalised group-relative
advantages; the closest analogue here of GRPO-style training), 2 seeds:

| optimiser in the sampled rule | NashConv |
|---|---|
| SGD (C9) | 0.37 / 0.33 |
| schedule-free SGD, same lr (C16) | 3.98 / 2.81 |
| schedule-free AdamW (C9) | 3.34 / 4.39 |
| DeepNash recipe, Adam $\beta_1 = 0$, 32k (C19) | 0.39 / 0.43 |
| DeepNash recipe, schedule-free AdamW, 32k (C19) | 2.27 / 2.34 |

### 5.2 A shared-weight network (C15)

When one 7k-parameter network produces the logits of all 288 information sets, adaptive optimisers normalise per
weight, as in large-model training. Exact values, NeuRD with DeepNash's threshold, magnet every 5000 steps, 30k
steps, mean of the last 10k, 3 seeds:

| optimiser (network) | mean NashConv | per seed |
|---|---|---|
| Adam $\beta_1 = 0$ | 0.91 | 0.77 / 1.04 / 0.91 |
| schedule-free AdamW, $\beta = 0$ | 1.13 | 0.82 / 1.63 / 0.95 |
| Adam $\beta_1 = 0.9$ | 1.48 | 1.25 / 1.69 / 1.49 |
| schedule-free AdamW, $\beta = 0.9$ | 1.92 | 3.11 / 1.25 / 1.41 |

The ordering matches the table setting, and schedule-free is worse than momentum-free Adam on every seed. But the
effect is smaller (1.2–4× rather than ~50×), the pre-stated bound (2×) is met narrowly, and no optimiser converges
well with this network (§8). Function approximation makes the inner problem much harder for all optimisers.

### 5.3 The papers' prescriptions (C17, C18, C20)

* **APMD** (Abe et al. 2024): the noisy-feedback schedule lowers the sampled floor to **0.225** from 0.377 at the
  same magnet interval (C18). This is the best sampled result here, and it is a prescription about timing.
* **Optimism / extragradient** (Daskalakis & Panageas; Lee et al.; Cen et al.; Gidel et al.), in the forms tested
  (gradient-level optimism, plain extragradient, constant steps): Adam $\beta_1 = 0.9$ improves only from 0.384 to
  0.320, extragradient does not help SGD, and **optimism does not rescue schedule-free** (4.80 → 4.80). C17 was
  refuted as stated.
* **Temperature annealing** (Sokota et al.) instead of a moving magnet: 0.251 against a pre-stated 0.25 (C20, a
  narrow refutation), with reach-average 0.12.

### 5.4 The mechanism in the linearised game (C22; C21 refuted by a design error)

$F(w) = (\lambda I + \omega J)w$ with $\lambda = 1$, 16 dimensions, 40k steps, step $\gamma$ = half of SGD's stability
bound at the largest $\omega$. Entries are $\|w_T\|/\|w_0\|$:

| optimiser | ω = 0 (minimisation) | ω = 3 | ω = 10 | ω = 20 |
|---|---|---|---|---|
| SGD | 4e-44 | 1e-43 | 1e-38 | 2e-22 |
| Adam $\beta_1 = 0$ | 2e-4 | 3e-3 | 0.011 | 0.021 |
| Adam $\beta_1 = 0.9$ | 9e-5 | 0.057 | 0.18 | 0.30 |
| **schedule-free SGD** ($\beta = 0.9$) | 6e-5 | **360** | **1.5e7** | **1.1e10** |
| **schedule-free AdamW** ($\beta = 0.9$) | 1e-4 | **6.4** | **12.1** | **13.4** |

Schedule-free converges where it was designed to ($\omega = 0$) and diverges as soon as the field has a rotational
component three times its contraction. Momentum (Adam $\beta_1 = 0.9$) degrades gracefully with $\omega$; schedule-free
does not. The first version of this test (C21) used a step above SGD's own bound at $\omega = 30$ and was refuted on
that precondition; C22 was registered after seeing it.

## 6. Mechanism

Near an equilibrium of a regularised zero-sum game the simultaneous-gradient field is, to first order,
$F(w) = (\lambda I + \omega J)w$: $\lambda$ is the contraction supplied by the regulariser (the magnet strength) and
$\omega$ the rotation supplied by the game. Minimisation is $\omega = 0$. Plain gradient steps converge iff
$\gamma < 2\lambda/(\lambda^2 + \omega^2)$: rotation shrinks the stable step but never removes it.

Schedule-free evaluates $F$ at $y$, a point pulled toward the running average $x$ of past base iterates. In
minimisation the average lies along the descent path, so evaluating there is the online-to-batch device behind
schedule-free's guarantee. In a rotational field the average lags the iterate by a phase. A force evaluated at a
lagged position acquires a tangential component that does not point back to the equilibrium. This is the same family
of instability as positive momentum (Gidel et al., AISTATS 2019) and delayed feedback, which Adam with
$\beta_1 = 0.9$ also exhibits more mildly (C7). The regulariser's contraction $\lambda$ competes with it, so the
failure appears when rotation dominates: $\omega \gg \lambda$, i.e. weak regularisation relative to the strategic
interaction. §5.4 quantifies it: with $\lambda = 1$ the boundary lies below $\omega = 3$ for both schedule-free variants
at a step where SGD tolerates $\omega = 20$.

Two observations sharpen the picture.
- **The averaged iterate is no refuge (C23).** In a rotational field the base sequence $z$ is driven by forces
  evaluated at lagged points, so it is not a no-regret sequence. Its average then has no reason to approach the
  equilibrium, unlike the average of plain no-regret dynamics.
- **The failure is a threshold in $\beta$ (C14).** At $\beta = 0.5$ schedule-free AdamW still converges (0.10) while
  schedule-free SGD does not (3.11). Adam's per-coordinate normalisation shrinks the effective step and moves the
  threshold, but at the default $\beta = 0.9$ both fail.

## 7. Implications for large-model training (hypotheses, not results)

* **H1.** In game-structured LLM training (self-play on zero-sum tasks, NLHF/Nash-MD, SPPO-style rounds, policy and
  reward-model co-training), last-iterate quality is worse with schedule-free AdamW than with AdamW at
  $\beta_1 \in \{0, 0.5\}$, and the gap grows as the KL coefficient falls (smaller $\lambda$).
* **H2.** Single-agent RLHF against a fixed reward model ($\omega \approx 0$) shows no such effect. Schedule-free's
  benefits observed in pretraining are expected to transfer there.
* **H3.** The reference/magnet update interval (iterative-DPO rounds, SPPO rounds) must be slower than the inner
  convergence of the policy to the regularised target. Refreshing faster locks in error (the timing account, C8/C12).

How to test: run the same game-structured objective with each optimiser and measure exploitability where it is
computable (small games, synthetic preference games with intransitive preferences), or head-to-head win rates
against fixed checkpoints at matched steps. Sweep the KL coefficient to vary $\lambda$, and sweep schedule-free's
$\beta$ (the dose-response of C14).

## 8. Conclusions

1. **Relative-regret dynamics converge when realised as mirror or gradient steps on a slow magnet clock**
   (reference 0.0006; SGD within ~3× of it). This reproduces Perolat et al. and Sokota et al.
2. **Schedule-free optimisation is a poor fit for these dynamics.** It fails on the exact objective (C13, C6),
   under sampling (C16), inside the DeepNash recipe (C19), and in the linearised game as soon as rotation appears
   (C22). The cause is its interpolation $\beta$ (C14), and its averaged iterate does not escape (C23). With a
   shared network the effect remains but is much smaller (C15).
3. **Momentum is a smaller version of the same problem** (C7, C15, C22). DeepNash's choice of Adam with
   $\beta_1 = 0$ is consistent with this, as is the recent ODE analysis of Adam in zero-sum games.
4. **Timing matters as much as the optimiser.** The magnet may move only when the inner problem is solved to
   ~0.01–0.03 log-probability (C8). A sampled per-information-set rule on a short clock is ~30× over that budget
   (C12). APMD's restart-and-decay schedule with a long interval recovers part of the gap (C18).
5. **For large-model training (hypotheses H1–H3, §7):** where the objective is game-structured (self-play,
   preference games, adversarial co-training), schedule-free and high-momentum optimisers should be treated as
   risks to last-iterate quality, and the reference-update interval as a convergence-critical hyperparameter.
   Where it is not (single-agent RLHF against a fixed reward model), the mechanism predicts no harm.

## 9. Limitations

* One game (2-player Leduc) and its linearisation; a 7k-parameter network (no optimiser converges well with it);
  CPU-scale runs; 2-3 seeds for stochastic arms. Nothing here is measured
  on a large model.
* The sampled rule has no function-approximation generalisation in its tabular form. The network results use exact
  values.
* "Fails" means fails at the tested learning rates and horizons. The criteria and ranges are in `docs/CLAIMS.md`.
* Four of our own pre-registered claims were refuted (C17, C20 narrowly, C21 by a design error, C23). They are kept in
  the record.

## 10. Related work

Perolat et al. 2021 (regularisation turns recurrent dynamics convergent); Perolat et al. 2022 (DeepNash: R-NaD at
scale, Adam with $b_1 = 0$, magnet interpolation, NeuRD threshold); Sokota et al. 2023 (MMD); Abe et al. 2024 (APMD);
Daskalakis & Panageas 2018; Daskalakis et al. 2018 (Optimistic Adam); Gidel et al. 2019 (VI perspective;
negative momentum); Lee, Kroer & Luo 2021; Cen et al. 2023; Hennes et al. 2020 (NeuRD); Defazio et al. 2024
(schedule-free); Feng, Ou & Wang 2026 (an ODE analysis of Adam in zero-sum games, finding momentum's role reversed
relative to minimisation); Munos et al. 2024 (NLHF / Nash-MD); Wu et al. 2024 (SPPO); Shao et al. 2024 (GRPO); Schulman
et al. 2017 (PPO); "Steering Equilibrium Selection in Regularized Self-Play" (arXiv 2609.19820, the reference policy as
an equilibrium-selection lever). Full references with links: `README.md` and `docs/PAPERS.md`.

---

*This paper and the documentation in `docs/` are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the code under the MIT licence (`LICENSE`).*
