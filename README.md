# Schedule-Free Shortcomings in Relation to Relative-Regret RL Algorithms

**Read the paper: [WHITEPAPER.md](WHITEPAPER.md)** · Code and every experiment: this repository

## Abstract

Regularised learning dynamics are the standard route to Nash equilibria in two-player zero-sum games: regret and policy-gradient updates pulled toward a slowly moving reference policy (R-NaD, magnetic mirror descent, APMD). The same structure (an action's value relative to a baseline over its alternatives, a softmax policy, a KL penalty toward a reference, periodic reference updates) underlies PPO and GRPO and the game-structured variants used to train large language models, such as Nash learning from human feedback and self-play preference optimisation. Convergence theory assumes mirror or gradient steps; practice uses AdamW and, increasingly, schedule-free optimisers. On two-player Leduc hold'em, where exploitability is computed exactly, we find that schedule-free SGD and schedule-free AdamW fail on an identical objective where plain SGD and a mirror step converge (NashConv 3.0-7.2 versus 0.06 and 0.011); that the failure is switched on by schedule-free's interpolation parameter (beta near 0 converges, the default 0.9 does not); that the averaged iterate fails as badly as the acting point; and that the effect persists with sampled values, inside the DeepNash recipe, and, more weakly, with a shared-weight network. In the linearised regularised game both schedule-free variants converge in pure minimisation and diverge once the rotational part exceeds three times the regularisation, at step sizes where SGD is stable. We also show that the timing of reference updates is convergence-critical, and that APMD's noisy-feedback schedule lowers a sampled rule's exploitability floor by 40%. All 23 claims were registered with refutation criteria before being run; four were refuted and are reported.

---

## The repository

This repository studies one sampled, regularised policy-gradient rule of the R-NaD family. Each piece is measured
against an exact ruler (NashConv over all 120 deals of 2-player Leduc hold'em) and an exact reference solver that
provably converges.

Everything is self-contained: the game, the exact evaluator, the reference solver, the sampled rule, the audits, the
experiment grids and a claim-by-claim reproduction script with pre-stated refutation criteria. Dependencies: numpy,
torch (CPU) and schedulefree.

## What this is, and what it is not

* It **is** a set of reproducible measurements on a small exactly-solvable game:
  - the reference converges to NashConv 0.0006;
  - the sampled rule plateaus at ~0.3 with SGD, and at 3–4 with its default optimizer;
  - each mechanism behind the gap is isolated by an experiment that can refute it.
* It is **not** a new theorem. Every mechanism found lies outside a hypothesis of a known convergence result: the
  optimizer is not a mirror/gradient step, there is a fixed regulariser toward uniform, the magnet moves before the
  inner problem is solved, and probabilities lose support. The quantitative part is new and specific to this rule:
  - **the error budget**: the outer (magnet) loop tolerates ~0.01–0.03 log-probability of error per refresh;
  - **the inner-loop time** of the sampled step, which carries ~1 log-probability of error after 16k steps, where
    the exact reference needs < 2000.

  See `docs/TIMING.md` for the argument and its relation to the literature.
* Against each paper the claims are classified in `docs/PAPERS.md` as CONFIRMED (its prediction reproduces), BOUNDARY
  (its guarantee does not survive dropping a named hypothesis; this refutes a common extrapolation, not the paper) or
  UNTESTED (a prescription not yet run, e.g. APMD's step-size schedule or optimistic updates inside the sampled rule).

## Quick start

```bash
pip install -r requirements.txt
python -m lrc.reference --audit          # exact evaluator vs an independent best response
python experiments/audit_harness.py      # 15 checks of the sampled harness, each with a negative control
python tests/test_core.py                # rules-level tests (or: python -m pytest tests)
python reproduce.py --list               # the claims
python reproduce.py C2 C3 C4             # run some (SUPPORTED / REFUTED + numbers, results/claims/*.json)
python reproduce.py all --jobs 8         # everything, ~3 h on 8 cores
```

On a free-threaded CPython build set `PYTHON_GIL=0`. Single runs:

```bash
python -m lrc.reference --steps 60000 --eta 0.03 --alpha 0.2 --magnet refresh --K 500 --zfloor 40 --score-every 5000
python -m lrc.harness --steps 3000 --out results/harness/base          # the sampled rule at its baseline
python experiments/ladder.py --grid p4 --steps 16000 --seeds 1,2 --jobs 4 && python experiments/ladder.py --grid p4 --summary
```

## Repository map

| path | content |
|---|---|
| `lrc/game.py` | Kuhn and Leduc (2 players): tree, deals, policy-table layout, exact EV / best response / NashConv, on-policy generation with reservoir-of-one pivots, Monte-Carlo and exact action values, the reach-weighted average accumulator |
| `lrc/reference.py` | full-width exact values (counterfactual values, soft/regularised values), behavioural MMD with fixed / refreshed / gated / geometric magnets, support floor, refresh noise; exact-gradient learners (softmax PG, NeuRD) with SGD / Adam / schedule-free AdamW |
| `lrc/harness.py` | the sampled rule (`docs/ALGORITHM.md` §5), every component switchable |
| `lrc/weights.py` | importance weights, advantage centring, the uniform PPO surrogate |
| `lrc/policy.py` | table and shared-weight MLP policies; SGD / Adam / schedule-free AdamW / schedule-free SGD; gradient-level optimism |
| `experiments/` | `ladder.py` (sampled-rule grids), `mmd_grid.py` (reference grids), `sf_linear.py` (optimisers on the linearised regularised game), `fixedpoint_probe.py`, `refresh_error.py`, `audit_harness.py`, `mmd_summary.py` |
| `reproduce.py` | claims C1–C23 with pre-stated refutation criteria |
| `WHITEPAPER.md` | the paper: schedule-free shortcomings in relative-regret RL algorithms, the mechanism, and the connection to PPO / GRPO / Nash-MD / SPPO |
| `docs/ALGORITHM.md` | the game, the regularised game, the reference, the gradient learners and the sampled rule in notation |
| `docs/TIMING.md` | the timing issues T1–T5 and their relation to the literature |
| `docs/CLAIMS.md` | claims, original evidence, criteria, revision log, rejected hypotheses |
| `docs/PAPERS.md` | each paper's result and hypotheses; which claims CONFIRM it, which establish a BOUNDARY (an extrapolation that fails), and what it prescribes that is still UNTESTED |
| `docs/VALIDATION.md` | equivalence with the original native implementation; internal audits |
| `RESULTS.md` | the output of `reproduce.py all` on this code (regenerate with `python make_results.py`) |
| `validation/compare_native.py` | provenance only (needs a native library that is not part of this repository) |

## The claims in one table

| id | claim |
|---|---|
| C1 | the instruments are right (audits with negative controls) |
| C2 | exact MMD with a fixed magnet converges linearly to the QRE |
| C3 | MMD + magnet refresh + a support floor reaches NashConv < 0.01; without the floor support is lost |
| C4 | fixed pulls toward uniform impose a QRE floor (NashConv ≈ 0.109 at the baseline's temperatures) |
| C5 | a node-only anchor converges as well as trajectory regularisation |
| C6 | on an identical exact objective the optimizer decides convergence (schedule-free AdamW fails) |
| C7 | momentum is a mechanism (β1 = 0.9 ≫ β1 ∈ {0, 0.5}) |
| C8 | the outer loop's error budget is ~0.01–0.03 log-prob per refresh |
| C9 | in the sampled rule schedule-free AdamW degrades (> 2), SGD stays < 0.6 |
| C10 | the sampled floor is not noise in Q |
| C11 | the sampled rule's expected step has the right fixed point (clips off) |
| C12 | the sampled inner loop has not solved its game after 16k steps; the exact reference takes < 2000 |
| C13 | schedule-free SGD fails where SGD converges (the failure is not Adam's) |
| C14 | dose-response: schedule-free's interpolation β switches the failure on |
| C15 | with a shared-weight network: same ordering, smaller effect (narrow) |
| C16 | schedule-free SGD fails in the sampled rule |
| C17 | **refuted**: optimism / extragradient (forms tested) do not help much; optimism does not rescue schedule-free |
| C18 | APMD's noisy-feedback prescription lowers the sampled floor 0.38 → 0.22 |
| C19 | the DeepNash recipe works with Adam β1 = 0 (0.41) and not with schedule-free (2.30) |
| C20 | **refuted (narrowly)**: temperature annealing 0.251 vs 0.25 |
| C21 | **refuted (design error)**: linear mechanism test with a step above SGD's bound |
| C22 | linear mechanism: schedule-free converges in minimisation, diverges with rotation ω ≥ 3λ |
| C23 | **refuted**: schedule-free's averaged iterate fails as badly as its acting point |

19 of 23 supported; the four refutations are kept and explained in `docs/CLAIMS.md`.

## How to challenge the findings

1. Run `python reproduce.py all`. A REFUTED verdict on your machine is a result: please report it with the json in
   `results/claims/`.
2. Break a criterion on purpose: every knob of the sampled rule is in `lrc/harness.py:defaults`. A configuration that
   gets the sampled rule (schedule-free AdamW, or SGD with the baseline's sampling) below NashConv 0.1 contradicts the
   timing account (T1/T3).
3. Attack the instruments: `experiments/audit_harness.py` and `python -m lrc.reference --audit` include negative
   controls. Add one that should fail and does not.
4. Read `docs/TIMING.md` §5 for the specific results that would change the conclusions.

## Literature

* J. Perolat et al., *From Poincaré Recurrence to Convergence in Imperfect Information Games: Finding Equilibrium via
  Regularization*, ICML 2021. https://proceedings.mlr.press/v139/perolat21a.html
* J. Perolat et al., *Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning* (DeepNash,
  R-NaD), Science 2022. https://arxiv.org/abs/2206.15378
* S. Sokota et al., *A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum
  Games* (MMD), ICLR 2023. https://arxiv.org/abs/2206.05825
* K. Abe et al., *Adaptively Perturbed Mirror Descent for Learning in Games* (APMD), ICML 2024.
  https://arxiv.org/abs/2305.16610
* C. Daskalakis, I. Panageas, *The Limit Points of (Optimistic) Gradient Descent in Min-Max Optimization*, NeurIPS
  2018. https://arxiv.org/abs/1807.03907
* C.-W. Lee, C. Kroer, H. Luo, *Last-iterate Convergence in Extensive-Form Games*, NeurIPS 2021.
  https://mlanthology.org/neurips/2021/lee2021neurips-lastiterate
* S. Cen et al., *Faster Last-iterate Convergence of Policy Optimization in Zero-Sum Markov Games*, ICLR 2023.
  https://arxiv.org/abs/2210.01050
* G. Gidel et al., *A Variational Inequality Perspective on Generative Adversarial Networks*, ICLR 2019.
  https://arxiv.org/abs/1802.10551
* Y. Feng, W. Ou, X. Wang, *Understanding Dynamics of Adam in Zero-Sum Games: An ODE Approach*, 2026.
  https://arxiv.org/abs/2605.19392
* R. Munos et al., *Nash Learning from Human Feedback*, ICML 2024. https://arxiv.org/abs/2312.00886
* Y. Wu et al., *Self-Play Preference Optimization for Language Model Alignment*, 2024. https://arxiv.org/abs/2405.00675
* Z. Shao et al., *DeepSeekMath* (GRPO), 2024. https://arxiv.org/abs/2402.03300
* J. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. https://arxiv.org/abs/1707.06347
* D. Hennes et al., *Neural Replicator Dynamics*, AAMAS 2020. https://arxiv.org/abs/1906.00190
* C. Daskalakis, A. Ilyas, V. Syrgkanis, H. Zeng, *Training GANs with Optimism*, ICLR 2018. https://arxiv.org/abs/1711.00141
* Added here: G. Gidel et al., *Negative Momentum for Improved Game Dynamics*, AISTATS 2019
  (https://arxiv.org/abs/1807.04740); R. T. Rockafellar, *Monotone Operators and the Proximal Point Algorithm*, SIAM J.
  Control Optim. 1976; A. Defazio et al., *The Road Less Scheduled* (schedule-free optimizers), 2024
  (https://arxiv.org/abs/2405.15682); F. Southey et al., *Bayes' Bluff: Opponent Modelling in Poker*, UAI 2005 (Leduc).

## Licence

Code: [MIT](LICENSE). The paper (`WHITEPAPER.md`) and the documentation in `docs/`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
