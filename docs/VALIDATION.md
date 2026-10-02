# Validation

## 1. The pure-numpy game equals the original native implementation

The experiments were first run on a native (C++) validation-game library that is part of a larger solver and is not
included here. `lrc/game.py` reimplements Kuhn and Leduc from the rules. `validation/compare_native.py` (provenance
only: it needs the native library) compared the two on 2026-10-02:

```
PASS  kuhn: entries  24 vs 24
PASS  kuhn: node ids, rows, actors, streets, action kinds identical
PASS  kuhn: EV and NashConv on 20 random tables  max err 5.55e-16
PASS  kuhn: Thm-9 accumulator  max err 1.11e-16
PASS  leduc: entries  672 vs 672
PASS  leduc: node ids, rows, actors, streets, action kinds identical
PASS  leduc: EV and NashConv on 20 random tables  max err 6.22e-15
PASS  leduc: Thm-9 accumulator  max err 2.22e-16
PASS  exact_q on identical pivots  n=5142 max err 0.00e+00
PASS  pure rollouts unbiased for exact Q (resampled board)  cells 652 mean z 0.074 var z 1.006
PASS  rollout sample variance native vs pure  ratio 0.9997
PASS  crn rollouts run and agree with exact  max |mean-exact| 0.587
PASS  generate: pivots per hand  native 1.6988 pure 1.6986
PASS  generate: street-1 share  native 0.4113 pure 0.4113
PASS  generate: mean n_street_nodes  native 1.2743 pure 1.2739
PASS  generate: mean reach_lp  native -0.4170 pure -0.4168
PASS  generate: distribution of pivot rows (two-sample chi2)  chi2/df 1.060 df 287 z 0.73

17/17 checks passed
```

Random-number streams differ (numpy vs the native generator), so sampled runs are equal in distribution, not
bit-identical. Deterministic runs (the exact reference) are bit-identical: e.g. QRE(α = 0.2) = 0.1913846679 and the
refresh-500 / floor-20 curve (0.0038 at step 15000) on both.

## 2. Independent checks inside this repository

* `python -m lrc.reference --audit`: the full-width evaluator's EV and best-response NashConv equal
  `ToyGame.exploitability` (a separately written best response) to 1e-9 on uniform and 4 random tables.
* `python experiments/audit_harness.py` (15 checks, each with a negative control): Kuhn Nash profile NashConv 0 and
  value −1/18; rollouts unbiased for Q(h, a) with the board resampled (the board-locked target is rejected);
  reservoir-of-one probabilities; reach = product of ε-mixed hero probabilities; the Thm-9 accumulator equals a brute
  force; the logit gradient is NeuRD; the end-to-end reach-average falls while information-free advantages do not.
* `python -m pytest tests`: rules-level checks (zero-sum, payoffs, known values).
* Known values: uniform NashConv 4.747222; the Leduc game value for player 1 is −0.08561 (reference policy at
  NashConv 0.0006), matching the literature's ≈ −0.0856.

## 3. Sampled reproduction on the pure game

First run of `experiments/ladder.py --grid p4 --arms Q_sfadam_b0.9_fixedpulls,Q_sgd_anneal --steps 16000 --seeds 1,2`
on the pure game, against the native run of the same grid:

| arm | native final (2 seeds) | pure final (2 seeds) | native reach-avg | pure reach-avg |
|---|---|---|---|---|
| schedule-free AdamW, fixed pulls | 4.07 | 3.87 (3.34 / 4.39) | 3.06 | 3.03 |
| SGD lr 10, pulls annealed | 0.33 | 0.35 (0.37 / 0.33) | 0.18 | 0.18 |
