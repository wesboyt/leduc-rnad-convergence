# Claims, evidence and refutation criteria

Every claim below is executable: `python reproduce.py C<n>` reruns its experiment from scratch, evaluates the stated
criterion, prints **SUPPORTED** or **REFUTED** with the numbers it used, and writes `results/claims/C<n>.json`.

* "Original" = the first measurement, made on the native validation-game library (`docs/VALIDATION.md` shows that the
  pure-numpy game used here is identical on every exact quantity and statistically identical on every sampled one).
* "This repository" = the output of `reproduce.py all` on the pure-numpy implementation (`results/claims/`).
* NashConv is in chips (big blind = ante = 1); uniform play = 4.7472.
* Temperatures written as $(a, b)$ are per round. "The baseline's magnet strength" = $\alpha = (0.16, 0.46)$ and "the
  fixed pulls" = $\tau = (0.072, 0.21)$ (`docs/ALGORITHM.md` §5.2).

| id | claim | refuted if | original | this repository |
|---|---|---|---|---|
| C1 | The exact evaluator and the sampled harness measure what they claim (audits with negative controls) | any audit check fails | 15/15 + reference audit PASS | see `RESULTS.md` |
| C2 | Exact MMD with a fixed uniform magnet converges linearly to the QRE | from step 2000 the residual shrinks < 5× per 1000 steps, or NashConv changes > 1e-9 between 8k and 12k | QRE(α=0.2) = 0.19138, flat | see `RESULTS.md` |
| C3 | MMD + magnet refresh every 500 + log-prob floor reaches NashConv < 0.01; without the floor support is lost (p = 0) and late NashConv is ≥ 3× higher | final (60k, α=0.2, floor 40) ≥ 0.01, or no-floor min p > 0, or no-floor late max < 3× floored late max (α=0.5) | 0.0006; no floor: spikes to 0.61 | see `RESULTS.md` |
| C4 | The fixed pulls set a hard floor: their QRE has NashConv ≈ 0.109 and a refreshing magnet cannot go below it while they are on | QRE(τ) outside (0.10, 0.12) or refreshing min < 0.95 × QRE(τ) | 0.1087; pinned at 0.1087 | see `RESULTS.md` |
| C5 | The node-only anchor (raw continuation values, the sampled rule's form) converges as well as trajectory regularisation | raw last-5 mean ≥ 0.02 or ≥ 2× soft | 0.0022 vs 0.0028 | see `RESULTS.md` |
| C6 | On the identical exact objective the optimizer decides convergence: schedule-free AdamW fails at every lr, SGD/logit steps converge | any sf-AdamW lr ∈ {0.003, 0.01, 0.03} ends ≤ 1.0, or mirror ≥ 0.03, NeuRD-SGD ≥ 0.1, softmax-SGD ≥ 0.2 | sf-AdamW 4.45/4.80/5.96; mirror 0.006; NeuRD 0.023; SGD 0.062 | see `RESULTS.md` |
| C7 | Momentum is a mechanism: Adam with β1 = 0.9 ends > 3× worse than β1 = 0 and > 2× worse than 0.5 | either ratio fails | 0.335 vs 0.044 / 0.055 | see `RESULTS.md` |
| C8 | The outer loop's error budget is small: magnet noise sd 0.3 per refresh raises NashConv > 10×; sd 0.03 keeps it < 0.06 | either fails | 0.0045 → 0.35; sd 0.03: 0.044 | see `RESULTS.md` |
| C9 | In the sampled rule, schedule-free AdamW degrades (> 2) while SGD stays < 0.6 (2 seeds, 16k) | any sf-AdamW seed ≤ 2 or any SGD seed ≥ 0.6 | 4.07 vs 0.33 (native); 3.87 vs 0.35 (pure, first run) | see `RESULTS.md` |
| C10 | The sampled rule's ~0.3 floor is not noise in Q: exact Q plateaus at the same level (within 1.5×, both > 0.2) | ratio ≥ 1.5 or either ≤ 0.2 | exact 0.38, sampled 0.33 | see `RESULTS.md` |
| C11 | The sampled rule's expected step (clips off) vanishes at the exact regularised equilibrium | ‖ḡ(π*)‖ ≥ 5% of ‖ḡ(uniform)‖ or > 10% of entries with \|z\| > 4 | 2.2e-3 vs 7.7e-2 (clips on at uniform); 3.7% | see `RESULTS.md` |
| C12 | TIMING: fixed magnet, exact Q, SGD lr 10: after 16k steps the sampled rule has not solved its regularised game (round-2 RMS log-prob error > 0.5, NashConv > 1.2× the QRE) while the exact reference solves it to 1% in < 2000 steps | any of the three fails | round-2 error 0.91, NashConv 0.23 vs 0.140 | see `RESULTS.md` |

## Claims and the papers they bear on

CONFIRMED = the paper's prediction reproduces. BOUNDARY = the paper's guarantee does not carry over once a named
hypothesis is dropped; this refutes an *extrapolation* of the paper, not the paper. No claim here contradicts a
theorem within its hypotheses. Paper-by-paper detail and the untested prescriptions: `docs/PAPERS.md`.

| claim | paper | relation |
|---|---|---|
| C2 | Sokota et al. 2023 (MMD); Cen et al. 2023 | CONFIRMED: linear last-iterate convergence to the regularised equilibrium |
| C3 | Perolat et al. 2021; Sokota et al. 2023; Perolat et al. 2022 | CONFIRMED: iterating the magnet reaches Nash; a support safeguard is needed |
| C4 | Sokota et al. 2023; Cen et al. 2023 | CONFIRMED: a fixed temperature limits at its QRE (bias floor) |
| C5 | Perolat et al. 2022 | CONFIRMED: node-only reward transformation suffices here |
| C6 | Sokota et al. 2023; Daskalakis & Panageas 2018 | BOUNDARY: not a mirror/gradient step → no convergence |
| C7 | Sokota et al. 2023; Gidel et al. 2019 (negative momentum) | BOUNDARY: momentum breaks the step-size condition |
| C8 | Rockafellar 1976; Perolat et al. 2021 | CONFIRMED (inexact proximal point) / BOUNDARY (fixed point not reached before the magnet moves) |
| C9 | Perolat et al. 2022 | BOUNDARY: R-NaD realised with schedule-free AdamW |
| C10 | Abe et al. 2024 (APMD) | BOUNDARY: periodic magnet + constant step does not absorb per-row sampling noise |
| C12 | Perolat et al. 2021; Abe et al. 2024 | BOUNDARY: refresh clock shorter than the sampled inner solve; heterogeneous noise |

## How the claims fit together

```
C1  instruments are right
C2  ─┐
C3  ─┴─ the game converges under a correct update (with a support floor: T5)
C4  ─── fixed pulls = QRE floor (T4)
C5  ─── the sampled rule's Q form is fine
C6  ─┐
C7  ─┼─ the optimizer is a cause (T2), exact and sampled
C9  ─┘
C8  ─┐
C10 ─┼─ timing: the outer loop needs a precise inner solve; the sampled inner loop cannot provide it on the
C11 ─┤   refresh clock, and the reason is not Q noise or a wrong fixed point but per-row stochastic
C12 ─┘   approximation (T1, T3)
```

## Revisions of criteria (logged, not hidden)

* **C2.** The first criterion was "NashConv changes < 1e-6 between steps 3000 and 6000". Its first run gave 1.8e-6:
  the iteration is linear but at ≈ 0.0023/step, not fast enough for that window. The claim (linear convergence) was
  kept and the criterion restated as a rate: residual ↓ ≥ 5× per 1000 steps from step 2000, NashConv constant to
  1e-9 from 8k to 12k.
* **C3** and **C7** were restated before their first run in this repository. For C3, the no-floor contrast at α = 0.2
  is weak because that run happens to end low, so α = 0.5 is used. For C7, β1 = 0 and β1 = 0.5 are close
  (0.044 vs 0.055), so a strict three-way ordering is not claimed.

## Rejected hypotheses (negative results you can also challenge)

| hypothesis | command | result |
|---|---|---|
| per-row step ∝ counterfactual reach is the floor | `python experiments/mmd_grid.py --grid cfw` | ~2× slower in the exact reference, no floor |
| lost support in the sampled rule | `python experiments/ladder.py --grid floor --steps 16000 --seeds 1,2` | floor 20: unchanged; 10: 0.29–0.33 |
| the magnet should be an average / lr should decay within the inner phase | `python experiments/ladder.py --grid avg --steps 16000 --seeds 1,2` | 0.31–0.38 |
| a gated refresh (residual < tol) beats a fixed K in the exact reference | `python -m lrc.reference --magnet gate --gate-tol 1e-6 --K 100 --steps 20000 --eta 0.03 --alpha 0.2` | no: too few refreshes (3 in 20k → 0.198) |
| removing one baseline component (clips, forward KL, IS weights, exploration) lifts the sampled floor | `python experiments/ladder.py --grid comp --steps 16000 --seeds 1,2` | no (0.37–1.16) |
