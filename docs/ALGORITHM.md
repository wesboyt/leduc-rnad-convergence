# The algorithms, in notation

This document defines the game, the regularised game, the exact reference solver, the exact-gradient learners and
the **sampled rule** (the algorithm under study) precisely enough to re-implement them. Section numbers are cited by
`docs/TIMING.md` and `docs/CLAIMS.md`. Code locations are given as `file:function`.

## 1. The game

Two-player zero-sum Leduc hold'em (`lrc/game.py`). Players $i \in \{1,2\}$; $-i$ is the opponent.

* Deck of 6 cards (2 suits × 3 ranks). Each player antes 1 and receives one private card. Round 1 betting, a public
  board card, round 2 betting. Limit betting: raise size 2 (round 1) and 4 (round 2), at most 2 raises per round;
  player 1 acts first in each round. Showdown: a private card that pairs the board wins, otherwise the higher rank;
  equal ranks split.
* The 120 ordered deals $d = (c_1, c_2, b)$ are equally likely: chance probability $p_c(d) = 1/120$.
* Histories $h$ (deal + public actions), terminal histories $z$ with utility $u_1(z) = -u_2(z)$.
* An **information set** $s \in \mathcal{S}_i$ is (public action sequence, own bucket); bucket = own rank in round 1,
  (own rank, board rank) in round 2. $A(s)$ = legal actions (fold only when facing a bet; raise only below the cap).
  There are 288 information sets with 672 (information set, action) entries.
* A behavioural policy $\pi_i(\cdot \mid s) \in \Delta(A(s))$. The joint policy $\pi = (\pi_1, \pi_2)$ is stored as one
  table over the 672 entries (row = information set).

Reach probabilities for $h$: $P^\pi(h) = p_c(h)\, x_1^\pi(h)\, x_2^\pi(h)$ with own reach
$x_i^\pi(h) = \prod_{(s', a') \sqsubseteq h,\ s' \in \mathcal{S}_i} \pi_i(a' \mid s')$.
Counterfactual reach of $s \in \mathcal{S}_i$: $r^\pi_{-i}(h) = p_c(h)\, x_{-i}^\pi(h)$ for $h \in s$.

Values: $V_i(\pi) = \sum_z P^\pi(z) u_i(z)$; state-action value $q_i^\pi(h, a)$ (expected $u_i$ after taking $a$ at $h$
and following $\pi$). The **information-set action value** (what a rollout from a sampled state estimates):

$$Q_i^\pi(s,a) = \frac{\sum_{h \in s} r_{-i}^\pi(h)\, q_i^\pi(h,a)}{\sum_{h \in s} r_{-i}^\pi(h)},\qquad
\mathrm{cfv}_i^\pi(s,a) = \sum_{h \in s} r_{-i}^\pi(h)\, q_i^\pi(h,a).$$

**NashConv** (the ruler, exact, every deal enumerated: `lrc/game.py:ToyGame.exploitability`):

$$\mathrm{NashConv}(\pi) = \sum_i \Big[\max_{\pi_i'} V_i(\pi_i', \pi_{-i}) - V_i(\pi)\Big] \ \ge 0,$$

zero iff $\pi$ is a Nash equilibrium. Uniform play has NashConv 4.7472. The game value for player 1 is ≈ −0.0856.

## 2. The regularised game and its equilibrium

Fix a **magnet** $\rho$ (a policy table), a magnet temperature $\alpha \ge 0$ and a uniform temperature $\tau \ge 0$
(both may depend on the round). Two ways of regularising appear in this repository.

* **Trajectory regularisation** (R-NaD's reward transform, Perolat et al. 2021): at every decision of player $i$ the
  reward is shifted by $-\alpha \log\frac{\pi_i(a|s)}{\rho(a|s)} - \tau \log\frac{\pi_i(a|s)}{u(a|s)}$ for $i$ and by
  the opposite amount for $-i$. The resulting soft values $Q^{\pi}_{\alpha,\rho,\tau}$ carry the penalties of every later
  decision. `lrc/reference.py:FullLeduc.values(soft=...)`.
* **Node-only regularisation** (the sampled rule's form): the penalty is applied at the decision being updated; the
  continuation return is the raw $Q^\pi$.

In both cases the **regularised equilibrium** $\pi^\star(\rho)$ is characterised per information set by the logit
fixed point

$$\pi^\star(a \mid s) \ \propto\ \rho(a|s)^{\frac{\alpha}{\alpha+\tau}}\ u(a|s)^{\frac{\tau}{\alpha+\tau}}\
\exp\!\Big(\frac{Q(s,a)}{\alpha+\tau}\Big),$$

with $Q$ the soft or raw information-set value under $\pi^\star$ itself. With $\rho = u$ this is a quantal response
equilibrium (QRE) of temperature $\alpha + \tau$. NashConv$(\pi^\star) > 0$ whenever $\alpha + \tau > 0$; it is the
**QRE floor** of that temperature.

## 3. The reference solver: magnetic mirror descent with a moving magnet

`lrc/reference.py:mmd`. Exact full-width $Q$ (all deals), every information set updated every step, closed form
(Sokota et al. 2023, behavioural form):

$$\log \pi_{t+1}(\cdot|s) = \frac{\log \pi_t(\cdot|s) + \eta\alpha \log\rho(\cdot|s) + \eta\tau\log u(\cdot|s) + \eta\, Q^{\pi_t}(s,\cdot)}
{1 + \eta(\alpha + \tau)} \quad (\text{then renormalised}).$$

* **Inner loop**: with $\rho$ fixed the iteration contracts to $\pi^\star(\rho)$; the measured fixed-point residual
  $\max_e |\pi_{t+1}(e) - \pi_t(e)|/\eta$ falls geometrically (claim C2).
* **Outer loop** (magnet refresh): $\rho_{k+1} \leftarrow \pi_t$ every $K$ steps (`magnet="refresh"`), or when the
  residual falls below a tolerance (`"gate"`), or geometrically $\log\rho \leftarrow (1-m)\log\rho + m\log\pi$
  (`"geo"`). Each refresh is one step of a proximal-point method with Bregman divergence = the (dilated) KL; its
  fixed points are Nash equilibria (Perolat et al. 2021).
* **Support floor** `zfloor` $= Z$: after every step $\log\pi(a|s) \ge -Z$ (then renormalised). Without it the
  refresh compounds $e^{-\Delta Q/\alpha}$ at every refresh and probabilities underflow to exactly 0 (claim C3).
* `refresh_noise` $=\sigma$: the refreshed magnet is $\log\rho \leftarrow \mathrm{norm}(\log\pi + \sigma\varepsilon)$,
  $\varepsilon \sim \mathcal{N}(0, I)$: a controlled inexact inner solve (claim C8).

## 4. Exact-gradient learners

`lrc/reference.py:grad_run`. Logits $\theta$ (one per entry), $\pi_\theta = \mathrm{softmax}$ per row; exact
full-width $Q$; node-only advantage

$$A(s,a) = Q^{\pi}(s,a) - \alpha\big(\log\pi(a|s) - \log\rho(a|s)\big) - \tau\big(\log\pi(a|s) - \log u(a|s)\big).$$

Gradient of the loss on the logits of row $s$ (every row weighted equally, or by counterfactual reach with
`row_weight="cfr"`):

* softmax policy gradient (PPO surrogate at ratio 1): $g(s,a) = -\pi(a|s)\big(A(s,a) - \sum_b \pi(b|s) A(s,b)\big)$
* NeuRD (R-NaD's update): $g(s,a) = -\big(A(s,a) - \tfrac{1}{|A(s)|}\sum_b A(s,b)\big)$

Optimizers on $\theta$: SGD $\theta \leftarrow \theta - \gamma g$; Adam $(\beta_1, \beta_2, \epsilon)$; schedule-free
AdamW (Defazio et al. 2024: the gradient is evaluated at $y = (1-\beta_1) z + \beta_1 x$, $z$ takes Adam-normalised
steps, $x$ is a weighted running average of $z$). The magnet is refreshed from the acting policy.

## 5. The sampled rule (the algorithm under study)

`lrc/harness.py:Harness.step`. One step $t$:

1. **Generation.** Hands are played with every seat on $\pi_t$; a uniformly chosen hero seat mixes
   $\epsilon_t$-uniform exploration ($\epsilon$: 0.25 → 0.10 over 3000 steps). On each round the $k$-th hero decision
   replaces the kept **pivot** with probability $1/k$ (reservoir of one). $\ell(h) = \sum \log \tilde\pi(a)$ over the
   hero's $\epsilon$-mixed actions before the pivot.
2. **Collection.** A static per-round quota of $B = 512$ pivots; the round is drawn before a hand is inspected; one
   pivot per hand.
3. **Action values.** For each legal $a$ at pivot $h$: $\hat q(h,a)$ = mean of $n$ continuations ($n$ = 32 in round 1,
   16 in round 2) with the private cards kept and the undealt board resampled, every seat on $\pi_t$; or exact
   $q(h,a)$ (`exact_q=True`). Values are multiplied by $\kappa = 5$ (`payoff_scale`).
4. **Weights.** $\log w = -\ell(h) + \log n_{\text{round}}(h)$; per round: centre, clip at $\pm\log C$ ($C = 10$),
   exponentiate, renormalise to mean 1 (`lrc/weights.py:clip_reach_weights`).
5. **Advantages.** $a^{\text{raw}} = \hat q - \text{mean}_{A(s)}\hat q$; per-round $\sigma_r$ = EMA (0.99) of
   $\mathbb{E}[a^2]$, floored at 1; $\sigma^{\text{ref}}_r$ frozen at step 50. With $\lambda_r = \sigma^{\text{ref}}_r/\sigma_r$:
   $$\tilde A = \mathrm{clip}_{\pm 3}\Big(\frac{a^{\text{raw}}}{\sigma_r} - \mathrm{clip}_{\pm1.5}\big(\lambda_r\,\mathrm{clip}_{\pm1}(\eta_{\text{reg}}(\log\pi - \log\rho)) + \lambda_r\,\eta_{\text{unif}}(\log\pi - \log u)\big)\Big),$$
   then re-centred over $A(s)$.
6. **Loss.** $L = \frac{\sum_h w_h\, \ell^{\text{PPO}}_h}{\sum_h w_h} - \alpha_{\text{ent}}\frac{\sum_h w_h \min(\lambda,2) H(\pi(\cdot|s_h))}{\sum w}
   + \beta_{\text{fwd}}\frac{\sum_h w_h \mathrm{KL}(\rho_{\text{ref}} \| \pi(\cdot|s_h))}{\sum w}$, with the uniform
   surrogate $\ell^{\text{PPO}} = -\frac{1}{|A|}\sum_a \min(r_a\tilde A_a, \mathrm{clip}(r_a, 0.7, 1.3)\tilde A_a)$,
   $r_a = \pi_\theta(a|s)/\mu(a|s)$, $\mu$ = the behaviour policy.
7. **Optimizer** on the logit table (default schedule-free AdamW, lr 0.03, warmup 100).
8. **Refresh.** Reg-KL plateau rule (refresh when the mean of the last 30 steps' weighted $\mathrm{KL}(\pi\|\rho)$
   grew ≤ 2% over the previous 30, not before `reg_min_period`, forced at `reg_max_period`) or a fixed period $K$:
   $\rho \leftarrow \pi$ (or the mean policy over the second half of the inner phase: `magnet_avg="mean"`).

### 5.1 Two identities (audited, `experiments/audit_harness.py` check 6)

At ratio 1 (behaviour = current policy, `lag = 0`) the PPO clip is inactive for one gradient step, and with
re-centred advantages ($\sum_a \tilde A_a = 0$) the logit gradient of the uniform surrogate is

$$\frac{\partial \ell^{\text{PPO}}}{\partial\theta(s,a)} = -\frac{1}{|A(s)|}\Big(\tilde A_a - \pi(a|s)\sum_b \tilde A_b\Big)
= -\frac{\tilde A_a}{|A(s)|},$$

i.e. the sampled rule is a **stochastic NeuRD step** with per-pivot weight $w_h/\sum w$. In expectation over pivots the
step at information set $s$ is proportional to its sampling frequency times its importance weight, which removes
the hero's own reach and leaves the counterfactual reach $\sum_{h\in s} r_{-i}(h)$ (up to the clip).

### 5.2 Temperatures in chips

Because the anchors are scaled by $\lambda_r = \sigma^{\text{ref}}_r/\sigma_r$ and the advantage by $1/\sigma_r$, the
magnet temperature in raw (chip) units is $\alpha_r = \eta_{\text{reg}}\,\sigma^{\text{ref}}_r/\kappa$ and the uniform
temperature is $\tau_r \approx (\eta_{\text{unif}} + \alpha_{\text{ent}})\,\sigma^{\text{ref}}_r/\kappa$. With the measured
$\sigma^{\text{ref}} \approx (4, 11.5)$ the baseline gives $\alpha \approx (0.16, 0.46)$ and $\tau \approx (0.072, 0.21)$.
These are the values the reference experiments use when they say "the baseline's magnet strength" or "the fixed pulls".

### Table 1 — baseline configuration (`lrc/harness.py:defaults`)

| knob | value | knob | value |
|---|---|---|---|
| pivots $B$ | 512 | rollouts $n$ | 32 / 16 |
| $\epsilon$ | 0.25 → 0.10 over 3000 | IS clip $C$ | 10 |
| $\eta_{\text{reg}}$ | 0.20 (clamp ±1) | $\eta_{\text{unif}}$ | 0.05 |
| anchor sum clamp | ±1.5 | advantage clamp | ±3 |
| PPO clip | 0.3 | $\alpha_{\text{ent}}$ | 0.04 (scale ≤ 2) |
| $\beta_{\text{fwd}}$ | 0.08 | payoff scale $\kappa$ | 5 |
| optimizer | schedule-free AdamW, lr 0.03, β 0.9/0.999, warmup 100 | refresh | plateau, min period 1000 (500 in the ladder's `PROD`), max 8000 |

Knobs added for the experiments (all default to the baseline): `optimizer` (`sfadamw` / `adam` / `sgd`), `beta1`,
`beta2`, `adam_eps`, `pull_anneal` (linear decay of $\eta_{\text{unif}}, \alpha_{\text{ent}}$ to 0), `logit_floor`,
`magnet_avg`, `inner_lr_tau` (lr decay within each inner phase), `refine_rule` / `refine_period`, `ref_target`,
`grad_norm`, `exact_q`, `lag`.
