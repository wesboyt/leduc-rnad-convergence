"""reproduce.py -- every claim of docs/CLAIMS.md as an executable test with a PRE-STATED refutation criterion.

    python reproduce.py --list
    python reproduce.py C3 C6            # run some claims
    python reproduce.py all --jobs 4     # everything (~1-2 h on 4 cores; per-claim times in --list)

Each claim runs its experiment from scratch, writes results/claims/<id>.json and prints SUPPORTED or REFUTED with the
numbers the criterion was evaluated on. A REFUTED verdict means the claim, as stated, does not hold on your machine:
please report it (with the json). Criteria were fixed before this file was run on the results it reports.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
OUT = os.path.join(ROOT, "results", "claims")

# per-round temperatures in chips that the baseline's sigma-normalised anchors amount to (docs/ALGORITHM.md s5.3):
# alpha_s = eta_reg * sigma_s / payoff_scale with sigma ~ (4, 11.5): eta_reg 0.2 -> (0.16, 0.46);
# fixed pulls (eta_unif 0.05 + alpha_ent 0.04) -> (0.072, 0.21)
A02, TAU = [0.16, 0.46], [0.072, 0.21]


def _ref():
    import lrc.reference as M
    return M, M.FullLeduc(2)


def _last(curve, k=1):
    xs = [r["nashconv"] for r in curve[-k:]]
    return sum(xs) / len(xs)


# ---- the experiments behind each claim (each returns a dict of numbers) ----------------------------------------------
def run_mmd(kw):
    M, G = _ref()
    _, c = M.mmd(G, **kw)
    return c


def run_grad(kw):
    M, G = _ref()
    _, c = M.grad_run(G, **kw)
    return c


def run_sampled(args):
    over, steps, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import lrc.harness as H
    cfg = H.defaults()
    cfg.update(over)
    cfg.update(steps=steps, seed=seed, report_every=500)
    _, curve = H.run(cfg, None, log=lambda *a: None)
    return [{"t": r["t"], "nashconv": r["current"], "reach_avg": r["reach_avg"]} for r in curve]


def pmap(fn, items, jobs):
    if jobs <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(jobs) as ex:
        return list(ex.map(fn, items))


def c1(jobs):
    """The evaluator and the sampled harness are correct (both audits, every check with a negative control)."""
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=ROOT)
    a = subprocess.run([py, "-m", "lrc.reference", "--audit"], cwd=ROOT, env=env, capture_output=True, text=True)
    b = subprocess.run([py, os.path.join("experiments", "audit_harness.py")], cwd=ROOT, env=env, capture_output=True,
                       text=True)
    tail = (b.stdout.strip().splitlines() or [""])[-1]
    num = {"reference_audit_pass": a.returncode == 0, "harness_audit": tail}
    return num, a.returncode == 0 and b.returncode == 0


def c2(jobs):
    """Exact MMD with a FIXED uniform magnet converges linearly to the QRE of temperature alpha: from step 2000 the
    fixed-point residual shrinks >= 5x per 1000 steps, and NashConv is constant to 1e-9 from 8000 to 12000."""
    c = run_mmd(dict(steps=12000, eta=0.03, alpha=0.2, score_every=1000))
    res = {r["t"]: r["resid"] for r in c}
    ratios = [res[t + 1000] / res[t] for t in range(2000, 12000, 1000)]
    nc = {r["t"]: r["nashconv"] for r in c}
    num = {"resid_ratio_per_1000_max": max(ratios), "nashconv@8000": nc[8000], "nashconv@12000": nc[12000]}
    return num, max(ratios) < 0.2 and abs(nc[12000] - nc[8000]) < 1e-9


def c3(jobs):
    """The reference (MMD + magnet refresh every 500 + log-prob floor 40) reaches NashConv < 0.01 by 60k steps.
    Without the floor probabilities underflow to exactly 0 and late NashConv is >= 3x higher (alpha 0.5, K 500)."""
    base = dict(steps=60000, eta=0.03, magnet="refresh", K=500, score_every=1000)
    conv, nf, fl = pmap(run_mmd, [dict(base, alpha=0.2, zfloor=40.0), dict(base, alpha=0.5),
                                  dict(base, alpha=0.5, zfloor=20.0)], jobs)
    late = lambda c: max(r["nashconv"] for r in c if r["t"] > 30000)  # noqa: E731
    num = {"a0.2_floor40_final": conv[-1]["nashconv"], "a0.5_nofloor_late_max": late(nf),
           "a0.5_floor20_late_max": late(fl), "a0.5_nofloor_min_p": nf[-1]["min_p"],
           "a0.5_floor20_min_p": fl[-1]["min_p"]}
    return num, (conv[-1]["nashconv"] < 0.01 and nf[-1]["min_p"] == 0.0 and late(nf) >= 3 * late(fl))


def c4(jobs):
    """The fixed pulls toward uniform (the baseline's eta_unif + alpha_ent) set a hard floor: their QRE has NashConv
    ~0.109 and a refreshing magnet cannot go below it while they are on."""
    qre, ref = pmap(run_mmd, [dict(steps=20000, eta=0.03, alpha=0.0, tau_u=TAU, zfloor=20.0, score_every=1000),
                              dict(steps=60000, eta=0.03, alpha=A02, tau_u=TAU, magnet="refresh", K=500, zfloor=20.0,
                                   score_every=1000)], jobs)
    num = {"qre_tau": qre[-1]["nashconv"], "refresh_with_tau_min": min(r["nashconv"] for r in ref),
           "refresh_with_tau_last": ref[-1]["nashconv"]}
    return num, 0.10 < num["qre_tau"] < 0.12 and num["refresh_with_tau_min"] > 0.95 * num["qre_tau"]


def c5(jobs):
    """The anchor applied only at the decision (raw continuation returns, the sampled rule's form) converges as well
    as R-NaD's trajectory-regularised values."""
    base = dict(steps=60000, eta=0.03, alpha=0.2, magnet="refresh", K=500, zfloor=20.0, score_every=1000)
    raw, soft = pmap(run_mmd, [dict(base, soft=False), dict(base, soft=True)], jobs)
    r, s = _last(raw, 5), _last(soft, 5)
    num = {"raw_last5": r, "soft_last5": s}
    return num, r < 0.02 and r < 2.0 * s


def c6(jobs):
    """On the IDENTICAL exact objective (node-only anchor, the baseline's magnet strength, refresh 500, floor 20),
    the optimizer decides convergence: schedule-free AdamW fails at every lr; plain SGD / logit steps converge."""
    base = dict(steps=30000, alpha=A02, K=500, zfloor=20.0, score_every=1000)
    arms = {"mirror": ("mmd", dict(steps=30000, eta=0.03, alpha=A02, magnet="refresh", K=500, zfloor=20.0,
                                   score_every=1000)),
            "neurd_sgd_lr0.03": ("grad", dict(base, opt="sgd", pg="neurd", lr=0.03)),
            "softmax_sgd_lr0.1": ("grad", dict(base, opt="sgd", pg="softmax", lr=0.1))}
    for lr in (0.003, 0.01, 0.03):
        arms[f"softmax_sfadamw_lr{lr}"] = ("grad", dict(base, opt="sfadamw", pg="softmax", lr=lr))
        arms[f"softmax_adam_lr{lr}"] = ("grad", dict(base, opt="adam", pg="softmax", lr=lr))
    names = list(arms)
    res = pmap(_dispatch, [arms[n] for n in names], jobs)
    num = {n: _last(c, 5) for n, c in zip(names, res)}
    sf = [num[f"softmax_sfadamw_lr{lr}"] for lr in (0.003, 0.01, 0.03)]
    ok = min(sf) > 1.0 and num["mirror"] < 0.03 and num["neurd_sgd_lr0.03"] < 0.1 and num["softmax_sgd_lr0.1"] < 0.2
    return num, ok


def _dispatch(x):
    kind, kw = x
    return run_mmd(kw) if kind == "mmd" else run_grad(kw)


def c7(jobs):
    """Momentum is a mechanism: with Adam on the same exact objective, beta1 = 0.9 ends > 3x worse than beta1 = 0 and
    > 2x worse than beta1 = 0.5."""
    base = dict(steps=30000, alpha=A02, K=500, zfloor=20.0, score_every=1000, opt="adam", pg="softmax", lr=0.003)
    b1s = (0.0, 0.5, 0.9)
    res = pmap(run_grad, [dict(base, betas=(b, 0.999)) for b in b1s], jobs)
    num = {f"beta1={b}": _last(c, 5) for b, c in zip(b1s, res)}
    v = [num[f"beta1={b}"] for b in b1s]
    return num, v[2] > 3 * v[0] and v[2] > 2 * v[1]


def c8(jobs):
    """The outer loop's error budget is small: Gaussian noise of sd 0.3 (log-prob) on the magnet at each refresh
    raises NashConv > 10x over the exact refresh; sd 0.03 keeps it < 0.06."""
    sds = (0.0, 0.03, 0.3)
    res = pmap(run_mmd, [dict(steps=40000, eta=0.03, alpha=A02, magnet="refresh", K=500, zfloor=20.0,
                              refresh_noise=sd, score_every=1000) for sd in sds], jobs)
    num = {f"sd={sd}": _last(c, 5) for sd, c in zip(sds, res)}
    v = [num[f"sd={sd}"] for sd in sds]
    return num, v[2] > 10 * v[0] and v[1] < 0.06


def c9(jobs):
    """In the SAMPLED rule (full baseline step at eta_reg 0.2, refresh 500), schedule-free AdamW degrades to NashConv
    > 2, plain SGD stays < 0.6 (two seeds each, 16k steps)."""
    import ladder as L
    items = [(L.GRIDS["p4"]["Q_sfadam_b0.9_fixedpulls"], 16000, s) for s in (1, 2)] + \
            [(L.GRIDS["p4"]["Q_sgd_anneal"], 16000, s) for s in (1, 2)]
    res = pmap(run_sampled, items, jobs)
    sf = [c[-1]["nashconv"] for c in res[:2]]
    sg = [c[-1]["nashconv"] for c in res[2:]]
    num = {"sfadamw_final": sf, "sgd_final": sg, "sgd_reach_avg": [c[-1]["reach_avg"] for c in res[2:]]}
    return num, min(sf) > 2.0 and max(sg) < 0.6


def c10(jobs):
    """The sampled rule's floor is NOT noise in the action values: with EXACT Q the SGD rule plateaus at the same
    level as with sampled Q (within 1.5x, both > 0.2)."""
    import ladder as L
    items = [(L.GRIDS["p4b"]["Q_sgd_exactQ"], 16000, s) for s in (1, 2)] + \
            [(L.GRIDS["p4"]["Q_sgd_anneal"], 16000, s) for s in (1, 2)]
    res = pmap(run_sampled, items, jobs)
    ex = sum(_last(c, 4) for c in res[:2]) / 2
    sa = sum(_last(c, 4) for c in res[2:]) / 2
    num = {"exactQ_last4_mean": ex, "sampledQ_last4_mean": sa}
    return num, min(ex, sa) > 0.2 and max(ex, sa) / min(ex, sa) < 1.5


def c11(jobs):
    """The sampled rule's EXPECTED step vanishes (approximately) at the exact regularised equilibrium with clips off:
    its fixed point is the right one."""
    import numpy as np
    import fixedpoint_probe as F
    M, G = _ref()
    sig = (4.0, 11.5)
    alpha = [0.2 * s / 5.0 for s in sig]
    pi, _ = M.mmd(G, steps=6000, eta=0.03, alpha=alpha, soft=False, magnet="fixed", zfloor=30.0, score_every=6000)
    noclip = dict(anchor_clip_z=1e9, anchor_sum_clip=1e9, adv_clip=1e9, ppo_clip=1e9)
    mu, _, _, _ = F.probe(noclip, G.g.uniform(), 200, sig, "uniform, clips off")
    mp, zp, _, _ = F.probe(noclip, pi, 200, sig, "pi*, clips off")
    num = {"grad_norm_uniform": float(np.linalg.norm(mu)), "grad_norm_at_qre": float(np.linalg.norm(mp)),
           "frac_entries_absz_gt4": float((np.abs(zp) > 4).mean())}
    return num, num["grad_norm_at_qre"] < 0.05 * num["grad_norm_uniform"] and num["frac_entries_absz_gt4"] < 0.1


def c12(jobs):
    """TIMING: with a FIXED magnet and EXACT Q, the sampled rule has not solved its regularised game after 16k steps
    (round-2 RMS log-prob error > 0.5, NashConv > 1.2x the exact QRE), while the exact reference solves the same
    game to 1% in < 2000 steps."""
    import numpy as np
    import ladder as L
    import lrc.harness as H
    M, G = _ref()
    g = G.g
    cfg = H.defaults()
    cfg.update(L.GRIDS["inner"]["I_sgd_lr10.0"])
    cfg.update(steps=16000, seed=1)
    h = H.Harness(cfg)
    for _ in range(16000):
        h.step()
    alpha = [cfg["eta_reg"] * s / cfg["payoff_scale"] for s in h.sigma_ref]
    pi_qre, cq = M.mmd(G, steps=6000, eta=0.03, alpha=alpha, soft=False, magnet="fixed", zfloor=20.0, score_every=100)
    nc_qre = cq[-1]["nashconv"]
    t_ref = next(r["t"] for r in cq if abs(r["nashconv"] - nc_qre) < 0.01 * nc_qre)
    p_it = h.probs_of(h.logits)
    vals = G.values(pi_qre)
    w = vals["own"] * vals["cfr"]
    err = np.log(np.maximum(p_it, 1e-12)) - np.log(np.maximum(pi_qre, 1e-12))
    st = g.row_street[g.entry_row]
    rms = [float(np.sqrt((w[st == s] * err[st == s] ** 2).sum() / w[st == s].sum())) for s in range(2)]
    nc_it = g.exploitability(p_it)["nashconv"]
    num = {"alpha_chips": alpha, "nashconv_qre": nc_qre, "reference_steps_to_1pct": t_ref,
           "sampled_nashconv_16k": nc_it, "sampled_rms_logerr_by_round": rms}
    return num, rms[1] > 0.5 and nc_it > 1.2 * nc_qre and t_ref < 2000


# ---- C13-C20: the previously untested hypotheses (criteria fixed before the first run) ---------------------------
_EX = dict(steps=30000, alpha=A02, K=500, zfloor=20.0, score_every=1000)


def c13(jobs):
    """SCHEDULE-FREE MECHANISM (exact, table): schedule-free SGD (interpolation beta 0.9) fails (> 1.0) on the same
    objective and learning rates at which plain SGD converges (< 0.2): the failure is schedule-free's averaging /
    interpolation, not Adam's normalisation."""
    arms = {"sgd_lr0.1": dict(_EX, opt="sgd", pg="softmax", lr=0.1),
            "sfsgd_lr0.1": dict(_EX, opt="sfsgd", pg="softmax", lr=0.1, betas=(0.9, 0.999)),
            "sfsgd_lr0.03": dict(_EX, opt="sfsgd", pg="softmax", lr=0.03, betas=(0.9, 0.999))}
    names = list(arms)
    res = pmap(run_grad, [arms[n] for n in names], jobs)
    num = {n: _last(c, 5) for n, c in zip(names, res)}
    return num, num["sgd_lr0.1"] < 0.2 and num["sfsgd_lr0.1"] > 1.0 and num["sfsgd_lr0.03"] > 1.0


def c14(jobs):
    """DOSE-RESPONSE (exact, table): for schedule-free SGD (lr 0.1) and schedule-free AdamW (lr 0.01), NashConv at
    interpolation beta = 0.9 is > 5x that at beta = 0 (where schedule-free acts at its base iterate), and beta = 0
    itself converges (< 0.3)."""
    arms = {}
    for b in (0.0, 0.5, 0.9):
        arms[f"sfsgd_b{b}"] = dict(_EX, opt="sfsgd", pg="softmax", lr=0.1, betas=(b, 0.999))
        arms[f"sfadamw_b{b}"] = dict(_EX, opt="sfadamw", pg="softmax", lr=0.01, betas=(b, 0.999))
    names = list(arms)
    res = pmap(run_grad, [arms[n] for n in names], jobs)
    num = {n: _last(c, 5) for n, c in zip(names, res)}
    ok = all(num[f"{o}_b0.9"] > 5 * num[f"{o}_b0.0"] and num[f"{o}_b0.0"] < 0.3 for o in ("sfsgd", "sfadamw"))
    return num, ok


_NET = dict(steps=30000, alpha=A02, K=5000, zfloor=20.0, score_every=1000, policy="mlp", pg="neurd",
            logit_threshold=2.0, lr=1e-3)


def c15(jobs):
    """SHARED-WEIGHT NETWORK (exact values, the regime of LLM policy training: normalisation per WEIGHT): mean
    NashConv over the last 10k of 30k steps, 3 seeds. Schedule-free AdamW (beta 0.9) is > 2x worse than Adam with
    beta1 = 0, and Adam with beta1 = 0.9 is worse than beta1 = 0."""
    arms = {"adam_b1_0": dict(opt="adam", betas=(0.0, 0.999)), "adam_b1_0.9": dict(opt="adam", betas=(0.9, 0.999)),
            "sfadamw_b0.9": dict(opt="sfadamw", betas=(0.9, 0.999)), "sfadamw_b0": dict(opt="sfadamw", betas=(0.0, 0.999))}
    items, keys = [], []
    for n, kw in arms.items():
        for sd in (0, 1, 2):
            items.append(dict(_NET, seed=sd, **kw))
            keys.append(n)
    res = pmap(run_grad, items, jobs)
    num = {}
    for n in arms:
        vals = [_last(c, 10) for k, c in zip(keys, res) if k == n]
        num[n] = sum(vals) / len(vals)
        num[n + "_per_seed"] = vals
    return num, num["sfadamw_b0.9"] > 2 * num["adam_b1_0"] and num["adam_b1_0.9"] > num["adam_b1_0"]


def c16(jobs):
    """SCHEDULE-FREE IN THE SAMPLED RULE: schedule-free SGD with SGD's learning rate (10) and everything else as the
    converging SGD arm of C9 ends > 1.0 on both seeds (16k steps); SGD ends < 0.6 (C9)."""
    import ladder as L
    base = L.GRIDS["p4"]["Q_sgd_anneal"]
    res = pmap(run_sampled, [(dict(base, optimizer="sfsgd", beta1=0.9), 16000, s) for s in (1, 2)], jobs)
    num = {"sfsgd_final": [c[-1]["nashconv"] for c in res], "sfsgd_reach_avg": [c[-1]["reach_avg"] for c in res]}
    return num, min(num["sfsgd_final"]) > 1.0


def c17(jobs):
    """OPTIMISM / EXTRAGRADIENT (exact, table; Daskalakis & Panageas, Lee et al., Cen et al., Gidel et al.):
    (a) optimism improves Adam beta1 0.9 (softmax PG, lr 0.003) by > 2x; (b) optimism does NOT rescue schedule-free
    AdamW (lr 0.01 stays > 1.0); (c) extragradient stabilises softmax-PG SGD at lr 0.3 (< 0.1)."""
    arms = {"adam_b0.9": dict(_EX, opt="adam", pg="softmax", lr=0.003, betas=(0.9, 0.999)),
            "adam_b0.9_optimistic": dict(_EX, opt="adam", pg="softmax", lr=0.003, betas=(0.9, 0.999), optimistic=True),
            "sfadamw": dict(_EX, opt="sfadamw", pg="softmax", lr=0.01, betas=(0.9, 0.999)),
            "sfadamw_optimistic": dict(_EX, opt="sfadamw", pg="softmax", lr=0.01, betas=(0.9, 0.999), optimistic=True),
            "sgd_lr0.3": dict(_EX, opt="sgd", pg="softmax", lr=0.3),
            "sgd_lr0.3_extragradient": dict(_EX, opt="sgd", pg="softmax", lr=0.3, extragradient=True)}
    names = list(arms)
    res = pmap(run_grad, [arms[n] for n in names], jobs)
    num = {n: _last(c, 5) for n, c in zip(names, res)}
    a = num["adam_b0.9_optimistic"] < 0.5 * num["adam_b0.9"]
    b = num["sfadamw_optimistic"] > 1.0
    cc = num["sgd_lr0.3_extragradient"] < 0.1
    num.update({"a_optimism_helps_adam": a, "b_optimism_does_not_rescue_sf": b, "c_extragradient_stabilises": cc})
    return num, a and b and cc


def _sampled_pair(over_a, over_b, steps, jobs):
    import ladder as L
    base = L.GRIDS["p4"]["Q_sgd_anneal"]
    items = [(dict(base, **over_a), steps, s) for s in (1, 2)] + [(dict(base, **over_b), steps, s) for s in (1, 2)]
    res = pmap(run_sampled, items, jobs)
    fa = [_last(c, 4) for c in res[:2]]
    fb = [_last(c, 4) for c in res[2:]]
    return fa, fb, res


def c18(jobs):
    """APMD's NOISY-FEEDBACK PRESCRIPTION in the sampled rule (Abe et al. 2024): learning rate restarted at every
    slingshot (magnet) update and decayed as lr / (1 + s / 500), slingshot interval T^(4/5) ~ 4000 for T = 32k. Mean
    of the last 2k steps (2 seeds) is < 0.8x the same interval with a constant learning rate, and < 0.25."""
    fa, fb, _ = _sampled_pair(dict(refine_period=4000, inner_lr_tau=500.0), dict(refine_period=4000), 32000, jobs)
    ma, mb = sum(fa) / 2, sum(fb) / 2
    num = {"apmd_last2k": fa, "constant_lr_last2k": fb, "apmd_mean": ma, "constant_mean": mb}
    return num, ma < 0.8 * mb and ma < 0.25


def c19(jobs):
    """THE DEEPNASH RECIPE in the sampled rule: Adam beta1 = 0, magnet interpolation alpha_n = min(1, 2n/K), NeuRD
    threshold 2, long inner phase K = 4000 (32k steps, 2 seeds): mean of the last 2k steps < 0.5; the same recipe with
    schedule-free AdamW (beta 0.9) is > 2x worse."""
    rec = dict(optimizer="adam", beta1=0.0, lr=0.003, magnet_interp=True, neurd_threshold=2.0, refine_period=4000)
    fa, fb, _ = _sampled_pair(rec, dict(rec, optimizer="sfadamw", beta1=0.9), 32000, jobs)
    ma, mb = sum(fa) / 2, sum(fb) / 2
    num = {"deepnash_adam_b0": fa, "deepnash_sfadamw": fb, "adam_mean": ma, "sfadamw_mean": mb}
    return num, ma < 0.5 and mb > 2 * ma


def c20(jobs):
    """TEMPERATURE ANNEALING instead of a moving magnet (Sokota et al. 2023) in the sampled rule: uniform magnet never
    refreshed, eta_reg annealed geometrically 0.2 -> 0.02 over 32k steps, SGD (2 seeds): mean of the last 2k steps
    < 0.25."""
    import ladder as L
    base = L.GRIDS["p4"]["Q_sgd_anneal"]
    over = dict(refine_period=10 ** 9, reg_max_period=10 ** 9, eta_reg_end=0.02, eta_reg_anneal=32000)
    res = pmap(run_sampled, [(dict(base, **over), 32000, s) for s in (1, 2)], jobs)
    f = [_last(c, 4) for c in res]
    num = {"anneal_last2k": f, "mean": sum(f) / 2, "reach_avg": [c[-1]["reach_avg"] for c in res]}
    return num, num["mean"] < 0.25


def c21(jobs):
    """MECHANISM (linearised regularised game F(w) = (lambda I + omega J) w, lambda = 1, step 0.005 at which SGD
    converges for every omega tested): schedule-free SGD and schedule-free AdamW converge in MINIMISATION (omega = 0:
    |w_T|/|w_0| < 1e-3 after 20k steps) but NOT in rotation-dominated games (omega = 30: > 0.1), while SGD converges
    at every omega (< 1e-3)."""
    import sf_linear as S
    res = S.grid(omegas=(0.0, 1.0, 10.0, 30.0), gamma=0.005)
    ok = (all(res["sgd"][k] < 1e-3 for k in res["sgd"])
          and all(res[o]["0.0"] < 1e-3 and res[o]["30.0"] > 0.1 for o in ("sfsgd_b0.9", "sfadamw_b0.9")))
    return res, ok


# claim -> [(paper, relation)]; relation: CONFIRMED (the paper's prediction reproduces) | BOUNDARY (the guarantee does
# not carry over once a named hypothesis is dropped: refutes an EXTRAPOLATION, not the paper). docs/PAPERS.md.
PAPERS = {
    "C1": [],
    "C2": [("Sokota+ 2023 MMD", "CONFIRMED: linear last-iterate rate to the QRE"),
           ("Cen+ 2023", "CONFIRMED: linear convergence to the regularised equilibrium")],
    "C3": [("Perolat+ 2021", "CONFIRMED: iterating the regularised fixed point converges to Nash"),
           ("Sokota+ 2023 MMD", "CONFIRMED: a moving magnet approaches Nash"),
           ("Perolat+ 2022 DeepNash", "CONFIRMED: a support safeguard (cf. logit thresholding) is needed")],
    "C4": [("Sokota+ 2023 MMD", "CONFIRMED: a fixed magnet/temperature limits at its QRE"),
           ("Cen+ 2023", "CONFIRMED: regularisation bias floor")],
    "C5": [("Perolat+ 2022 DeepNash", "CONFIRMED: node-only reward transformation suffices on Leduc")],
    "C6": [("Sokota+ 2023 MMD", "BOUNDARY: the objective is not optimisable by an arbitrary first-order optimizer"),
           ("Daskalakis & Panageas 2018", "CONFIRMED in spirit: simultaneous dynamics need not converge")],
    "C7": [("Sokota+ 2023 MMD", "BOUNDARY: momentum breaks the mirror-step guarantee"),
           ("Gidel+ 2019 (negative momentum)", "CONFIRMED in spirit: positive momentum hurts game dynamics")],
    "C8": [("Rockafellar 1976", "CONFIRMED: non-summable proximal-point errors stall the outer loop"),
           ("Perolat+ 2021", "BOUNDARY: the fixed point must be reached before the magnet moves")],
    "C9": [("Perolat+ 2022 DeepNash", "BOUNDARY: R-NaD realised with schedule-free AdamW does not converge")],
    "C10": [("Abe+ 2024 APMD", "BOUNDARY: a periodic magnet with a constant step does not absorb per-row sampling noise")],
    "C11": [],
    "C12": [("Perolat+ 2021", "BOUNDARY: a short refresh clock vs a slow sampled inner solve"),
            ("Abe+ 2024 APMD", "BOUNDARY: heterogeneous per-information-set noise, one constant step")],
    "C13": [("Defazio+ 2024 schedule-free", "BOUNDARY: averaging/interpolation breaks last-iterate game dynamics"),
            ("Sokota+ 2023 MMD", "BOUNDARY: the objective needs a mirror-like step")],
    "C14": [("Defazio+ 2024 schedule-free", "BOUNDARY: dose-response in the interpolation parameter")],
    "C15": [("Perolat+ 2022 DeepNash", "test: momentum-free Adam (DeepNash's b1 = 0) vs schedule-free with a network"),
            ("Feng, Ou & Wang 2026 (Adam ODE in zero-sum games)", "test: momentum role in games, network policy")],
    "C16": [("Defazio+ 2024 schedule-free", "BOUNDARY: schedule-free SGD under sampled feedback")],
    "C17": [("Daskalakis & Panageas 2018; Lee+ 2021; Cen+ 2023", "test: optimism"),
            ("Gidel+ 2019 VI", "test: extragradient"), ("Daskalakis+ 2018 Optimistic Adam", "test: optimism vs schedule-free")],
    "C18": [("Abe+ 2024 APMD", "test: the noisy-feedback learning-rate and slingshot-interval prescription")],
    "C19": [("Perolat+ 2022 DeepNash", "test: the full recipe (b1 = 0, interpolation, threshold, long phases)")],
    "C20": [("Sokota+ 2023 MMD", "test: temperature annealing instead of a moving magnet")],
    "C21": [("Defazio+ 2024 schedule-free", "BOUNDARY test: minimisation (designed regime) vs rotation-dominated games"),
            ("Gidel+ 2019 VI; Daskalakis & Panageas 2018", "rotational vector fields and optimizer stability")],
}

CLAIMS = {"C1": (c1, "1 min"), "C2": (c2, "15 s"), "C3": (c3, "4 min"), "C4": (c4, "4 min"), "C5": (c5, "4 min"),
          "C6": (c6, "10 min / 4 jobs"), "C7": (c7, "4 min"), "C8": (c8, "3 min"), "C9": (c9, "15 min / 4 jobs"),
          "C10": (c10, "15 min / 4 jobs"), "C11": (c11, "3 min"), "C12": (c12, "8 min"),
          "C13": (c13, "4 min"), "C14": (c14, "6 min"), "C15": (c15, "15 min / 8 jobs"), "C16": (c16, "8 min"),
          "C17": (c17, "6 min"), "C18": (c18, "30 min / 4 jobs"), "C19": (c19, "30 min / 4 jobs"),
          "C20": (c20, "30 min / 2 jobs"), "C21": (c21, "2 min")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("claims", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.claims:
        for k, (fn, t) in CLAIMS.items():
            print(f"{k:4s} ({t:>16s})  {fn.__doc__.split(chr(10))[0]}")
        return
    ids = list(CLAIMS) if a.claims == ["all"] else [c.upper() for c in a.claims]
    os.makedirs(OUT, exist_ok=True)
    verdicts = {}
    for cid in ids:
        fn, _ = CLAIMS[cid]
        t0 = time.time()
        num, ok = fn(a.jobs)
        verdicts[cid] = ok
        rec = {"claim": cid, "statement": " ".join(fn.__doc__.split()), "verdict": "SUPPORTED" if ok else "REFUTED",
               "numbers": num, "seconds": round(time.time() - t0, 1),
               "papers": [{"paper": p, "relation": r} for p, r in PAPERS[cid]]}
        with open(os.path.join(OUT, cid + ".json"), "w") as fh:
            json.dump(rec, fh, indent=1, default=float)
        print(f"{cid}: {rec['verdict']}  ({rec['seconds']}s)  {json.dumps(num, default=float)}", flush=True)
    print("\n" + "  ".join(f"{k}={'ok' if v else 'REFUTED'}" for k, v in verdicts.items()))
    sys.exit(0 if all(verdicts.values()) else 1)


if __name__ == "__main__":
    main()
