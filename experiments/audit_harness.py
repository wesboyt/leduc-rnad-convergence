"""audit_harness.py -- the harness measures what it claims. Every check has a negative control that must be rejected.

    python experiments/audit_harness.py

  1. exact scoring: the Kuhn Nash profile (alpha = 0) has NashConv 0 and value -1/18. control: a perturbed profile
  2. rollouts estimate Q(h,a) with the board resampled: pooled z over (pivot, action) at 4000 sims. control: the
     board-locked target, which a resampling bug would estimate
  3. reservoir of one: among hands with 2 hero decisions on a street, the kept node is the first w.p. 1/2.
     control: keep-first
  4. reach: exp(reach_lp) equals the product of the hero's eps-mixed probabilities on the kept node's path.
     control: the un-mixed product
  5. Thm-9 accumulator equals a brute-force reach-weighted average of two policies. control: the plain average
  6. the loss is NeuRD: with ratio 1, weights 1 and no regularisers the logit gradient equals -A_b/(|A| B).
     control: uncentred advantages with anchors on break the identity
  7. end to end at the baseline: the reach-weighted average's NashConv falls. control: information-free
     advantages do not. Also: the sigma floor is not binding
"""
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import lrc.harness as L  # noqa: E402  (sets CUDA_VISIBLE_DEVICES)

import numpy as np  # noqa: E402
from lrc.game import ToyGame  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def kuhn_nash(g):
    """alpha=0 Kuhn equilibrium in the tree's layout. Cards 0=J 1=Q 2=K."""
    p = g.uniform().copy()
    # identify nodes by (actor, facing a bet, depth) from kinds: fold present <=> facing a bet
    for nid, nA in g.node_nA.items():
        kinds = g.node_kinds[nid]
        actor = g.node_actor[nid]
        facing = 0 in kinds
        root = nid == 0
        for card in range(3):
            base = g.node_row0[nid] + card * nA
            if not facing:                       # [check, bet]
                if root:                          # P0 first: never bet (alpha=0)
                    bet = 0.0
                else:                             # P1 after a check: bluff J 1/3, bet K
                    bet = {0: 1 / 3, 1: 0.0, 2: 1.0}[card]
                p[base], p[base + 1] = 1 - bet, bet
            else:                                 # [fold, call]
                if actor == 1:                    # P1 facing a bet
                    call = {0: 0.0, 1: 1 / 3, 2: 1.0}[card]
                else:                             # P0 after check-bet
                    call = {0: 0.0, 1: 1 / 3, 2: 1.0}[card]
                p[base], p[base + 1] = 1 - call, call
    return p


def paths(g):
    """parent map: node -> (parent, action index)"""
    par = {}
    for nid, kids in g.node_children.items():
        for a, c in enumerate(kids):
            par[int(c)] = (nid, a)
    return par


def main():
    # 1
    k = ToyGame("kuhn", 2)
    nash = kuhn_nash(k)
    x = k.exploitability(nash)
    check("kuhn Nash profile: NashConv 0, value -1/18", x["nashconv"] < 1e-12 and abs(x["ev"][0] + 1 / 18) < 1e-12,
          "nashconv=%.2e ev0=%.6f" % (x["nashconv"], x["ev"][0]))
    bad = nash.copy()
    b0 = k.node_row0[0] + 1 * k.node_nA[0]        # root, Q: start betting 30%
    bad[b0], bad[b0 + 1] = 0.7, 0.3
    check("control: perturbed Kuhn profile exploitable", k.exploitability(bad)["nashconv"] > 1e-3,
          "nashconv=%.4f" % k.exploitability(bad)["nashconv"])

    g = ToyGame("leduc", 2)
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 1.0, g.entries)
    h = L.Harness(dict(L.defaults()))
    probs = h.probs_of(logits)

    # 2
    piv = g.generate(probs, 400, 0.1, seed=5)
    sel = np.nonzero(piv.street == 0)[0][:150].tolist() + np.nonzero(piv.street == 1)[0][:150].tolist()
    piv = piv.take(np.asarray(sel))
    ns = 4000
    q, v = g.rollouts(probs, piv, [ns, ns], seed=9)
    qe = g.exact_q(probs, piv)
    locked = piv.take(np.arange(len(piv)))
    locked.i[:, 2] = 1                               # street>=1: keep the dealt board (the wrong target)
    ql = g.exact_q(probs, locked)
    z, zl = [], []
    for i in range(len(piv)):
        for a in range(g.node_nA[int(piv.node[i])]):
            if v[i, a] > 0:
                se = math.sqrt(v[i, a] / ns)
                z.append((q[i, a] - qe[i, a]) / se)
                if piv.street[i] == 0:
                    zl.append((q[i, a] - ql[i, a]) / se)
    z, zl = np.asarray(z), np.asarray(zl)
    chi, df = float((z * z).sum()), len(z)
    zc = ((chi / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    check("rollouts unbiased for Q(h,a) with resampled runout", zc < 4 and np.abs(z).max() < 5.5,
          "cells=%d chi2/df=%.3f z_chi=%.2f max|z|=%.2f" % (df, chi / df, zc, np.abs(z).max()))
    chil = float((zl * zl).sum()) / len(zl)
    check("control: board-locked target rejected", chil > 3, "chi2/df=%.1f over %d preflop cells" % (chil, len(zl)))

    # 3 and 4
    par = paths(g)
    piv = g.generate(probs, 60000, 0.25, seed=11)
    firsts, n2 = 0, 0
    lp_err, lp_err_nomix = 0.0, 0.0
    checked = 0
    for i in range(len(piv)):
        node, hero, street = int(piv.node[i]), int(piv.hero[i]), int(piv.street[i])
        cards = piv.i[i, 6:6 + 3]
        # walk up to the root
        chain = []
        x_ = node
        while x_ in par:
            pn, a = par[x_]
            chain.append((pn, a))
            x_ = pn
        chain.reverse()
        hero_same_street = sum(1 for pn, _ in chain if g.node_actor[pn] == hero and g.node_street[pn] == street)
        if piv.n_street_nodes[i] == 2:
            n2 += 1
            firsts += hero_same_street == 0
        if checked < 4000:
            lp, lp0 = 0.0, 0.0
            for pn, a in chain:
                if g.node_actor[pn] != hero:
                    continue
                st = g.node_street[pn]
                pc = int(cards[hero]) // 2
                bucket = pc if st == 0 else pc * 3 + int(cards[2]) // 2
                base = g.node_row0[pn] + bucket * g.node_nA[pn]
                pa = probs[base + a]
                lp += math.log(0.75 * pa + 0.25 / g.node_nA[pn])
                lp0 += math.log(max(pa, 1e-300))
            lp_err = max(lp_err, abs(lp - piv.reach_lp[i]))
            lp_err_nomix = max(lp_err_nomix, abs(lp0 - piv.reach_lp[i]))
            checked += 1
    phat = firsts / n2
    se = math.sqrt(0.25 / n2)
    check("reservoir keeps first of two with prob 1/2", abs(phat - 0.5) < 4 * se, "n=%d p=%.4f se=%.4f" % (n2, phat, se))
    check("control: keep-first hypothesis rejected", abs(phat - 1.0) > 10 * se, "")
    check("reach_lp == product of eps-mixed hero probs on the path", lp_err < 1e-9, "max err=%.2e (%d pivots)" % (lp_err, checked))
    check("control: un-mixed product rejected", lp_err_nomix > 1e-3, "max err=%.3f" % lp_err_nomix)

    # 5
    p1 = probs
    p2 = h.probs_of(rng.normal(0, 1.0, g.entries))
    S = np.zeros(g.entries)
    g.avg_accumulate(p1, 1.0, S)
    g.avg_accumulate(p2, 3.0, S)
    native = g.normalise(S)
    # brute force: own reach per row, all deals
    reach1 = np.zeros(len(g.row_base))
    reach2 = np.zeros(len(g.row_base))
    D = 6
    import itertools
    deals = list(itertools.permutations(range(D), 3))
    for cards in deals:
        pr = 1.0 / len(deals)
        for hero in range(2):
            for pol, acc in ((p1, reach1), (p2, reach2)):
                stack = [(0, 1.0)]
                while stack:
                    nid, om = stack.pop()
                    if nid not in g.node_nA:
                        continue
                    nA, actor, st = g.node_nA[nid], g.node_actor[nid], g.node_street[nid]
                    pc = cards[actor] // 2
                    bucket = pc if st == 0 else pc * 3 + cards[2] // 2
                    base = g.node_row0[nid] + bucket * nA
                    ch = g.node_children[nid]
                    if actor == hero:
                        acc[np.searchsorted(g.row_base, base)] += pr * om
                        for a in range(nA):
                            if g.node_kinds[nid][a] != 0 and pol[base + a] > 0:
                                stack.append((int(ch[a]), om * pol[base + a]))
                    else:
                        for a in range(nA):
                            stack.append((int(ch[a]), om))
    r_e = g.entry_row
    brute = (1.0 * reach1[r_e] * p1 + 3.0 * reach2[r_e] * p2)
    tot = np.bincount(r_e, weights=brute, minlength=len(g.row_base))[r_e]
    brute = np.where(tot > 0, brute / np.where(tot > 0, tot, 1), g.uniform())
    check("Thm 9 accumulator == brute-force reach-weighted average", np.abs(native - brute).max() < 1e-9,
          "max diff=%.2e" % np.abs(native - brute).max())
    plain = (1.0 * p1 + 3.0 * p2) / 4.0
    check("control: unweighted average differs", np.abs(native - plain).max() > 1e-2,
          "max diff=%.3f" % np.abs(native - plain).max())

    # 6
    cfg = dict(L.defaults(), eta_reg=0.0, eta_unif=0.0, alpha_ent=0.0, beta_ref_fwd=0.0, isw="none",
               reg_outer=False, adv_clip=1e9, anchor_sum_clip=1e9, pivots=256, exact_q=True)
    hh = L.Harness(cfg)
    with __import__("torch").no_grad():
        hh.logits.copy_(__import__("torch").as_tensor(logits))
    out = hh.step(dry=True)
    B = len(out["piv"])
    k_ = out["valid"].sum(1, keepdims=True)
    expect = np.zeros(g.entries)
    np.add.at(expect, out["ent_idx"][out["valid"]], (-out["adv"] / (k_ * B))[out["valid"]])
    err = np.abs(out["grad"] - expect).max()
    check("logit gradient is NeuRD: -A_b/(|A|B)", err < 1e-12, "max err=%.2e" % err)
    cfg2 = dict(cfg, center=False, eta_unif=0.5, anchor_sum_clip=1.5, adv_clip=3.0)
    h2 = L.Harness(cfg2)
    with __import__("torch").no_grad():
        h2.logits.copy_(__import__("torch").as_tensor(logits))
    o2 = h2.step(dry=True)
    k2 = o2["valid"].sum(1, keepdims=True)
    e2 = np.zeros(g.entries)
    np.add.at(e2, o2["ent_idx"][o2["valid"]], (-o2["adv"] / (k2 * len(o2["piv"])))[o2["valid"]])
    check("control: uncentred advantages break the identity", np.abs(o2["grad"] - e2).max() > 1e-6,
          "max err=%.2e" % np.abs(o2["grad"] - e2).max())

    # 7 -- baseline defaults. The LAST iterate of unregularised NeuRD cycles in
    # zero-sum games, so the end-to-end claim is about the reach-weighted average.
    base = dict(L.defaults(), steps=400, report_every=400)
    _, cur = L.run(base, None, log=lambda *a: None)
    _, noise = L.run(dict(base, q_noise=True), None, log=lambda *a: None)
    u0 = g.exploitability(g.uniform())["nashconv"]
    check("end to end (baseline defaults): reach-weighted average NashConv falls", cur[-1]["reach_avg"] < 0.5 * u0,
          "uniform %.3f -> reach_avg %.3f, current %.3f after 400 steps" % (u0, cur[-1]["reach_avg"], cur[-1]["current"]))
    check("control: information-free advantages do not", noise[-1]["reach_avg"] > 0.8 * u0,
          "noise -> reach_avg %.3f" % noise[-1]["reach_avg"])
    last_sigma = cur[-1]["sigma"]
    check("sigma floor not binding at the default payoff scale", min(last_sigma) > 1.5,
          "sigma per street %s" % ["%.2f" % s for s in last_sigma])
    print("\n%d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
