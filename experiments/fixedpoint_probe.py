"""fixedpoint_probe.py -- is the regularised equilibrium a fixed point of the sampled rule's EXPECTED step?

Compute the exact equilibrium pi* of the regularised game (uniform magnet, temperature alpha_s = eta_reg * sigma_s /
payoff_scale per round, anchor at the decision only = the sampled rule's form), load it into the harness with the
magnet = uniform and sigma frozen so that the temperature matches, and average step(dry=True) gradients over N seeds.
At a fixed point of the expected update the mean gradient is 0 up to Monte-Carlo error; a systematic residual names
the rows where the expected step disagrees with the regularised game.

    python experiments/fixedpoint_probe.py --n 200
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lrc.reference as M  # noqa: E402
import lrc.harness as H  # noqa: E402


def probe(over, pi, n, sig, label):
    cfg = H.defaults()
    cfg.update(dict(eta_unif=0.0, alpha_ent=0.0, beta_ref_fwd=0.0, exact_q=True, adv_var_decay=1.0,
                    sigma_ref=list(sig), reg_outer=True, refine_rule="fixed", refine_period=10 ** 9,
                    eps_start=0.10, eps_floor=0.10))
    cfg.update(over)
    h = H.Harness(cfg)
    with torch.no_grad():
        h.logits.copy_(torch.as_tensor(np.log(np.maximum(pi, 1e-300))))
    h.reg_logits = torch.zeros_like(h.logits.detach())        # magnet = uniform
    h.running_adv_var = [s * s for s in sig]
    G = []
    for k in range(n):
        h.t = 1000 + k                                         # distinct counter-based seeds; eps at its floor
        G.append(h.step(dry=True)["grad"])
    G = np.stack(G)
    m, se = G.mean(0), G.std(0) / np.sqrt(n)
    z = np.where(se > 0, m / np.maximum(se, 1e-300), 0.0)
    g = h.g
    # per-row: residual norm and its z
    rows = g.entry_row
    rn = np.sqrt(np.bincount(rows, weights=m * m, minlength=len(g.row_base)))
    print("[%s] |mean grad| %.3e  max|z| %.1f  frac entries |z|>4: %.3f" % (
        label, np.linalg.norm(m), np.abs(z).max(), float((np.abs(z) > 4).mean())))
    return m, z, rn, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--eta-reg", type=float, default=0.2)
    a = ap.parse_args()
    sig = (4.0, 11.5)
    alpha = [a.eta_reg * s / 5.0 for s in sig]
    G = M.FullLeduc(2)
    pi, curve = M.mmd(G, steps=6000, eta=0.03, alpha=alpha, soft=False, magnet="fixed", zfloor=30.0, score_every=6000)
    print("reference QRE at alpha %s: NashConv %.4f" % (np.round(alpha, 3), curve[-1]["nashconv"]))
    # sanity: the reference's own residual at pi*
    vals = G.values(pi)
    q = G.cond_q(vals)
    print("uniform policy, same probe, for scale:")
    probe({}, G.g.uniform(), a.n, sig, "uniform pi, no clips? (baseline clips on)")
    m, z, rn, h = probe({}, pi, a.n, sig, "pi* baseline clips ON")
    noclip = dict(anchor_clip_z=1e9, anchor_sum_clip=1e9, adv_clip=1e9, ppo_clip=1e9)
    m2, z2, rn2, _ = probe(noclip, pi, a.n, sig, "pi* clips OFF")
    m3, z3, rn3, _ = probe(dict(noclip, center=False), pi, a.n, sig, "pi* clips OFF, no centring")
    m4, z4, rn4, _ = probe(dict(noclip, anchor_units="raw"), pi, a.n, sig, "pi* clips OFF, anchor raw units")
    g = h.g
    order = np.argsort(-rn2)[:8]
    print("worst rows (clips OFF): row street actor | residual | pi* row | z")
    for r in order:
        es = np.where(g.entry_row == r)[0]
        print("  %4d %d %d | %.2e | %s | %s" % (r, g.row_street[r], g.row_actor[r], rn2[r], np.round(pi[es], 4),
                                             np.round(z2[es], 1)))


if __name__ == "__main__":
    main()
