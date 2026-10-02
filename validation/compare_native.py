"""compare_native.py -- lrc/game.py (pure numpy) against the native validation-game library the experiments were
first run on. Exact quantities must agree to 1e-9; sampled ones within Monte-Carlo error.

This script is kept for provenance. It needs that native library, which is NOT part of this repository (it belongs to
a larger solver); the recorded output is in docs/VALIDATION.md. Nothing else in the repository depends on it.
"""
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.environ.get("NATIVE_LIB_SRC", ""))      # path to the native library's python package
import numpy as np  # noqa: E402

from donksolver.harness import ToyGame as NativeGame, Pivots as NPivots  # noqa: E402
from lrc.game import ToyGame as PureGame, Pivots as PPivots  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def main():
    rng = np.random.default_rng(0)
    for gname in ("kuhn", "leduc"):
        N, P = NativeGame(gname, 2), PureGame(gname, 2)
        check(f"{gname}: entries", N.entries == P.entries, f"{N.entries} vs {P.entries}")
        same_layout = (np.array_equal(N.row_base, P.row_base) and np.array_equal(N.row_width, P.row_width)
                       and N.node_row0 == P.node_row0 and N.node_actor == P.node_actor
                       and N.node_street == P.node_street and N.node_kinds == P.node_kinds)
        check(f"{gname}: node ids, rows, actors, streets, action kinds identical", same_layout)
        errs = []
        for k in range(20):
            z = rng.normal(0, 2.0, N.entries)
            p = np.exp(z - np.repeat(np.maximum.reduceat(z, N.row_base), N.row_width))
            p /= np.repeat(np.add.reduceat(p, N.row_base), N.row_width)
            a, b = N.exploitability(p), P.exploitability(p)
            errs.append(max(abs(a["nashconv"] - b["nashconv"]), max(abs(x - y) for x, y in zip(a["ev"], b["ev"]))))
        check(f"{gname}: EV and NashConv on 20 random tables", max(errs) < 1e-9, "max err %.2e" % max(errs))
        S1, S2 = np.zeros(N.entries), np.zeros(N.entries)
        N.avg_accumulate(p, 2.5, S1)
        P.avg_accumulate(p, 2.5, S2)
        check(f"{gname}: Thm-9 accumulator", np.abs(S1 - S2).max() < 1e-12, "max err %.2e" % np.abs(S1 - S2).max())
    N, P = NativeGame("leduc", 2), PureGame("leduc", 2)
    z = rng.normal(0, 1.0, N.entries)
    p = np.exp(z - np.repeat(np.maximum.reduceat(z, N.row_base), N.row_width))
    p /= np.repeat(np.add.reduceat(p, N.row_base), N.row_width)
    # exact Q on the SAME (native) pivots
    npv = N.generate(p, 3000, 0.1, seed=3)
    ppv = PPivots(npv.i[:, :9], npv.reach_lp)
    qa, qb = N.exact_q(p, npv)[:, :3], P.exact_q(p, ppv)[:, :3]
    check("exact_q on identical pivots", np.abs(qa - qb).max() < 1e-9, "n=%d max err %.2e" % (len(npv), np.abs(qa - qb).max()))
    # rollouts: pure MC mean vs exact (z-test), and native vs pure variance ratio
    sel = np.arange(min(400, len(ppv)))
    sub = ppv.take(sel)
    m, v = P.rollouts(p, sub, [2000, 2000], seed=7)
    zs = []
    for k in range(len(sub)):
        for a in range(N.node_nA[int(sub.node[k])]):
            if v[k, a] > 0:
                zs.append((m[k, a] - qb[sel[k], a]) / math.sqrt(v[k, a] / 2000))
    zs = np.asarray(zs)
    check("pure rollouts unbiased for exact Q (resampled board)", abs(zs.mean()) < 4 / math.sqrt(len(zs)) * 1.5
          and 0.85 < zs.var() < 1.15, "cells %d mean z %.3f var z %.3f" % (len(zs), zs.mean(), zs.var()))
    mn, vn = N.rollouts(p, npv.take(sel), [2000, 2000], seed=7)
    ratio = v[:, :3][v[:, :3] > 0].mean() / vn[:, :3][vn[:, :3] > 0].mean()
    check("rollout sample variance native vs pure", abs(ratio - 1) < 0.05, "ratio %.4f" % ratio)
    m2, _ = P.rollouts(p, sub, [2000, 2000], seed=8, crn=True)
    check("crn rollouts run and agree with exact", np.abs(m2[:, :3] - qb[sel, :3]).max() < 1.0,
          "max |mean-exact| %.3f" % np.abs(m2[:, :3] - qb[sel, :3]).max())
    # generation statistics
    a = N.generate(p, 200000, 0.25, seed=11)
    b = P.generate(p, 200000, 0.25, seed=11)
    for name, fa, fb in (("pivots per hand", len(a) / 200000, len(b) / 200000),
                         ("street-1 share", (a.street == 1).mean(), (b.street == 1).mean()),
                         ("mean n_street_nodes", a.n_street_nodes.mean(), b.n_street_nodes.mean()),
                         ("mean reach_lp", a.reach_lp.mean(), b.reach_lp.mean())):
        check("generate: " + name, abs(fa - fb) < 0.01 * max(1.0, abs(fa)), "native %.4f pure %.4f" % (fa, fb))
    # pivot distribution over rows
    ra = np.bincount(N.row_of(a.node, a.bucket), minlength=N.entries)
    rb = np.bincount(P.row_of(b.node, b.bucket), minlength=N.entries)
    m_ = (ra + rb) > 0
    na_, nb_ = ra.sum(), rb.sum()
    e_a = (ra + rb)[m_] * na_ / (na_ + nb_)
    e_b = (ra + rb)[m_] * nb_ / (na_ + nb_)
    chi = float((((ra[m_] - e_a) ** 2) / e_a + ((rb[m_] - e_b) ** 2) / e_b).sum())
    df = int(m_.sum()) - 1
    zc = ((chi / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    check("generate: distribution of pivot rows (two-sample chi2)", abs(zc) < 4, "chi2/df %.3f df %d z %.2f" % (
        chi / df, df, zc))
    print("\n%d/%d checks passed" % (sum(RES), len(RES)))
    sys.exit(0 if all(RES) else 1)


if __name__ == "__main__":
    main()
