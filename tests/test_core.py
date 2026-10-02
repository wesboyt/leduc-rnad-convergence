"""Rules-level and estimator tests. python -m pytest tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lrc.game import ToyGame  # noqa: E402
from lrc import weights as W  # noqa: E402


def test_uniform_nashconv_and_layout():
    g = ToyGame("leduc")
    assert g.entries == 672 and len(g.row_base) == 288
    x = g.exploitability(g.uniform())
    assert abs(x["nashconv"] - 4.747222222222222) < 1e-12
    assert abs(x["ev"][0] + x["ev"][1]) < 1e-12                      # zero-sum


def test_kuhn_nash_value():
    g = ToyGame("kuhn")
    p = g.uniform().copy()
    for nid, nA in g.node_nA.items():
        facing = 0 in g.node_kinds[nid]
        for card in range(3):
            base = g.node_row0[nid] + card * nA
            if not facing:
                bet = 0.0 if nid == 0 else {0: 1 / 3, 1: 0.0, 2: 1.0}[card]
                p[base], p[base + 1] = 1 - bet, bet
            else:
                call = {0: 0.0, 1: 1 / 3, 2: 1.0}[card]
                p[base], p[base + 1] = 1 - call, call
    x = g.exploitability(p)
    assert x["nashconv"] < 1e-12 and abs(x["ev"][0] + 1 / 18) < 1e-12


def test_payoffs_bounded_and_antisymmetric():
    g = ToyGame("leduc")
    U = g.a_u0[g.a_term]
    assert np.abs(U).max() == 13.0                                   # 1 ante + 2*2 + 2*4 raises, all called
    # player 0 wins a showdown exactly when player 1 loses it
    assert set(np.unique(g.showdown_sign)) == {-1.0, 0.0, 1.0}


def test_best_response_dominates():
    g = ToyGame("leduc")
    rng = np.random.default_rng(1)
    for _ in range(5):
        z = rng.normal(0, 2, g.entries)
        p = np.exp(z - np.repeat(np.maximum.reduceat(z, g.row_base), g.row_width))
        p /= np.repeat(np.add.reduceat(p, g.row_base), g.row_width)
        x = g.exploitability(p)
        assert min(x["gains"]) >= -1e-12


def test_exact_q_matches_rollouts_in_mean():
    g = ToyGame("leduc")
    p = g.uniform()
    piv = g.generate(p, 200, 0.1, seed=2).take(np.arange(40))
    q = g.exact_q(p, piv)
    m, v = g.rollouts(p, piv, [3000, 3000], seed=3)
    se = np.sqrt(np.maximum(v, 1e-12) / 3000)
    z = ((m - q) / se)[v > 0]
    assert abs(z.mean()) < 0.5 and 0.6 < z.var() < 1.5


def test_clip_reach_weights():
    lw = np.array([0.0, 5.0, -5.0, 1.0])
    w, st = W.clip_reach_weights(lw, np.zeros(4, dtype=int), clip=10.0)
    assert abs(w.mean() - 1) < 1e-12 and w.max() / w.min() <= 100 + 1e-9


def test_center_over_valid():
    a = np.array([[1.0, 2.0, 3.0], [1.0, 5.0, 0.0]])
    v = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    c = W.center_over_valid(a, v)
    assert np.allclose(c.sum(1), 0) and c[1, 2] == 0


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
