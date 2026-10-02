"""sf_linear.py -- the mechanism in its simplest form: optimizers on the linearised regularised game.

Near any equilibrium of a regularised two-player zero-sum game the simultaneous-gradient field is, to first order,

    F(w) = (lambda I + omega J) w,        J = [[0, 1], [-1, 0]],  w = (x, y) in R^{2d}

lambda > 0 = the contraction contributed by the regulariser (the magnet), omega = the rotation contributed by the game
(omega = 0 is plain strongly convex MINIMISATION, the regime schedule-free optimizers were designed and validated in;
omega >> lambda is a rotation-dominated GAME). The equilibrium is w* = 0. Each optimizer receives F(w) as its
"gradient" (descent for x, ascent for y is already folded into F) and we report |w_T| / |w_0|.

Plain SGD with step gamma converges iff |1 - gamma (lambda +- i omega)| < 1, i.e. gamma < 2 lambda / (lambda^2 + omega^2).
The question: at a step where SGD converges for EVERY omega tested, which optimizers keep converging as omega grows?

    python experiments/sf_linear.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lrc.policy import make_optimizer  # noqa: E402

OPTS = {"sgd": ("sgd", (0.9, 0.999)), "adam_b1_0": ("adam", (0.0, 0.999)), "adam_b1_0.9": ("adam", (0.9, 0.999)),
        "sfsgd_b0.9": ("sfsgd", (0.9, 0.999)), "sfadamw_b0.9": ("sfadamw", (0.9, 0.999))}


def run(opt, omega, lam=1.0, gamma=0.005, steps=20000, d=8, seed=0, warmup=0):
    torch.manual_seed(seed)
    w = torch.randn(2 * d, dtype=torch.float64)
    w0 = float(w.norm())
    p = torch.nn.Parameter(w.clone())
    name, betas = OPTS[opt]
    o = make_optimizer([p], name, gamma, betas=betas, warmup=warmup)
    for _ in range(steps):
        with torch.no_grad():
            x, y = p[:d], p[d:]
            g = torch.cat([lam * x + omega * y, -omega * x + lam * y])
        o.zero_grad()
        p.grad = g
        o.step()
        if not torch.isfinite(p).all():
            return float("inf")
    return float(p.detach().norm()) / w0


def grid(omegas=(0.0, 1.0, 10.0, 30.0), gamma=0.005, d=8, seed=0, steps=20000):
    out = {}
    for opt in OPTS:
        out[opt] = {str(om): run(opt, om, gamma=gamma, d=d, seed=seed, steps=steps) for om in omegas}
    return out


if __name__ == "__main__":
    g = 0.005
    print("SGD stability bound gamma < 2 lam/(lam^2+omega^2): " +
          ", ".join("omega %g: %.4g" % (om, 2 / (1 + om * om)) for om in (0, 1, 10, 30)) + "; gamma = %g" % g)
    res = grid(gamma=g)
    for opt, r in res.items():
        print("%-14s " % opt + "  ".join("omega=%s: %.3e" % (k, v) for k, v in r.items()))
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"), exist_ok=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                                     "sf_linear.json"), "w"), indent=1)
