"""mmd_grid.py -- grids for the exact tabular MMD / R-NaD reference and exact-gradient learners (lrc/reference.py).
Deterministic (exact Q, full width): no seeds.

    python experiments/mmd_grid.py --grid outer --jobs 8
    python experiments/mmd_summary.py "results/mmd/outer_*.jsonl"
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), "results", "mmd")

GRIDS = {}
# outer-loop rate. alpha sets the proximal step (1/alpha) and the inner contraction (eta*alpha).
GRIDS["outer"] = {}
for _a in (0.05, 0.1, 0.2, 0.5):
    for _K in (200, 500, 1000):
        _eta = 0.03 if _a >= 0.1 else 0.01
        GRIDS["outer"][f"outer_a{_a}_e{_eta}_K{_K}"] = dict(steps=60000, eta=_eta, alpha=_a, magnet="refresh", K=_K)
# support floor (the outer grid lost support: min_p == 0 from early on, spikes to 0.6-3.6)
GRIDS["floor"] = {}
for _a, _K in ((0.2, 500), (0.2, 1000), (0.5, 200), (0.5, 500)):
    for _zf in (20.0, 40.0):
        GRIDS["floor"][f"floor_a{_a}_K{_K}_z{int(_zf)}"] = dict(steps=60000, eta=0.03, alpha=_a, magnet="refresh",
                                                                K=_K, zfloor=_zf)
# baseline-form (raw, local anchor only) vs soft values, same schedule
GRIDS["rawsoft"] = {}
for _a in (0.2, 0.5, 1.0):
    for _soft in (True, False):
        GRIDS["rawsoft"][f"rawsoft_a{_a}_soft{int(_soft)}_K500"] = dict(steps=60000, eta=0.03, alpha=_a, soft=_soft,
                                                                        magnet="refresh", K=500, zfloor=20.0)


# QRE floor of the baseline's fixed pulls, and the magnet at the baseline's eta_reg 0.2 in chips.
# Baseline (harness units, payoff_scale 5): sigma ~ 4 (street 0) / 11.5 (street 1); anchor in chips = eta*sigma/5.
# eta_reg 0.2 -> alpha (0.16, 0.46); eta_unif .05 + alpha_ent .04 -> tau_u (0.072, 0.21); eta_reg 1.0 -> (0.8, 2.3).
A02, TAU, A10 = [0.16, 0.46], [0.072, 0.21], [0.8, 2.3]
GRIDS["prodfloor"] = {
    "pf_qre_tau_fixedmagnet": dict(steps=20000, eta=0.03, alpha=0.0, tau_u=TAU, magnet="fixed", zfloor=20.0),
    "pf_eta0.2_K500": dict(steps=60000, eta=0.03, alpha=A02, magnet="refresh", K=500, zfloor=20.0),
    "pf_eta0.2_K500_tau": dict(steps=60000, eta=0.03, alpha=A02, tau_u=TAU, magnet="refresh", K=500, zfloor=20.0),
    "pf_eta1.0_K500": dict(steps=60000, eta=0.03, alpha=A10, magnet="refresh", K=500, zfloor=20.0),
    "pf_eta0.2_K2000": dict(steps=60000, eta=0.03, alpha=A02, magnet="refresh", K=2000, zfloor=20.0),
}
# the OPTIMIZER on the identical exact objective (the baseline's raw-Q form, eta_reg 0.2 in chips, K 500)
GRIDS["opt"] = {}
for _lr in (0.01, 0.03):
    GRIDS["opt"][f"opt_neurd_sgd_lr{_lr}"] = dict(fn="grad", opt="sgd", pg="neurd", lr=_lr)
for _lr in (0.1, 0.3, 1.0):
    GRIDS["opt"][f"opt_softmax_sgd_lr{_lr}"] = dict(fn="grad", opt="sgd", pg="softmax", lr=_lr)
for _lr in (0.003, 0.01, 0.03):
    GRIDS["opt"][f"opt_softmax_adam_lr{_lr}"] = dict(fn="grad", opt="adam", pg="softmax", lr=_lr)
    GRIDS["opt"][f"opt_softmax_sfadamw_lr{_lr}"] = dict(fn="grad", opt="sfadamw", pg="softmax", lr=_lr)
for _k, _v in GRIDS["opt"].items():
    _v.update(steps=30000, alpha=A02, K=500, zfloor=20.0)
# can an Adam variant recover mirror-descent behaviour? (baseline cannot easily drop Adam on a transformer)
GRIDS["adamv"] = {}
for _pg in ("softmax", "neurd"):
    for _eps in (1e-2, 1e-1, 1.0):
        for _lr in (0.01, 0.03, 0.1):
            GRIDS["adamv"][f"av_{_pg}_adam_eps{_eps}_lr{_lr}"] = dict(fn="grad", opt="adam", pg=_pg, lr=_lr, eps=_eps)
    GRIDS["adamv"][f"av_{_pg}_adam_b1_0_lr0.003"] = dict(fn="grad", opt="adam", pg=_pg, lr=0.003, betas=(0.0, 0.999))
    GRIDS["adamv"][f"av_{_pg}_sfadamw_eps0.1_lr0.03"] = dict(fn="grad", opt="sfadamw", pg=_pg, lr=0.03, eps=0.1)
GRIDS["adamv"]["av_neurd_adam_lr0.003"] = dict(fn="grad", opt="adam", pg="neurd", lr=0.003)
for _k, _v in GRIDS["adamv"].items():
    _v.update(steps=30000, alpha=A02, K=500, zfloor=20.0)
# MOMENTUM is the suspect (adamv: only beta1 = 0 converged, 0.044). beta1 x beta2 x lr, softmax PG.
GRIDS["mom"] = {}
for _b1 in (0.0, 0.5, 0.9):
    for _b2 in (0.99, 0.999):
        for _lr in (0.001, 0.003, 0.01):
            GRIDS["mom"][f"mom_b1{_b1}_b2{_b2}_lr{_lr}"] = dict(fn="grad", opt="adam", pg="softmax", lr=_lr,
                                                              betas=(_b1, _b2), steps=30000, alpha=A02, K=500,
                                                              zfloor=20.0)

# per-row step proportional to counterfactual reach (what the baseline's sampled step does in expectation)
GRIDS["cfw"] = {}
for _rw in ("uniform", "cfr"):
    for _lr in (0.03, 0.1, 0.3):
        for _K in (500, 2000):
            GRIDS["cfw"][f"cfw_{_rw}_neurd_sgd_lr{_lr}_K{_K}"] = dict(fn="grad", opt="sgd", pg="neurd", lr=_lr, K=_K,
                                                                    row_weight=_rw, steps=30000, alpha=A02,
                                                                    zfloor=20.0)
# how much magnet error per refresh does the outer loop tolerate? (exact mirror step, eta_reg-0.2 magnet)
GRIDS["rnoise"] = {}
for _sd in (0.0, 0.01, 0.03, 0.1, 0.3):
    for _K in (500, 2000):
        GRIDS["rnoise"][f"rn_sd{_sd}_K{_K}"] = dict(steps=40000, eta=0.03, alpha=A02, magnet="refresh", K=_K,
                                                   zfloor=20.0, refresh_noise=_sd)

def run_arm(args):
    name, kw, score_every = args
    import lrc.reference as M
    G = M.FullLeduc(2)
    path = os.path.join(OUT, name + ".jsonl")
    with open(path, "w") as fh:
        fh.write(json.dumps({"cfg": kw}) + "\n")

        def log(s):
            fh.write(s + "\n")
            fh.flush()

        kw = dict(kw)
        if kw.pop("fn", "mmd") == "grad":
            M.grad_run(G, score_every=score_every, log=log, **kw)
        else:
            M.mmd(G, score_every=score_every, log=log, **kw)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--score-every", type=int, default=1000)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    arms = [(n, kw, a.score_every) for n, kw in GRIDS[a.grid].items() if not a.only or a.only in n]
    with ProcessPoolExecutor(a.jobs) as ex:
        for n in ex.map(run_arm, arms):
            print("done", n, flush=True)


if __name__ == "__main__":
    main()
