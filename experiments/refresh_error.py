"""refresh_error.py -- how far is the sampled rule's iterate from the EXACT regularised fixed point of its own magnet?

Runs the sampled rule (lrc/harness.py) and, before every refresh (or every --every steps), solves the regularised game
for the harness's CURRENT magnet exactly (lrc/reference.py, warm-started at the iterate, same temperature: alpha_s =
eta_reg * sigma_ref_s / payoff_scale, plus the current uniform/entropy pull), then reports the iterate's log-prob
error against that fixed point (reach-weighted RMS by round), total-variation per row, the NashConv of the iterate,
of the exact fixed point and of the magnet. The exact reference tolerates little magnet error per refresh
(experiments/mmd_grid.py --grid rnoise); this measures how much the sampled rule actually carries.

    python experiments/refresh_error.py --arm Q_sgd_anneal --grid p4 --steps 8000
    python experiments/refresh_error.py --arm I_sgd_lr10.0 --grid inner --steps 16000 --every 1000
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402

import ladder as L  # noqa: E402
import lrc.reference as M  # noqa: E402
import lrc.harness as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="Q_sgd_anneal")
    ap.add_argument("--grid", default="p4")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--solve-steps", type=int, default=3000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=0, help="measure every N steps instead of before refreshes")
    a = ap.parse_args()
    cfg = H.defaults()
    cfg.update(L.GRIDS[a.grid][a.arm])
    cfg.update(steps=a.steps, seed=a.seed)
    h = H.Harness(cfg)
    G = M.FullLeduc(2)
    g = G.g
    K = cfg["refine_period"]
    fh = open(a.out, "w") if a.out else None
    for _ in range(a.steps):
        t_next = h.t + 1
        if a.every > 0:
            due = t_next % a.every == 0
        else:
            due = cfg["refine_rule"] == "fixed" and (t_next - h.last_refine) >= K
        if due and h.sigma_ref is not None:
            # state just before the step that refreshes: compare the iterate to the exact FP of the current magnet
            alpha = [cfg["eta_reg"] * s / cfg["payoff_scale"] for s in h.sigma_ref]
            pull = 1.0 if cfg["pull_anneal"] <= 0 else max(0.0, 1.0 - t_next / float(cfg["pull_anneal"]))
            tau = [(cfg["eta_unif"] + cfg["alpha_ent"]) * pull * s / cfg["payoff_scale"] for s in h.sigma_ref]
            p_it = h.probs_of(h.logits)
            rho = h.probs_of(h.reg_logits)

            def run_fp():
                # mmd() with a non-uniform fixed magnet: init at the iterate, magnet = rho via a 1-step refresh trick
                z = np.log(np.maximum(p_it, 1e-300))
                log_rho = np.log(np.maximum(rho, 1e-300))
                a_e, t_e = M.per_entry(G, alpha), M.per_entry(G, tau)
                log_u = np.log(g.uniform())
                for _k in range(a.solve_steps):
                    q = G.cond_q(G.values(np.exp(z)))
                    z = G.normalise_log((z + 0.03 * a_e * log_rho + 0.03 * t_e * log_u + 0.03 * q)
                                        / (1 + 0.03 * (a_e + t_e)))
                    z = G.normalise_log(np.maximum(z, -20.0))
                return np.exp(z)

            p_fp = run_fp()
            vals = G.values(p_fp)
            w = vals["own"] * vals["cfr"]
            err = np.log(np.maximum(p_it, 1e-12)) - np.log(np.maximum(p_fp, 1e-12))
            st = g.row_street[g.entry_row]
            rms = [float(np.sqrt((w[st == s] * err[st == s] ** 2).sum() / max(w[st == s].sum(), 1e-300)))
                   for s in range(2)]
            tv = 0.5 * np.bincount(g.entry_row, weights=np.abs(p_it - p_fp))
            rec = {"t": h.t, "rms_logerr": rms, "tv_mean": float(tv.mean()), "tv_max": float(tv.max()),
                   "nc_iterate": g.exploitability(p_it)["nashconv"], "nc_fp": g.exploitability(p_fp)["nashconv"],
                   "nc_magnet": g.exploitability(rho)["nashconv"], "pull": pull}
            print(json.dumps(rec), flush=True)
            if fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        h.step()


if __name__ == "__main__":
    main()
