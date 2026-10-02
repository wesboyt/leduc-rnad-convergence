"""ladder.py -- grids of the SAMPLED RULE (lrc/harness.py) on 2-player Leduc with EXACT NashConv.

Start from an idealised R-NaD that should converge, add the baseline's components one at a time, then vary the
optimizer, the refresh schedule, the estimator and the magnet. See docs/CLAIMS.md for which grid backs which claim.

    python experiments/ladder.py --grid p4 --steps 16000 --seeds 1,2 --jobs 4
    python experiments/ladder.py --grid p4 --summary
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), 'results', 'ladder')

IDEAL = dict(exact_q=True, eta_unif=0.0, alpha_ent=0.0, beta_ref_fwd=0.0, eta_reg=0.5, lr=0.1,
             refine_rule='fixed', refine_period=2000, anchor_clip_z=1e9, anchor_sum_clip=1e9, adv_clip=1e9,
             ppo_clip=1e9)
PROD = dict(eta_reg=0.20, eta_unif=0.05, alpha_ent=0.04, beta_ref_fwd=0.08, ref_target='reg',
            reg_min_period=500, refine_rule='plateau')          # baseline (PLATEAU_RESET, REG_MIN_PERIOD 500)
ARMS = {
    # L0: an idealised R-NaD (exact Q, no fixed regularisers, no clips, long fixed refine): can the class converge?
    'L0_ideal': IDEAL,
    # L1: the baseline's update RULE with exact Q
    'L1_prod_exactQ': dict(PROD, exact_q=True),
    # L2: L1 without the FIXED regularisers toward uniform (eta_unif, alpha_ent): the QRE floor
    'L2_prod_exactQ_noUnifEnt': dict(PROD, exact_q=True, eta_unif=0.0, alpha_ent=0.0),
    # L3: L1 with refine every 2000 instead of the plateau rule at 500: the refine schedule
    'L3_prod_exactQ_refine2000': dict(PROD, exact_q=True, refine_rule='fixed', refine_period=2000),
    # L4: L1 with the baseline's anchor/advantage/PPO clips removed: the clips
    'L4_prod_exactQ_noClips': dict(PROD, exact_q=True, anchor_clip_z=1e9, anchor_sum_clip=1e9, adv_clip=1e9, ppo_clip=1e9),
    # L5: baseline rule + SAMPLED Q (the baseline's estimator)
    'L5_prod_sampledQ': dict(PROD),
    # L6: L5 + the always-binding norm clip (direction-only steps)
    'L6_prod_sampledQ_gradnorm': dict(PROD, grad_norm=0.05),
    # L7: the ideal rule with SAMPLED Q: does noise alone stop the ideal rule?
    'L7_ideal_sampledQ': dict(IDEAL, exact_q=False),
}


# the DYNAMICS grid -- every ladder arm shared schedule-free Adam. Exact Q, no fixed regularisers, no clips, no
# ref anchor, refine every 2000: optimizer x step size x magnet strength.
_BASE = dict(exact_q=True, eta_unif=0.0, alpha_ent=0.0, beta_ref_fwd=0.0, anchor_clip_z=1e9, anchor_sum_clip=1e9,
             adv_clip=1e9, ppo_clip=1e9, refine_rule='fixed', refine_period=2000)
GRIDS = {'ladder': ARMS, 'dyn': {}, 'prod_eta': {}}
# the FULL baseline rule (clamps, PPO, plateau refine from 500, fixed regularisers, ref -> pi_reg) at stronger magnets
for _eta in (0.5, 1.0):
    GRIDS['prod_eta'][f"P_prod_exactQ_eta{_eta}"] = dict(PROD, exact_q=True, eta_reg=_eta)
    GRIDS['prod_eta'][f"P_prod_sampledQ_eta{_eta}"] = dict(PROD, eta_reg=_eta)
    GRIDS['prod_eta'][f"P_prod_sampledQ_eta{_eta}_gradnorm"] = dict(PROD, eta_reg=_eta, grad_norm=0.05)
GRIDS['prod_eta']['P_prod_sampledQ_eta1.0_clampOff'] = dict(PROD, eta_reg=1.0, anchor_clip_z=1e9, anchor_sum_clip=1e9)
# the REFINE schedule under the full baseline rule at eta 1.0, sampled Q (the baseline's estimator)
GRIDS['prod_ref'] = {
    'R_eta1.0_plateau500': dict(PROD, eta_reg=1.0),                                          # = baseline schedule
    'R_eta1.0_plateau2000': dict(PROD, eta_reg=1.0, reg_min_period=2000),
    'R_eta1.0_fixed2000': dict(PROD, eta_reg=1.0, refine_rule='fixed', refine_period=2000),
    'R_eta1.0_fixed4000': dict(PROD, eta_reg=1.0, refine_rule='fixed', refine_period=4000),
    'R_eta1.0_fixed2000_noUnifEnt': dict(PROD, eta_reg=1.0, refine_rule='fixed', refine_period=2000,
                                         eta_unif=0.0, alpha_ent=0.0),
    'R_eta0.2_fixed4000': dict(PROD, eta_reg=0.2, refine_rule='fixed', refine_period=4000),   # magnet alone
}
for _eta in (0.2, 1.0):
    for _lr in (0.003, 0.01, 0.03, 0.1):
        GRIDS['dyn'][f"D_adam_eta{_eta}_lr{_lr}"] = dict(_BASE, optimizer='sfadamw', eta_reg=_eta, lr=_lr)
    for _lr in (1.0, 10.0, 100.0):
        GRIDS['dyn'][f"D_sgd_eta{_eta}_lr{_lr}"] = dict(_BASE, optimizer='sgd', eta_reg=_eta, lr=_lr)

# the exact reference's fixes carried into the baseline's SAMPLED step (PPO, clips, IS, sigma units).
# Base = the full baseline rule at eta_reg 0.2 (= alpha 0.16/0.46 chips, the reference's best), refine fixed every 500.
_P4 = dict(PROD, refine_rule='fixed', refine_period=500)
GRIDS['p4'] = {
    'Q_sfadam_b0.9_fixedpulls': dict(_P4),                                                   # baseline optimizer
    'Q_adam_b0_fixedpulls': dict(_P4, optimizer='adam', beta1=0.0, lr=0.003),                # floor 0.109 expected
    'Q_adam_b0_anneal': dict(_P4, optimizer='adam', beta1=0.0, lr=0.003, pull_anneal=4000),  # the candidate rule
    'Q_adam_b0.9_anneal': dict(_P4, optimizer='adam', beta1=0.9, lr=0.003, pull_anneal=4000),  # momentum back
    'Q_sfadam_b0.9_anneal': dict(_P4, pull_anneal=4000),                                      # baseline opt + anneal
    'Q_sgd_anneal': dict(_P4, optimizer='sgd', lr=10.0, pull_anneal=4000),                   # SGD lr of the dynamics grid
}
# SGD is the only stable sampled rule (0.33, reach-avg 0.18). Noise vs baseline components; levers.
_S = dict(_P4, optimizer='sgd', lr=10.0, pull_anneal=4000)
GRIDS['p4b'] = {
    'Q_sgd_exactQ': dict(_S, exact_q=True),                 # baseline components without sampling noise
    'Q_sgd_lr3': dict(_S, lr=3.0),
    'Q_sgd_lr30': dict(_S, lr=30.0),
    'Q_sgd_piv2048': dict(_S, pivots=2048),                 # 4x batch
    'Q_sgd_K2000': dict(_S, refine_period=2000),
    'Q_adam_b0_lr0.001': dict(_P4, optimizer='adam', beta1=0.0, lr=0.001, pull_anneal=4000),
}
# p4b showed noise is NOT the floor (exact Q 0.38 = sampled 0.33). Baseline components, exact Q,
# SGD, removed one at a time.
_C = dict(_S, exact_q=True)
_NOCLIP = dict(anchor_clip_z=1e9, anchor_sum_clip=1e9, adv_clip=1e9, ppo_clip=1e9)
GRIDS['comp'] = {
    'C_noClips': dict(_C, **_NOCLIP),
    'C_noFwdKL': dict(_C, beta_ref_fwd=0.0),
    'C_noEps': dict(_C, eps_start=0.0, eps_floor=0.0),
    'C_isNone': dict(_C, isw='none', node_correct=False, reach_reweight=False),
    'C_noClips_noFwdKL': dict(_C, beta_ref_fwd=0.0, **_NOCLIP),
    'C_all': dict(_C, beta_ref_fwd=0.0, eps_start=0.0, eps_floor=0.0, **_NOCLIP),
}

# the support floor in the baseline's sampled step (exact Q and sampled Q), SGD
# the INNER loop rate of the baseline's step: fixed uniform magnet, no refine, exact Q, no pulls, clips off.
# The exact QRE at this temperature is NashConv 0.2468 (leduc_fixedpoint_probe.py); steps to reach it = inner time.
_IN = dict(_C, pull_anneal=0, eta_unif=0.0, alpha_ent=0.0, beta_ref_fwd=0.0, refine_rule='fixed',
           refine_period=10 ** 9, reg_max_period=10 ** 9, **_NOCLIP)
GRIDS['inner'] = {f'I_sgd_lr{_lr}': dict(_IN, lr=_lr) for _lr in (10.0, 30.0, 100.0, 300.0)}
GRIDS['inner']['I_sgd_lr30_clipsOn'] = dict(_IN, lr=30.0, anchor_clip_z=1.0, anchor_sum_clip=1.5, adv_clip=3.0,
                                           ppo_clip=0.3)
GRIDS['floor'] = {
    'F_sgd_exactQ_floor20': dict(_C, logit_floor=20.0),
    'F_sgd_exactQ_floor10': dict(_C, logit_floor=10.0),
    'F_sgd_sampledQ_floor20': dict(_S, logit_floor=20.0),
    'F_sgd_sampledQ_floor10': dict(_S, logit_floor=10.0),
    'F_adam_b0_sampledQ_floor20': dict(_P4, optimizer='adam', beta1=0.0, lr=0.003, pull_anneal=4000, logit_floor=20.0),
    'F_sfadam_sampledQ_floor20': dict(_P4, pull_anneal=4000, logit_floor=20.0),
}
# noisy inner loop -> averaged magnet and/or inner lr decay (the baseline's full sampled step, SGD)
GRIDS['avg'] = {
    'A_sgd_magmean': dict(_S, magnet_avg='mean'),
    'A_sgd_lrdecay100': dict(_S, inner_lr_tau=100.0),
    'A_sgd_both': dict(_S, magnet_avg='mean', inner_lr_tau=100.0),
    'A_sgd_both_K1000': dict(_S, magnet_avg='mean', inner_lr_tau=100.0, refine_period=1000),
    'A_sgd_both_exactQ': dict(_C, magnet_avg='mean', inner_lr_tau=100.0),
    'A_adam_b0_magmean': dict(_P4, optimizer='adam', beta1=0.0, lr=0.003, pull_anneal=4000, magnet_avg='mean'),
}

def run_arm(args):
    name, over, steps, seed = args
    import lrc.harness as H
    cfg = H.defaults()
    cfg.update(over)
    cfg.update(steps=steps, seed=seed, report_every=200)
    out = os.path.join(OUT, f"{name}_s{seed}")
    _, curve = H.run(cfg, out, log=lambda *a: None)
    cur = [r['current'] for r in curve]
    ra = [r['reach_avg'] for r in curve if r['reach_avg'] is not None]
    return name, seed, {'final_cur': cur[-1], 'min_cur': min(cur), 'final_reach_avg': ra[-1] if ra else None,
                        'final_avg_model': curve[-1]['avg_model'], 'refines': curve[-1]['refines'],
                        'cur_at': {r['t']: r['current'] for r in curve if r['t'] % 2000 == 0}}


GRID = 'ladder'


def summary():
    rows = {}
    for d in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, d, 'log.jsonl')
        if not os.path.exists(p):
            continue
        name, seed = d.rsplit('_s', 1)
        recs = [json.loads(l) for l in open(p)]
        rows.setdefault(name, []).append(recs)
    print(f"{'arm':30s} {'seeds':>5s} {'final cur':>10s} {'min cur':>9s} {'reach_avg':>10s} {'avg_model':>10s} {'refines':>8s}  cur @2k/4k/6k/8k")
    for name in GRIDS[GRID]:
        if name not in rows:
            continue
        R = rows[name]
        f = lambda key: sum(r[-1][key] for r in R) / len(R)
        mn = sum(min(x['current'] for x in r) for r in R) / len(R)
        ra = [r[-1]['reach_avg'] for r in R if r[-1]['reach_avg'] is not None]
        traj = []
        for t in (2000, 4000, 8000, 12000, 16000):
            v = [x['current'] for r in R for x in r if x['t'] == t]
            traj.append(f"{sum(v) / len(v):.3f}" if v else '-')
        print(f"{name:30s} {len(R):5d} {f('current'):10.4f} {mn:9.4f} {sum(ra) / len(ra) if ra else float('nan'):10.4f} "
              f"{f('avg_model'):10.4f} {sum(r[-1]['refines'] for r in R) / len(R):8.1f}  {' / '.join(traj)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--arms', default='')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--grid', default='ladder', choices=('ladder', 'dyn', 'prod_eta', 'prod_ref', 'p4', 'p4b', 'comp', 'floor', 'inner', 'avg'))
    a = ap.parse_args()
    GRID = a.grid
    if a.summary:
        summary()
        sys.exit(0)
    os.makedirs(OUT, exist_ok=True)
    names = [n for n in GRIDS[a.grid] if not a.arms or n in a.arms.split(',')]
    jobs = [(n, GRIDS[a.grid][n], a.steps, int(s)) for n in names for s in a.seeds.split(',')]
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, seed, res in ex.map(run_arm, jobs):
            print(f"{name} s{seed}: final cur {res['final_cur']:.4f} min {res['min_cur']:.4f} "
                  f"reach_avg {res['final_reach_avg']} refines {res['refines']}", flush=True)
    summary()
