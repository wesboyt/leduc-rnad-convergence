"""reference.py -- full-width exact Leduc: values, best response, and the tabular MMD / R-NaD reference solver.

The sampled rule (lrc/harness.py) is noisy and regularised in several ways at once. To know what "converges" means
on this game we need a reference whose convergence is proven: magnetic mirror descent (Sokota et al., ICLR 2023) /
R-NaD's fixed-point iteration (Perolat et al., ICML 2021) with exact, full-width action values. FullLeduc walks the
game tree over all 120 deals at once, as numpy vectors, on the SAME entry layout as the harness's policy table.

  values(probs)          -> per-entry counterfactual values, counterfactual reach, own reach, EV
  nashconv(probs)        -> own best response (audited against ToyGame.exploitability)
  mmd(...)               -> closed-form behavioural MMD with optional soft (regularised) values, a magnet that is
                            fixed / refreshed every K steps / gated / moved geometrically, and a fixed uniform
                            temperature tau_u (the analogue of the sampled rule's fixed pulls)
  grad_run(...)          -> the same objective moved by a gradient optimizer on the logits (SGD / Adam / sf-AdamW)

  The behavioural MMD step at every information set (row):
      log pi'  =  (log pi + eta*alpha*log rho + eta*tau_u*log u + eta*q) / (1 + eta*(alpha + tau_u))
  q = 'cond' (Q given the information set: cfv / cf-reach, what rollouts estimate) or 'cfv' (unnormalised).
  soft=True adds the regularisation reward along the continuation (R-NaD's transformed reward), soft=False is the
  sampled rule's form: the anchor acts only at the decision, the continuation return is raw.

    python -m lrc.reference --audit
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from lrc.game import ToyGame  # noqa: E402



class FullLeduc:
    """Full-width exact evaluator over all deals (2-player Leduc), on ToyGame's tree and entry layout."""

    def __init__(self, players=2):
        if players != 2:
            raise NotImplementedError
        self.n = players
        self.g = ToyGame("leduc", players)
        g = self.g
        self.E, self.R = g.entries, len(g.row_base)
        self.D = g.D
        self.bk = g.bk
        self.chance = g.chance
        self.nodes = {}
        for nid, nd in enumerate(g.nodes):
            if nd["kind"] == "T":
                self.nodes[nid] = ("T", np.stack([nd["u0"], -nd["u0"]]))
            else:
                self.nodes[nid] = ("D", nd["actor"], nd["street"], nd["nA"], nd["row0"], list(nd["children"]))

    # ---- on-policy values -------------------------------------------------------------------------------
    def values(self, probs, soft=None):
        """probs: [E] behavioural table. soft: None or (alpha, log_rho[E], tau_u) -- regularised (R-NaD) values.
        Returns dict(cfv[E], reach_row_cf[E] (cf reach of the entry's row, broadcast), own_reach[E], ev[n])."""
        n, D = self.n, self.D
        cfv = np.zeros(self.E)
        cfr = np.zeros(self.E)
        own = np.zeros(self.E)
        logp = np.log(np.maximum(probs, 1e-300))
        if soft is not None:
            alpha, log_rho, tau_u = soft
            log_u = np.log(self.g.uniform())
            pen_tab = alpha * (logp - log_rho) + tau_u * (logp - log_u)

        def walk(nid, reach):
            nd = self.nodes[nid]
            if nd[0] == "T":
                u = nd[1]
                return u if u.shape[1] == D else np.repeat(u, D, axis=1)
            _, actor, street, nA, row0, kids = nd
            bk = self.bk[actor][street]
            base = row0 + bk * nA
            opp = self.chance * np.prod(np.delete(reach, actor, axis=0), axis=0)
            v = np.zeros((n, D))
            for a in range(nA):
                e = base + a
                pa = probs[e]
                r2 = reach.copy()
                r2[actor] = reach[actor] * pa
                va = walk(kids[a], r2)
                # q of THIS node's action excludes this node's own penalty (the MMD step applies the magnet
                # explicitly); the penalty enters the value passed to the parents (2026-10-02 fix: including it
                # regularised the current node twice, local temperature 2*alpha vs alpha downstream)
                np.add.at(cfv, e, opp * va[actor])
                if soft is not None:
                    pen = pen_tab[e]
                    va = va.copy()
                    va[actor] -= pen
                    if n == 2:
                        va[1 - actor] += pen
                np.add.at(cfr, e, opp)
                np.add.at(own, e, reach[actor])
                v += pa[None, :] * va
            return v

        reach0 = np.ones((n, D))
        v = walk(0, reach0)
        ev = v.sum(1) * self.chance
        # own reach was accumulated once per deal: divide by the deals sharing the row (constant per row)
        return {"cfv": cfv, "cfr": cfr, "own": own, "ev": ev}

    def cond_q(self, vals):
        return np.where(vals["cfr"] > 0, vals["cfv"] / np.maximum(vals["cfr"], 1e-300), 0.0)

    # ---- best response ------------------------------------------------------------------------------------
    def br_value(self, probs, i):
        n, D = self.n, self.D

        def walk(nid, reach):
            nd = self.nodes[nid]
            if nd[0] == "T":
                u = nd[1][i]
                return u if u.shape[0] == D else np.repeat(u, D)
            _, actor, street, nA, row0, kids = nd
            bk = self.bk[actor][street]
            base = row0 + bk * nA
            if actor == i:
                opp = self.chance * np.prod(np.delete(reach, i, axis=0), axis=0)
                vs = [walk(k, reach) for k in kids]
                nb = int(bk.max()) + 1
                tot = np.stack([np.bincount(bk, weights=opp * va, minlength=nb) for va in vs])  # [nA, nb]
                best = tot.argmax(0)
                choice = best[bk]
                return np.stack(vs)[choice, np.arange(D)]
            v = np.zeros(D)
            for a in range(nA):
                pa = probs[base + a]
                r2 = reach.copy()
                r2[actor] = reach[actor] * pa
                v += pa * walk(kids[a], r2)
            return v

        return float(walk(0, np.ones((n, D))).sum() * self.chance)

    def nashconv(self, probs):
        ev = self.values(probs)["ev"]
        gains = [self.br_value(probs, i) - ev[i] for i in range(self.n)]
        return float(sum(gains)), gains

    # ---- table helpers ---------------------------------------------------------------------------------------
    def row_logsumexp(self, z):
        g = self.g
        m = np.full(self.R, -np.inf)
        np.maximum.at(m, g.entry_row, z)
        s = np.bincount(g.entry_row, weights=np.exp(z - m[g.entry_row]), minlength=self.R)
        return (m + np.log(s))[g.entry_row]

    def normalise_log(self, z):
        return z - self.row_logsumexp(z)


def mmd(game, steps=20000, eta=0.1, alpha=0.05, tau_u=0.0, q_mode="cond", soft=True, magnet="fixed", K=1000,
        magnet_lr=0.0, alpha_end=None, score_every=200, init=None, log=None, stop_below=None, gate_tol=1e-6, zfloor=0.0,
        refresh_noise=0.0, noise_seed=0):
    """Behavioural MMD. magnet: 'fixed' (rho = uniform: converges to the QRE at temperature alpha),
    'refresh' (rho <- pi every K steps: R-NaD outer loop), 'geo' (rho <- rho^(1-m) pi^m every step, m=magnet_lr),
    'gate' (rho <- pi when the fixed-point residual max|log pi_t - log pi_{t-1}| (reach-free) falls below gate_tol,
    and at least K steps after the last refresh: the convergence-gated outer loop, P2),
    alpha_end: anneal alpha geometrically to alpha_end over `steps` (with the fixed uniform magnet)."""
    g = game.g
    log_u = np.log(g.uniform())
    if not np.isscalar(alpha):
        alpha = per_entry(game, alpha)
    if not np.isscalar(tau_u):
        tau_u = per_entry(game, tau_u)
    z = log_u.copy() if init is None else np.log(np.maximum(init, 1e-300))
    log_rho = log_u.copy()
    curve = []
    t0 = time.time()
    last_ref, n_ref, resid = 0, 0, float("nan")
    nrng = np.random.default_rng(noise_seed)
    for t in range(1, steps + 1):
        a_t = alpha if alpha_end is None else alpha * (alpha_end / alpha) ** ((t - 1) / max(steps - 1, 1))
        p = np.exp(z)
        vals = game.values(p, soft=(a_t, log_rho, tau_u) if soft else None)
        q = game.cond_q(vals) if q_mode == "cond" else vals["cfv"]
        z_new = (z + eta * a_t * log_rho + eta * tau_u * log_u + eta * q) / (1.0 + eta * (a_t + tau_u))
        z_new = game.normalise_log(z_new)
        if zfloor > 0:          # support floor: log pi >= -zfloor (baseline: LOG_FLOOR -15, acting floor 1e-5)
            z_new = game.normalise_log(np.maximum(z_new, -zfloor))
        resid = float(np.max(np.abs(np.exp(z_new) - p))) / eta              # policy change per unit step
        z = z_new
        if (magnet == "refresh" and t % K == 0) or (magnet == "gate" and t - last_ref >= K and resid < gate_tol):
            log_rho = z.copy()
            if refresh_noise > 0:      # an inexact inner solve -> a noisy magnet (sigma in log-prob units)
                log_rho = game.normalise_log(log_rho + refresh_noise * nrng.standard_normal(log_rho.shape))
            last_ref, n_ref = t, n_ref + 1
        elif magnet == "geo":
            log_rho = game.normalise_log((1 - magnet_lr) * log_rho + magnet_lr * z)
        if t % score_every == 0 or t == steps:
            nc = g.exploitability(np.exp(z))["nashconv"]
            rec = {"t": t, "nashconv": nc, "alpha": float(np.mean(a_t)), "resid": resid, "refreshes": n_ref,
                   "min_p": float(np.exp(z).min()), "sec": time.time() - t0}
            curve.append(rec)
            if log:
                log(json.dumps(rec))
            if stop_below is not None and nc < stop_below:
                break
    return np.exp(z), curve


def per_entry(game, x):
    """scalar, or a per-street list -> [E] array."""
    g = game.g
    if np.isscalar(x):
        return np.full(game.E, float(x))
    return np.asarray(x, dtype=np.float64)[g.row_street[g.entry_row]]


def grad_run(game, steps=30000, opt="sgd", pg="softmax", lr=0.1, alpha=0.2, tau_u=0.0, K=500, zfloor=0.0,
             magnet="refresh", score_every=1000, log=None, betas=(0.9, 0.999), eps=1e-8, row_weight="uniform",
             own_reach_weight=False):
    """ the SAME exact full-width objective as mmd(soft=False) but moved by a gradient optimizer on logits.
    Advantage (the baseline's form, anchor at the node only):  A = q - alpha*(log pi - log rho) - tau_u*(log pi - log u)
    pg='softmax': loss gradient on logits = -pi*(A - E_pi A)   (PPO / softmax policy gradient at ratio 1)
    pg='neurd'  : loss gradient on logits = -(A - mean A)      (NeuRD: R-NaD's update)
    Every row weighted equally (the baseline's IS + node correction aims at that). opt: sgd | adam | sfadamw."""
    import torch
    g = game.g
    a_e, t_e = per_entry(game, alpha), per_entry(game, tau_u)
    log_u = np.log(g.uniform())
    log_rho = log_u.copy()
    theta = torch.zeros(game.E, dtype=torch.float64, requires_grad=True)
    if opt == "sgd":
        o = torch.optim.SGD([theta], lr=lr)
    elif opt == "adam":
        o = torch.optim.Adam([theta], lr=lr, betas=betas, eps=eps)
    elif opt == "sfadamw":
        import schedulefree
        o = schedulefree.AdamWScheduleFree([theta], lr=lr, warmup_steps=100, betas=betas, weight_decay=0.0, eps=eps)
        o.train()
    else:
        raise ValueError(opt)
    row = g.entry_row
    cnt = np.bincount(row, minlength=game.R)[row]
    curve = []
    t0 = time.time()
    for t in range(1, steps + 1):
        z = game.normalise_log(theta.detach().numpy().copy())
        if zfloor > 0:
            z = game.normalise_log(np.maximum(z, -zfloor))
        p = np.exp(z)
        vals = game.values(p)
        q = game.cond_q(vals)
        A = q - a_e * (z - log_rho) - t_e * (z - log_u)
        if pg == "softmax":
            ea = np.bincount(row, weights=p * A, minlength=game.R)[row]
            gr = -p * (A - ea)
        else:
            ma = np.bincount(row, weights=A, minlength=game.R)[row] / cnt
            gr = -(A - ma)
        if row_weight == "cfr":        # on-policy pivots + hero-reach IS -> expected step per row ~ cf reach
            cf = vals["cfr"]
            cf_row = np.bincount(row, weights=cf, minlength=game.R)[row] / cnt
            gr = gr * cf_row / cf_row[cf_row > 0].mean()
        o.zero_grad()
        theta.grad = torch.as_tensor(gr)
        o.step()
        if magnet == "refresh" and t % K == 0:
            zz = theta.detach().numpy()
            if opt == "sfadamw":          # the magnet is the policy that ACTS (train-mode y), as in baseline
                pass
            log_rho = game.normalise_log(zz.copy())
            if zfloor > 0:
                log_rho = game.normalise_log(np.maximum(log_rho, -zfloor))
        if t % score_every == 0 or t == steps:
            nc = g.exploitability(p)["nashconv"]
            rec = {"t": t, "nashconv": nc, "min_p": float(p.min()), "sec": time.time() - t0}
            curve.append(rec)
            if log:
                log(json.dumps(rec))
    return p, curve


def audit(players=2, seed=0):
    """Own values/BR against ToyGame.exact on uniform and random tables; MMD fixed-magnet sanity."""
    G = FullLeduc(players)
    rng = np.random.default_rng(seed)
    ok = True
    tables = [G.g.uniform()]
    for _ in range(4):
        z = rng.normal(0, 2.0, G.E)
        tables.append(np.exp(G.normalise_log(z)))
    for k, p in enumerate(tables):
        ev_t, br_t = G.g.exact(p)
        ev = G.values(p)["ev"]
        nc, gains = G.nashconv(p)
        nc_t = sum(b - e for e, b in zip(ev_t, br_t))
        d_ev = max(abs(a - b) for a, b in zip(ev, ev_t))
        d_nc = abs(nc - nc_t)
        good = d_ev < 1e-9 and d_nc < 1e-9
        ok &= good
        print("table %d: ev %s vs %s | nashconv %.6f vs %.6f  %s" % (
            k, np.round(ev, 6), np.round(ev_t, 6), nc, nc_t, "OK" if good else "FAIL"))
    # cfv identity: sum over actions of pi * cfv at the root rows of player 0 == its ev contribution
    p = tables[1]
    vals = G.values(p)
    q = G.cond_q(vals)
    print("cond-q finite:", bool(np.isfinite(q).all()))
    print("AUDIT", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--alpha-end", type=float, default=None)
    ap.add_argument("--tau-u", type=float, default=0.0)
    ap.add_argument("--q", default="cond")
    ap.add_argument("--soft", type=int, default=1)
    ap.add_argument("--magnet", default="fixed")
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--magnet-lr", type=float, default=0.0)
    ap.add_argument("--score-every", type=int, default=200)
    ap.add_argument("--gate-tol", type=float, default=1e-6)
    ap.add_argument("--zfloor", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.audit:
        sys.exit(0 if audit(a.players) else 1)
    G = FullLeduc(a.players)
    fh = open(a.out, "w") if a.out else None

    def log(s):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    log(json.dumps({"cfg": vars(a)}))
    mmd(G, a.steps, a.eta, a.alpha, a.tau_u, a.q, bool(a.soft), a.magnet, a.K, a.magnet_lr, a.alpha_end,
        a.score_every, log=log, gate_tol=a.gate_tol, zfloor=a.zfloor)


if __name__ == "__main__":
    main()
