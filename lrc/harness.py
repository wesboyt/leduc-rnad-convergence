"""harness.py -- the SAMPLED RULE on 2-player Leduc, scored by exact best response.

The sampled rule is a regularised policy-gradient method of the R-NaD family (docs/ALGORITHM.md, section 3):
on-policy hands, one pivot per hand, Monte-Carlo action values at the pivot, a magnet (pi_reg) that is refreshed
periodically, fixed regularisers toward uniform, clipped importance weights, a uniform-weight PPO surrogate and a
first-order optimizer on a TABLE of logits (one row per information set). Every policy it produces is scored by
exact NashConv (all deals enumerated; lrc/game.py).

Step (defaults = the baseline configuration, docs/ALGORITHM.md table 1)
  generation   every seat plays the current policy; hero seat uniform; hero-only eps exploration (0.25 -> 0.10 over
               3000 steps); reservoir-of-one pivot per street; reach = hero's eps-mixed log-probability before the pivot
  collector    static per-street quota; the street is drawn before a hand is inspected; one pivot per hand
  action value per legal action, n_sims continuations (32 round 1, 16 round 2); private cards kept, the undealt board
               resampled; every seat continues with the current policy; or exact Q (exact_q=True)
  weights      log w = -reach + log n_street_nodes, then clip_reach_weights(C=10) per street
  advantage    q - uniform mean over legal actions; per-street EMA sigma (decay 0.99, floor 1); anchors: magnet
               eta_reg * (log pi - log pi_reg) clamped +-1, uniform eta_unif * (log pi - log u), both scaled by
               sigma_ref/sigma, sum clamped +-1.5; subtracted; clamped +-3; re-centred over legal actions
  loss         uniform PPO surrogate vs the behaviour policy (clip 0.3) + entropy bonus alpha_ent x min(sigma_ref/
               sigma, 2) + forward KL(ref || pi) x beta_ref_fwd, weighted means over the step
  optimizer    schedule-free AdamW (betas 0.9/0.999, warmup 100) | Adam | SGD on the logits
  refine       Reg-KL plateau rule (refine when the last-30 mean grew <= 2% over the previous 30, not before
               reg_min_period, forced at reg_max_period) or a fixed period; magnet <- current policy (or an average)

Not modelled: function approximation (a TABLE, so no generalisation between information sets), asynchronous
generation (behaviour = the policy that generated the batch, `lag` steps old). Action values are multiplied by
payoff_scale (5) before the advantage stage so that sigma lands well above its floor.

    python -m lrc.harness --steps 3000 --out results/harness/base
"""
import argparse
import json
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lrc import weights as W  # noqa: E402
from lrc.game import ToyGame, Pivots  # noqa: E402

LOG_FLOOR = -15.0


def defaults():
    return dict(
        game="leduc", players=2, steps=3000, seed=1,
        pivots=512, quota=None, n_sims=[32, 16], crn=False, exact_q=False,
        eps_start=0.25, eps_floor=0.10, eps_anneal=3000, lag=0,
        isw="clip", isw_clip=10.0, node_correct=True, reach_reweight=True,
        adv_var_decay=0.99, sigma_floor=1.0, anchor_units="bb", sigma_ref="auto", sigma_ref_step=50,
        eta_reg=0.20, anchor_clip_z=1.0, eta_unif=0.05, anchor_sum_clip=1.5, adv_clip=3.0, center=True,
        ppo_clip=0.30, alpha_ent=0.04, ent_scale_max=2.0, beta_ref_fwd=0.08, ref="uniform",
        lr=0.03, warmup=100, optimizer="sfadamw",
        reg_outer=True, reg_min_period=1000, reg_max_period=8000, reg_plateau_tol=0.02,
        refine_rule="plateau", refine_period=1000,
        report_every=50, thm9_every=1, adv_sign=1.0, payoff_scale=5.0, q_noise=False,
        # baseline parity knobs. ref_target 'fixed' = the KL target is `ref` (baseline); 'reg' = the
        # magnet pi_reg (a reset variant retargets the pivot anchor to pi_reg). grad_norm k > 0 = every step
        # is k * g / ||g|| (the baseline's clip_grad_norm_(1.0) binds on every step).
        ref_target="fixed", grad_norm=0.0,
        # the exact reference (lrc/reference.py) found momentum (beta1 / schedule-free) and the
        # fixed uniform/entropy pulls (QRE floor 0.109) block convergence. beta1: Adam/sf-AdamW first moment;
        # optimizer 'adam' = plain torch Adam; pull_anneal N > 0 = eta_unif and alpha_ent decay linearly to 0 over N
        # steps (0 = fixed, baseline); beta_ref_fwd decays with them when ref_target == 'fixed' (it is a pull to
        # the fixed uniform ref).
        beta1=0.9, beta2=0.999, adam_eps=1e-8, pull_anneal=0,
        # logit_floor Z > 0: after every step each row's logits are clamped to >= row max - Z (support floor; the
        # exact reference lost support without it; DeepNash thresholds NeuRD logits for the same reason)
        logit_floor=0.0,
        # the sampled inner loop is a stochastic approximation (noise ball ~ lr even with exact Q), so a
        # refresh copies a NOISY iterate into the magnet. magnet_avg 'mean' = the magnet is the mean policy over the
        # second half of the inner phase (fixed refine rule); inner_lr_tau T > 0 = lr * T / (T + steps since refine).
        magnet_avg="last", inner_lr_tau=0.0,
    )


class Harness:
    def __init__(self, cfg):
        self.c = cfg
        self.g = ToyGame(cfg["game"], cfg["players"])
        g = self.g
        self.E = g.entries
        self.R = len(g.row_base)
        self.A = g.max_actions
        # row -> entry index matrix [R, A] with a validity mask
        idx = np.zeros((self.R, self.A), dtype=np.int64)
        valid = np.zeros((self.R, self.A), dtype=bool)
        for r in range(self.R):
            w = int(g.row_width[r])
            idx[r, :w] = g.row_base[r] + np.arange(w)
            valid[r, :w] = True
        self.idx, self.valid = idx, valid
        self.n_streets = len(g.buckets)
        torch.manual_seed(cfg["seed"])
        self.logits = torch.zeros(self.E, dtype=torch.float64, requires_grad=True)
        if cfg["optimizer"] == "sfadamw":
            import schedulefree
            self.opt = schedulefree.AdamWScheduleFree([self.logits], lr=cfg["lr"], warmup_steps=cfg["warmup"],
                                                      betas=(cfg["beta1"], cfg["beta2"]), weight_decay=0.0,
                                                      eps=cfg["adam_eps"])
            self.opt.train()
        elif cfg["optimizer"] == "adam":
            self.opt = torch.optim.Adam([self.logits], lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
                                        eps=cfg["adam_eps"])
        elif cfg["optimizer"] == "sgd":
            self.opt = torch.optim.SGD([self.logits], lr=cfg["lr"])
        else:
            raise ValueError(cfg["optimizer"])
        self.reg_logits = self.logits.detach().clone()
        self.avg_logits = self.logits.detach().clone()
        self.avg_wsum = 0.0
        self.S = np.zeros(self.E)                      # Thm 9 accumulator
        self.ref_probs = g.uniform() if cfg["ref"] == "uniform" else np.load(cfg["ref"])
        self.running_adv_var = [1.0] * self.n_streets
        self.sigma_ref = None if cfg["sigma_ref"] == "auto" else list(cfg["sigma_ref"])
        q = cfg["quota"] or [1.0 / self.n_streets] * self.n_streets
        self.quota_base = np.asarray(q, dtype=np.float64) / np.sum(q)
        self.reg_hist, self.last_refine, self.refines = [], 0, 0
        self.behav_hist = []
        self.isw_trunc = W.TruncatedISNormaliser(n_streets=4, beta=0.99)
        self.t = 0
        self.mag_sum, self.mag_n = None, 0

    # ---- policy tables --------------------------------------------------------
    def probs_of(self, logits_flat):
        z = logits_flat.detach().numpy() if torch.is_tensor(logits_flat) else logits_flat
        zr = np.where(self.valid, z[self.idx], -np.inf)
        zr = zr - zr.max(axis=1, keepdims=True)
        p = np.exp(zr)
        p /= p.sum(axis=1, keepdims=True)
        out = np.zeros(self.E)
        out[self.idx[self.valid]] = p[self.valid]
        return out

    def acting_probs(self, probs):
        """select_action_batch's floor: clamp(min=1e-5) on valid, renormalise."""
        pr = np.where(self.valid, np.maximum(probs[self.idx], 1e-5), 0.0)
        pr /= pr.sum(axis=1, keepdims=True)
        out = np.zeros(self.E)
        out[self.idx[self.valid]] = pr[self.valid]
        return out

    def eps_now(self):
        c = self.c
        if c["eps_anneal"] <= 0:
            return c["eps_floor"]
        f = min(1.0, self.t / float(c["eps_anneal"]))
        return c["eps_start"] + (c["eps_floor"] - c["eps_start"]) * f

    # ---- collector ---------------------------------------------------------------
    def quota_draw(self, rng):
        B = self.c["pivots"]
        exact = self.quota_base * B
        q = np.floor(exact).astype(int)
        r = B - int(q.sum())
        if r > 0:
            frac = exact - q
            q += rng.multinomial(r, frac / frac.sum())
        return q

    def collect(self, act_probs, rng, seed):
        c = self.c
        q = self.quota_draw(rng)
        picked = []
        batch = 0
        while q.sum() > 0:
            piv = self.g.generate(act_probs, n_hands=c["pivots"] * 2, eps=self.eps_now(),
                                  seed=seed * 1000003 + batch)
            batch += 1
            by_hand = {}
            for k in range(len(piv)):
                by_hand.setdefault(int(piv.hand[k]), {})[int(piv.street[k])] = k
            hands = sorted(by_hand)
            h_i = 0
            while q.sum() > 0 and h_i < len(hands):
                s = int(rng.choice(len(q), p=q / q.sum()))       # drawn before inspecting a hand
                while h_i < len(hands) and s not in by_hand[hands[h_i]]:
                    h_i += 1                                     # discard hands without street s
                if h_i >= len(hands):
                    break
                picked.append((batch - 1, piv, by_hand[hands[h_i]][s]))
                q[s] -= 1
                h_i += 1
            if batch > 50:
                raise RuntimeError("collector starved (quota asks for a street hands rarely reach)")
        rows = np.stack([p.i[k] for _, p, k in picked])
        lps = np.asarray([p.reach_lp[k] for _, p, k in picked])
        out = Pivots.__new__(Pivots)
        out.i, out.reach_lp = np.ascontiguousarray(rows), np.ascontiguousarray(lps)
        return out

    # ---- one sampled step -----------------------------------------------------------------
    def step(self, dry=False):
        """One sampled step. dry=True computes everything (and returns the gradient,
        the advantages and the pivots) but does not move the policy, the
        averages or the refine state -- for the audits."""
        c = self.c
        self.t += 1
        t = self.t
        rng = np.random.default_rng([c["seed"], t])            # counter-based: replayable
        probs = self.probs_of(self.logits)
        self.behav_hist.append(probs)
        behav = self.behav_hist[max(0, len(self.behav_hist) - 1 - c["lag"])]
        if len(self.behav_hist) > c["lag"] + 1:
            self.behav_hist.pop(0)
        act = self.acting_probs(behav)
        piv = self.collect(act, rng, seed=c["seed"] * 7919 + t)
        B = len(piv)
        rows = self.g.row_of(piv.node, piv.bucket)
        row_idx = np.searchsorted(self.g.row_base, rows)
        valid = self.valid[row_idx]
        ent_idx = self.idx[row_idx]
        streets = piv.street.astype(np.int64)

        # action values
        rollout_probs = self.acting_probs(probs)
        if c["exact_q"]:
            q = self.g.exact_q(rollout_probs, piv)[:, :self.A]
            noise_var = np.zeros_like(q)
        else:
            q16, v16 = self.g.rollouts(rollout_probs, piv, c["n_sims"], seed=c["seed"] * 104729 + t, crn=c["crn"])
            q, noise_var = q16[:, :self.A], v16[:, :self.A]
        if c["q_noise"]:                  # audit control: advantages carry no information
            q = rng.normal(0.0, 1.0, q.shape)
        q = np.where(valid, q * c["payoff_scale"], 0.0)

        # weights
        lw = np.zeros(B)
        if c["reach_reweight"]:
            lw -= piv.reach_lp
        if c["node_correct"]:
            lw += np.log(np.maximum(piv.n_street_nodes, 1))
        if c["isw"] == "clip":
            w, wst = W.clip_reach_weights(lw, streets, c["isw_clip"])
        elif c["isw"] == "truncated":     # alternative: running normaliser + Ionides truncation
            w, wst = self.isw_trunc.weights(lw, streets)
            if not dry:
                self.isw_trunc.update(lw, streets)
        elif c["isw"] == "none":
            w, wst = np.ones(B), {"ess": [float((streets == s).sum()) for s in range(4)], "truncated_frac": [0.0] * 4}
        elif c["isw"] == "raw":           # untruncated, self-normalised per street (the unbiased-in-the-limit form)
            w = np.ones(B)
            for s in range(self.n_streets):
                m = streets == s
                if m.any():
                    u = np.exp(lw[m] - lw[m].max())
                    w[m] = u / u.mean()
            wst = {"ess": [float(w[streets == s].sum() ** 2 / max((w[streets == s] ** 2).sum(), 1e-12))
                           for s in range(4)], "truncated_frac": [0.0] * 4}
        else:
            raise ValueError(c["isw"])
        w_t = torch.as_tensor(w, dtype=torch.float64)

        vf = valid.astype(np.float64)
        k = vf.sum(1, keepdims=True)
        adv_raw = c["adv_sign"] * (q - (q * vf).sum(1, keepdims=True) / k) * vf
        # per-street sigma, once per step
        for s in range(self.n_streets):
            m = streets == s
            if m.any():
                bv = float((adv_raw[m] ** 2).sum() / vf[m].sum())
                d = c["adv_var_decay"]
                self.running_adv_var[s] = d * self.running_adv_var[s] + (1 - d) * bv
        sig = np.asarray([max(math.sqrt(v), c["sigma_floor"]) for v in self.running_adv_var])
        if self.sigma_ref is None and c["sigma_ref"] == "auto" and t >= c["sigma_ref_step"]:
            self.sigma_ref = list(sig)
        ref_sig = np.asarray(self.sigma_ref if self.sigma_ref is not None else sig)
        sig_row = sig[streets][:, None]
        adv_norm = adv_raw / sig_row

        # log-policies at the pivots
        theta_all = self.logits[torch.as_tensor(ent_idx)]
        mask = torch.as_tensor(valid)
        vmask = torch.as_tensor(vf)
        neg = torch.full_like(theta_all, float("-inf"))
        theta_log = torch.nan_to_num(torch.log_softmax(torch.where(mask, theta_all, neg), dim=1),
                                     nan=LOG_FLOOR, neginf=LOG_FLOOR)
        tl = theta_log.detach().numpy()
        rl = np.log(np.maximum(self.probs_of(self.reg_logits)[ent_idx], 1e-300))
        rl = np.where(valid, np.maximum(rl, LOG_FLOOR), LOG_FLOOR)
        log_unif = np.where(valid, -np.log(k), 0.0)
        pull = 1.0 if c["pull_anneal"] <= 0 else max(0.0, 1.0 - t / float(c["pull_anneal"]))
        unif_anchor = c["eta_unif"] * pull * (tl - log_unif)
        if c["reg_outer"]:
            primary = np.clip(c["eta_reg"] * (tl - rl), -c["anchor_clip_z"], c["anchor_clip_z"])
        else:
            primary = np.zeros_like(tl)
        if c["anchor_units"] == "bb":
            scale = (ref_sig[streets] / sig[streets])[:, None]
            primary, unif_anchor = primary * scale, unif_anchor * scale
        anchor_total = np.clip(primary + unif_anchor, -c["anchor_sum_clip"], c["anchor_sum_clip"])
        pre_clip = adv_norm - anchor_total
        clip_bind = float(((np.abs(pre_clip) > c["adv_clip"]) * vf).sum() / vf.sum())
        adv_reg = np.clip(pre_clip, -c["adv_clip"], c["adv_clip"]) * vf
        if c["center"]:
            adv_reg = W.center_over_valid(adv_reg, vf)
        adv_t = torch.as_tensor(adv_reg)

        behav_log = torch.as_tensor(np.log(np.maximum(np.where(valid, behav[ent_idx], 1.0), 1e-6)))
        pol_row, clip_frac = W.ppo_uniform_surrogate(theta_log, behav_log, adv_t, vmask, c["ppo_clip"])
        dw_sum = w_t.sum().clamp(min=1e-6)
        policy_loss = (pol_row * w_t).sum() / dw_sum
        pi = torch.exp(theta_log)
        ent = -(pi * theta_log * vmask).sum(1)
        ent = ent * torch.as_tensor(np.minimum(ref_sig[streets] / sig[streets], c["ent_scale_max"]))
        entropy_loss = -c["alpha_ent"] * pull * (ent * w_t).sum() / dw_sum
        _refp = self.probs_of(self.reg_logits) if c["ref_target"] == "reg" else self.ref_probs
        ref_pi = torch.as_tensor(np.where(valid, _refp[ent_idx], 0.0))
        ref_pi = ref_pi / ref_pi.sum(1, keepdim=True).clamp(min=1e-8)
        ref_log = torch.log(ref_pi.clamp(min=1e-300))
        kl_ref = (ref_pi * torch.nan_to_num(ref_log - theta_log, nan=0.0, posinf=0.0, neginf=0.0) * vmask).sum(1).clamp(min=0.0)
        ref_loss = c["beta_ref_fwd"] * (pull if c["ref_target"] == "fixed" else 1.0) * (kl_ref * w_t).sum() / dw_sum
        loss = policy_loss + entropy_loss + ref_loss

        with torch.no_grad():
            p_d = np.exp(tl) * vf
            reg_kl = float((w * (p_d * (tl - rl)).sum(1)).sum() / w.sum())

        self.opt.zero_grad()
        loss.backward()
        if dry:
            self.t -= 1
            return {"grad": self.logits.grad.detach().numpy().copy(), "adv": adv_reg, "valid": valid,
                    "ent_idx": ent_idx, "w": w, "piv": piv, "q": q, "loss": float(loss.detach())}
        if c["inner_lr_tau"] > 0:
            _f = c["inner_lr_tau"] / (c["inner_lr_tau"] + (t - self.last_refine))
            for _g in self.opt.param_groups:
                _g["lr"] = c["lr"] * _f
        if c["grad_norm"] > 0.0:                      # the baseline's always-binding norm clip
            with torch.no_grad():
                _gn = self.logits.grad.norm()
                if _gn > 0:
                    self.logits.grad.mul_(c["grad_norm"] / _gn)
        self.opt.step()
        if c["logit_floor"] > 0:
            with torch.no_grad():
                z = self.logits.detach().numpy()
                zr = np.where(self.valid, z[self.idx], -np.inf)
                lo = zr.max(axis=1, keepdims=True) - c["logit_floor"]
                zc = np.where(self.valid, np.maximum(zr, lo), 0.0)
                z2 = z.copy()
                z2[self.idx[self.valid]] = zc[self.valid]
                self.logits.copy_(torch.as_tensor(z2))

        # avg_model: parameter average, weight t
        with torch.no_grad():
            self.avg_wsum += float(t)
            alpha = float(t) / self.avg_wsum
            self.avg_logits.mul_(1 - alpha).add_(self.logits.detach(), alpha=alpha)
        # Thm 9 object: reach-weighted average of the policies actually played
        if c["thm9_every"] and t % c["thm9_every"] == 0:
            self.g.avg_accumulate(self.acting_probs(probs), float(t) * c["thm9_every"], self.S)

        # NashPG refine
        refined = False
        if c["reg_outer"]:
            self.reg_hist.append(reg_kl)
            since = t - self.last_refine
            plateau = False
            if since >= c["reg_min_period"] and len(self.reg_hist) >= 60:
                old = sum(self.reg_hist[-60:-30]) / 30.0
                new = sum(self.reg_hist[-30:]) / 30.0
                plateau = (new - old) <= c["reg_plateau_tol"] * max(old, 1e-6)
            if c["refine_rule"] == "fixed":        # APMD-style fixed interval (the A/B arm)
                plateau = since >= c["refine_period"]
            if c["magnet_avg"] == "mean" and since > c["refine_period"] // 2:
                _pp = self.probs_of(self.logits)
                self.mag_sum = _pp if self.mag_sum is None else self.mag_sum + _pp
                self.mag_n += 1
            if plateau or since >= c["reg_max_period"]:
                if c["magnet_avg"] == "mean" and self.mag_n > 0:
                    self.reg_logits = torch.as_tensor(np.log(np.maximum(self.mag_sum / self.mag_n, 1e-300)))
                else:
                    self.reg_logits = self.logits.detach().clone()
                self.mag_sum, self.mag_n = None, 0
                self.last_refine, self.reg_hist, refined = t, [], True
                self.refines += 1

        return {"t": t, "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "reg_kl": reg_kl,
                "clip_bind": clip_bind, "ppo_clip_frac": float(clip_frac), "sigma": list(map(float, sig)),
                "ess": [float(x) for x in wst["ess"][:self.n_streets]],
                "n": [int((streets == s).sum()) for s in range(self.n_streets)],
                "trunc": [float(x) for x in wst["truncated_frac"][:self.n_streets]],
                "noise_var": [float(noise_var[streets == s].sum() / max(vf[streets == s].sum(), 1))
                              for s in range(self.n_streets)],
                "eps": self.eps_now(), "refined": refined}

    def score(self):
        cur = self.g.exploitability(self.probs_of(self.logits))
        avg = self.g.exploitability(self.probs_of(self.avg_logits))
        thm9 = self.g.exploitability(self.g.normalise(self.S)) if self.S.sum() > 0 else None
        return {"current": cur, "avg_model": avg, "reach_avg": thm9}


def run(cfg, out_dir=None, log=print):
    h = Harness(cfg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as fh:
            json.dump(cfg, fh, indent=1)
        fh = open(os.path.join(out_dir, "log.jsonl"), "w")
    t0 = time.time()
    curve = []
    for _ in range(cfg["steps"]):
        st = h.step()
        if h.t % cfg["report_every"] == 0 or h.t == cfg["steps"]:
            sc = h.score()
            rec = dict(st, current=sc["current"]["nashconv"], avg_model=sc["avg_model"]["nashconv"],
                       reach_avg=sc["reach_avg"]["nashconv"] if sc["reach_avg"] else None,
                       max_gain=sc["current"]["max_gain"], refines=h.refines, sec=time.time() - t0)
            curve.append(rec)
            if out_dir:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            log("t=%5d nashconv cur=%.4f avg_model=%.4f reach_avg=%s | regKL=%.4f refines=%d clip=%.3f "
                "ess=%s sigma=%s %.0fs" % (
                    h.t, rec["current"], rec["avg_model"],
                    "%.4f" % rec["reach_avg"] if rec["reach_avg"] is not None else "-",
                    st["reg_kl"], h.refines, st["clip_bind"],
                    "/".join("%.0f" % e for e in st["ess"]), "/".join("%.2f" % s for s in st["sigma"]),
                    rec["sec"]))
    if out_dir:
        fh.close()
    return h, curve


def main():
    ap = argparse.ArgumentParser()
    d = defaults()
    for k, v in d.items():
        if isinstance(v, bool):
            ap.add_argument("--" + k.replace("_", "-"), type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        elif isinstance(v, list) or v is None:
            ap.add_argument("--" + k.replace("_", "-"), type=lambda s: [float(x) for x in s.split(",")], default=v)
        else:
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--out", default=None)
    a = vars(ap.parse_args())
    out = a.pop("out")
    if a["n_sims"]:
        a["n_sims"] = [int(x) for x in a["n_sims"]]
    run(a, out)


if __name__ == "__main__":
    main()
