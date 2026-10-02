"""Estimator helpers of the sampled rule: importance weights, advantage centring and the uniform PPO surrogate.

clip_reach_weights(log_w, streets, C)    per street: centre log-weights on the batch mean, clip at +-log C,
                                         exponentiate, renormalise to mean 1 (any two weights differ by <= C^2)
TruncatedISNormaliser                    per street: exp(min(lw - m_s, 0.5 log n_s)) with m_s a running log-mean of
                                         exp(lw) from PREVIOUS steps (Ionides 2008 truncation at mean * sqrt(n))
center_over_valid(adv, valid)            adv - mean over legal actions; zero on illegal ones. After it sum_a adv_a = 0,
                                         the condition under which the uniform surrogate's logit gradient is NeuRD
ppo_uniform_surrogate(...)               per row: -(1/|A|) sum_a min(r_a A_a, clip(r_a, 1-e, 1+e) A_a),
                                         r_a = pi_a / mu_a (uniform weight over legal actions, not pi-weighted)
"""
import math

import numpy as np


def _logmeanexp(x):
    m = float(np.max(x))
    return m + math.log(float(np.mean(np.exp(x - m))))


def clip_reach_weights(log_w, streets, clip=10.0):
    log_w = np.asarray(log_w, dtype=np.float64)
    streets = np.asarray(streets)
    w = np.ones_like(log_w)
    n_st = 4 if streets.size == 0 else max(4, int(streets.max()) + 1)
    stats = {"truncated_frac": [0.0] * n_st, "ess": [0.0] * n_st, "n": [0] * n_st}
    lc = math.log(float(clip))
    for s in range(n_st):
        sel = streets == s
        n = int(sel.sum())
        stats["n"][s] = n
        if n == 0:
            continue
        c = log_w[sel] - log_w[sel].mean()
        u = np.exp(np.clip(c, -lc, lc))
        u = u / u.mean()
        top = c.max()
        raw_sum = float(np.exp(c - top).sum())
        kept_sum = float(np.exp(np.minimum(c, lc) - top).sum())
        stats["truncated_frac"][s] = max(1.0 - kept_sum / raw_sum, 0.0)
        stats["ess"][s] = float(u.sum() ** 2 / (u * u).sum())
        w[sel] = u
    return w, stats


class TruncatedISNormaliser:
    def __init__(self, n_streets=4, beta=0.99):
        self.n_streets = n_streets
        self.beta = float(beta)
        self.log_mean = [None] * n_streets

    def weights(self, log_w, streets):
        log_w = np.asarray(log_w, dtype=np.float64)
        streets = np.asarray(streets)
        w = np.ones_like(log_w)
        stats = {"truncated_frac": [0.0] * self.n_streets, "ess": [0.0] * self.n_streets, "n": [0] * self.n_streets}
        for s in range(self.n_streets):
            sel = streets == s
            n = int(sel.sum())
            stats["n"][s] = n
            if n == 0:
                continue
            lw = log_w[sel]
            ref = self.log_mean[s] if self.log_mean[s] is not None else _logmeanexp(lw)
            r = lw - ref
            rt = np.minimum(r, 0.5 * math.log(max(n, 1)))
            ws = np.exp(rt)
            m = r.max()
            raw_sum = float(np.exp(r - m).sum())
            over = np.clip(np.exp(r - m) - np.exp(rt - m), 0.0, None).sum()
            stats["truncated_frac"][s] = float(over / raw_sum) if raw_sum > 0 else 0.0
            sw, sw2 = float(ws.sum()), float((ws * ws).sum())
            stats["ess"][s] = sw * sw / sw2 if sw2 > 0 else 0.0
            w[sel] = ws
        return w, stats

    def update(self, log_w, streets):
        log_w = np.asarray(log_w, dtype=np.float64)
        streets = np.asarray(streets)
        b = self.beta
        for s in range(self.n_streets):
            sel = streets == s
            if not sel.any():
                continue
            cur = _logmeanexp(log_w[sel])
            if self.log_mean[s] is None:
                self.log_mean[s] = cur
            else:                       # EMA of the mean weight, kept in log space
                self.log_mean[s] = float(np.logaddexp(math.log(b) + self.log_mean[s], math.log(1 - b) + cur))


def center_over_valid(adv, valid_f):
    if isinstance(adv, np.ndarray):
        k = np.maximum(valid_f.sum(-1, keepdims=True), 1.0)
        return (adv - (adv * valid_f).sum(-1, keepdims=True) / k) * valid_f
    k = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (adv - (adv * valid_f).sum(dim=-1, keepdim=True) / k) * valid_f


def ppo_uniform_surrogate(theta_log, behavior_log, adv, valid_f, ppo_clip):
    """torch. adv is detached by the caller. Returns (loss_per_row, clip_frac)."""
    import torch
    ratio = torch.exp((theta_log - behavior_log).clamp(min=-10.0, max=10.0))
    ratio_c = ratio.clamp(min=1.0 - ppo_clip, max=1.0 + ppo_clip)
    surr = torch.min(ratio * adv, ratio_c * adv)
    with torch.no_grad():
        sel = (ratio * adv) > (ratio_c * adv)
        clip_frac = (sel.float() * valid_f).sum() / valid_f.sum().clamp(min=1.0)
    k = valid_f.sum(dim=1).clamp(min=1.0)
    return -(surr * valid_f).sum(dim=1) / k, clip_frac
