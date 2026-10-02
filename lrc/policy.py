"""policy.py -- policy parameterisations and optimizers shared by the exact learners and the sampled rule.

table  one free logit per (information set, action): adaptive optimizers normalise PER INFORMATION-SET ACTION.
mlp    one shared network maps information-set FEATURES to the logits of every information set: adaptive optimizers
       normalise PER WEIGHT, shared across information sets -- the regime of policy-gradient training of large models.

Features of an information set (no information-set id; the network must generalise):
  round one-hot (2), actor one-hot (2), own rank one-hot (3), board rank one-hot (3, zero in round 1), pair flag (1),
  round-1 betting sequence (4 slots x {call/check, raise}), round-2 betting sequence (4 x 2), facing-a-bet flag (1),
  raise-allowed flag (1)  ->  29 inputs. (history, bucket) is fully encoded, so any tabular policy is representable.

Optimizers: sgd | adam | sfadamw (schedule-free AdamW) | sfsgd (schedule-free SGD). For the schedule-free ones the
first entry of `betas` is the interpolation parameter beta of y = (1 - beta) z + beta x (Defazio et al. 2024).
`optimistic=True` feeds 2 g_t - g_{t-1} to the optimizer (gradient-level optimism, the optimistic-gradient step of
Daskalakis et al. 2018 when the optimizer is SGD; with Adam it is the gradient-level form, not Optimistic Adam's
update-level form).
"""
import numpy as np
import torch

N_FEAT = 29


def row_features(g):
    """[n_rows, 29] float64 features in the row order of the policy table."""
    feats = []
    rows = []
    for nid in sorted(g.node_nA):
        nd = g.nodes[nid]
        for b in range(g.buckets[nd["street"]]):
            rows.append((g.node_row0[nid] + b * nd["nA"], nid, b))
    rows.sort()
    for _, nid, b in rows:
        nd = g.nodes[nid]
        f = np.zeros(N_FEAT)
        st, actor = nd["street"], nd["actor"]
        f[st] = 1.0
        f[2 + actor] = 1.0
        own = b if st == 0 else b // g.R
        f[4 + own] = 1.0
        if st == 1:
            board = b % g.R
            f[7 + board] = 1.0
            f[10] = float(own == board)
        h = nd["hist"].split("/")
        for rnd, seq in enumerate(h[:2]):
            for k, ch in enumerate(seq[:4]):
                f[11 + rnd * 8 + k * 2 + (0 if ch == "c" else 1)] = 1.0
        f[27] = float(nd["facing"])
        f[28] = float(nd["can_raise"])
        feats.append(f)
    F = np.stack(feats)
    assert len(F) == len(g.row_base)
    return F


class MLPPolicy(torch.nn.Module):
    """features -> 3 logits per row; flat entry logits gathered by (row, action index). Zero last layer: uniform."""

    def __init__(self, g, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.F = torch.as_tensor(row_features(g))
        self.net = torch.nn.Sequential(torch.nn.Linear(N_FEAT, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, 3)).double()
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.erow = torch.as_tensor(g.entry_row)
        self.eact = torch.as_tensor(np.arange(g.entries) - g.row_base[g.entry_row])

    def forward(self):
        out = self.net(self.F)
        return out[self.erow, self.eact]


def make_optimizer(params, opt, lr, betas=(0.9, 0.999), eps=1e-8, warmup=100):
    params = list(params)
    if opt == "sgd":
        o = torch.optim.SGD(params, lr=lr)
    elif opt == "adam":
        o = torch.optim.Adam(params, lr=lr, betas=tuple(betas), eps=eps)
    elif opt == "sfadamw":
        import schedulefree
        o = schedulefree.AdamWScheduleFree(params, lr=lr, warmup_steps=warmup, betas=tuple(betas), weight_decay=0.0,
                                           eps=eps)
        o.train()
    elif opt == "sfsgd":
        import schedulefree
        o = schedulefree.SGDScheduleFree(params, lr=lr, momentum=betas[0], warmup_steps=warmup, weight_decay=0.0)
        o.train()
    else:
        raise ValueError(opt)
    return o


class Optimism:
    """Replace each parameter's gradient g_t by 2 g_t - g_{t-1} (in place) before the optimizer step."""

    def __init__(self, params):
        self.params = list(params)
        self.prev = None

    def apply(self):
        cur = [p.grad.detach().clone() for p in self.params]
        if self.prev is not None:
            for p, c, q in zip(self.params, cur, self.prev):
                p.grad.copy_(2.0 * c - q)
        self.prev = cur
