"""Kuhn and Leduc poker (2 players) in pure numpy, with the policy-table interface the harness uses.

This replaces the native validation-game library the experiments were first run on. Tree construction, node ids, the
entry layout, the action order and every quantity below were cross-checked against that library (see
docs/VALIDATION.md): identical node ids and row layout, exploitability equal to 1e-12 on random tables,
generation/rollout statistics equal within Monte-Carlo error.

Rules
  Kuhn   3 cards (J<Q<K), ante 1, one betting round, bet size 1, at most 1 raise (a bet counts as a raise).
  Leduc  6 cards (2 suits x 3 ranks), ante 1, two betting rounds (raise size 2 then 4, at most 2 raises per round),
         one public board card dealt between the rounds; a private card pairing the board wins, otherwise the higher
         rank wins, equal ranks split. Player 0 acts first in every round.

Policy table
  A behavioural policy is a flat float64 vector over ENTRIES. Decision node n with nA legal actions owns
  n_buckets(street) consecutive ROWS of nA entries: entry = row0[n] + bucket * nA[n] + action. Action order:
  fold (only when facing a bet), call/check, raise (only below the cap). Buckets: own card rank in round 1,
  own rank * R + board rank in round 2 (lossless: suits are irrelevant).

Interface (what bench code uses)
  entries, row_base, row_width, row_street, row_actor, entry_row, max_actions, buckets,
  node_nA / node_kinds / node_actor / node_street / node_row0 / node_children (dicts over decision nodes),
  row_of(node, bucket), uniform(), normalise(S),
  generate(probs, n_hands, eps, seed)        on-policy hands -> Pivots (one reservoir-of-one pivot per street)
  rollouts(probs, piv, n_sims, seed, crn)    Monte-Carlo Q(h, a) per legal action: mean and sample variance
  exact_q(probs, piv)                        exact Q(h, a) (expectation over the undealt board and all play)
  exact(probs) / exploitability(probs)       exact EV and best-response values (every deal enumerated)
  avg_accumulate(probs, weight, S)           S += weight * P(deal) * own reach * policy (Thm-9 accumulator)
"""
import itertools

import numpy as np

A_FOLD, A_CALL, A_RAISE = 0, 1, 2
N_CARD_COLS = 3             # pivot record: hand, hero, street, node, bucket, n_street_nodes, card0, card1, board
REC_I = 6 + N_CARD_COLS


class Pivots:
    """Generated pivots. Columns of `i`: hand, hero, street, node, bucket, n_street_nodes, cards (p0, p1, board or -1);
    `reach_lp` is the hero's log reach (eps-mixed action probabilities) before the pivot."""

    def __init__(self, i, reach_lp):
        self.i = np.ascontiguousarray(i, dtype=np.int64)
        self.reach_lp = np.ascontiguousarray(reach_lp, dtype=np.float64)

    def __len__(self):
        return len(self.i)

    hand = property(lambda self: self.i[:, 0])
    hero = property(lambda self: self.i[:, 1])
    street = property(lambda self: self.i[:, 2])
    node = property(lambda self: self.i[:, 3])
    bucket = property(lambda self: self.i[:, 4])
    n_street_nodes = property(lambda self: self.i[:, 5])

    def take(self, idx):
        return Pivots(self.i[idx], self.reach_lp[idx])


class ToyGame:
    def __init__(self, game="leduc", players=2):
        if players != 2:
            raise NotImplementedError("this repository implements 2-player Kuhn and Leduc")
        self.game, self.players = game, players
        if game == "kuhn":
            self.R, self.n_streets_game, self.raise_sizes, self.max_raises = 3, 1, (1,), 1
            deals = np.asarray(list(itertools.permutations(range(3), 2)), dtype=np.int64)
            deals = np.concatenate([deals, np.full((len(deals), 1), -1)], axis=1)
            rank = deals[:, :2]
            self.bk = [[rank[:, p]] for p in range(2)]
            strength = rank
            self.buckets = [3]
            self.board_of = None
        elif game == "leduc":
            self.R, self.n_streets_game, self.raise_sizes, self.max_raises = 3, 2, (2, 4), 2
            deals = np.asarray(list(itertools.permutations(range(6), 3)), dtype=np.int64)
            rank = deals // 2
            self.bk = [[rank[:, p], rank[:, p] * 3 + rank[:, 2]] for p in range(2)]
            strength = np.where(rank[:, :2] == rank[:, 2:3], 1000 + rank[:, :2], rank[:, :2])
            self.buckets = [3, 9]
        else:
            raise ValueError(game)
        self.deals = deals
        self.D = len(deals)
        self.chance = 1.0 / self.D
        self.showdown_sign = np.sign(strength[:, 0] - strength[:, 1]).astype(np.float64)
        lut = {tuple(d): k for k, d in enumerate(deals.tolist())}
        self._deal_index = lut
        if game == "leduc":   # deals sharing the private cards (the round-1 state), for board averaging/resampling
            key = deals[:, 0] * 6 + deals[:, 1]
            _, self.private_group = np.unique(key, return_inverse=True)
            self.n_groups = int(self.private_group.max()) + 1
            members = [np.nonzero(self.private_group == gi)[0] for gi in range(self.n_groups)]
            self.group_members = np.stack(members)                 # [groups, 4] deal ids
        self._build()

    # ---- tree ---------------------------------------------------------------------------------------------------
    def _build(self):
        nodes = []
        entry = [0]

        def terminal(street, commit, folder):
            nid = len(nodes)
            if folder is None:
                u0 = self.showdown_sign * commit[0]
            else:
                u0 = np.full(self.D, float(commit[1] if folder == 1 else -commit[0]))
            nodes.append(dict(kind="T", street=street, u0=u0))
            return nid

        def rec(street, commit, raises, actor, acts_in_round):
            facing = max(commit) - commit[actor]
            kinds = ([A_FOLD] if facing > 0 else []) + [A_CALL] + ([A_RAISE] if raises < self.max_raises else [])
            nid = len(nodes)
            nd = dict(kind="D", actor=actor, street=street, nA=len(kinds), kinds=kinds, children=[], row0=None,
                      actor_commit=commit[actor])
            nodes.append(nd)
            for a in kinds:
                c2 = list(commit)
                if a == A_FOLD:
                    ch = terminal(street, c2, actor)
                elif a == A_CALL:
                    c2[actor] += facing
                    if not (facing > 0 or acts_in_round >= 1):
                        ch = rec(street, c2, raises, 1 - actor, acts_in_round + 1)
                    elif street + 1 < self.n_streets_game:
                        ch = rec(street + 1, c2, 0, 0, 0)
                    else:
                        ch = terminal(street, c2, None)
                else:
                    c2[actor] += facing + self.raise_sizes[street]
                    ch = rec(street, c2, raises + 1, 1 - actor, acts_in_round + 1)
                nd["children"].append(ch)
            return nid

        rec(0, [1, 1], 0, 0, 0)
        for nd in nodes:                         # rows laid out in node-id order
            if nd["kind"] == "D":
                nd["row0"] = entry[0]
                entry[0] += self.buckets[nd["street"]] * nd["nA"]
        self.nodes = nodes
        self.entries = entry[0]
        dec = [k for k, nd in enumerate(nodes) if nd["kind"] == "D"]
        self.node_nA = {k: nodes[k]["nA"] for k in dec}
        self.node_kinds = {k: list(nodes[k]["kinds"]) for k in dec}
        self.node_actor = {k: nodes[k]["actor"] for k in dec}
        self.node_street = {k: nodes[k]["street"] for k in dec}
        self.node_row0 = {k: nodes[k]["row0"] for k in dec}
        self.node_children = {k: list(nodes[k]["children"]) for k in dec}
        self.max_actions = max(self.node_nA.values())
        base, width, street, actor = [], [], [], []
        for k in dec:
            for b in range(self.buckets[nodes[k]["street"]]):
                base.append(nodes[k]["row0"] + b * nodes[k]["nA"])
                width.append(nodes[k]["nA"])
                street.append(nodes[k]["street"])
                actor.append(nodes[k]["actor"])
        self.row_base = np.asarray(base, dtype=np.int64)
        self.row_width = np.asarray(width, dtype=np.int64)
        self.row_street = np.asarray(street, dtype=np.int64)
        self.row_actor = np.asarray(actor, dtype=np.int64)
        assert np.all(np.diff(self.row_base) > 0) and int(self.row_width.sum()) == self.entries
        self.entry_row = np.repeat(np.arange(len(self.row_base)), self.row_width)
        # padded arrays for vectorised play
        N = len(nodes)
        self.a_term = np.asarray([nd["kind"] == "T" for nd in nodes])
        self.a_actor = np.asarray([nd.get("actor", -1) for nd in nodes])
        self.a_street = np.asarray([nd["street"] for nd in nodes])
        self.a_nA = np.asarray([nd.get("nA", 0) for nd in nodes])
        self.a_row0 = np.asarray([nd["row0"] if nd["kind"] == "D" else 0 for nd in nodes])
        self.a_child = np.zeros((N, 3), dtype=np.int64)
        for k in dec:
            self.a_child[k, :nodes[k]["nA"]] = nodes[k]["children"]
        self.a_u0 = np.stack([nd["u0"] if nd["kind"] == "T" else np.zeros(self.D) for nd in nodes])
        self.bk_arr = np.zeros((2, self.n_streets_game, self.D), dtype=np.int64)
        for p in range(2):
            for s in range(self.n_streets_game):
                self.bk_arr[p, s] = self.bk[p][s]

    # ---- table helpers -------------------------------------------------------------------------------------------
    def row_of(self, node, bucket):
        node = np.asarray(node, dtype=np.int64)
        return self.a_row0[node] + np.asarray(bucket, dtype=np.int64) * self.a_nA[node]

    def uniform(self):
        return (1.0 / self.row_width[self.entry_row]).astype(np.float64)

    def normalise(self, S):
        """Row-normalise an accumulated table; rows with no mass become uniform."""
        tot = np.bincount(self.entry_row, weights=S, minlength=len(self.row_base))[self.entry_row]
        return np.where(tot > 0, S / np.where(tot > 0, tot, 1.0), self.uniform())

    def deal_id(self, cards):
        return self._deal_index[tuple(int(c) for c in cards)]

    # ---- exact values ----------------------------------------------------------------------------------------------
    def node_values(self, probs):
        """v[node] -> [2, D]: each player's expected utility from `node` for every deal (both players play probs)."""
        probs = np.asarray(probs, dtype=np.float64)
        out = {}

        def walk(nid):
            nd = self.nodes[nid]
            if nd["kind"] == "T":
                v = np.stack([nd["u0"], -nd["u0"]])
            else:
                bk = self.bk[nd["actor"]][nd["street"]]
                base = nd["row0"] + bk * nd["nA"]
                v = np.zeros((2, self.D))
                for a, ch in enumerate(nd["children"]):
                    v += probs[base + a][None, :] * walk(ch)
            out[nid] = v
            return v

        walk(0)
        return out

    def exact(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        ev = self.node_values(probs)[0].sum(1) * self.chance
        br = [self._br_value(probs, i) for i in range(2)]
        return list(map(float, ev)), br

    def _br_value(self, probs, i):
        def walk(nid, opp):
            nd = self.nodes[nid]
            if nd["kind"] == "T":
                return nd["u0"] if i == 0 else -nd["u0"]
            bk = self.bk[nd["actor"]][nd["street"]]
            base = nd["row0"] + bk * nd["nA"]
            if nd["actor"] == i:
                vs = np.stack([walk(ch, opp) for ch in nd["children"]])          # [nA, D]
                nb = self.buckets[nd["street"]]
                tot = np.stack([np.bincount(bk, weights=opp * v, minlength=nb) for v in vs])
                best = tot.argmax(0)
                return vs[best[bk], np.arange(self.D)]
            v = np.zeros(self.D)
            for a, ch in enumerate(nd["children"]):
                pa = probs[base + a]
                v += pa * walk(ch, opp * pa)
            return v

        return float((walk(0, np.full(self.D, self.chance)) * self.chance).sum())

    def exploitability(self, probs):
        ev, br = self.exact(probs)
        gains = [b - e for e, b in zip(ev, br)]
        return {"ev": ev, "gains": gains, "nashconv": float(sum(gains)), "max_gain": float(max(gains))}

    def avg_accumulate(self, probs, weight, S):
        """S += weight * sum_deals P(d) * own reach(d) * probs, every seat (opponents full width)."""
        probs = np.asarray(probs, dtype=np.float64)
        assert S.dtype == np.float64

        def walk(nid, own):          # own: [2, D] each player's own reach
            nd = self.nodes[nid]
            if nd["kind"] == "T":
                return
            i = nd["actor"]
            bk = self.bk[i][nd["street"]]
            base = nd["row0"] + bk * nd["nA"]
            for a, ch in enumerate(nd["children"]):
                pa = probs[base + a]
                np.add.at(S, base + a, weight * self.chance * own[i] * pa)
                o2 = own.copy()
                o2[i] = own[i] * pa
                walk(ch, o2)

        walk(0, np.ones((2, self.D)))

    # ---- sampling ---------------------------------------------------------------------------------------------------
    def _row_probs(self, probs, nodes, deals, actors):
        """[M, 3] action probabilities (zero-padded) at (node, deal) for the acting player."""
        st = self.a_street[nodes]
        b = self.bk_arr[actors, st, deals]
        base = self.a_row0[nodes] + b * self.a_nA[nodes]
        nA = self.a_nA[nodes]
        P = np.zeros((len(nodes), 3))
        for a in range(3):
            m = a < nA
            P[m, a] = probs[base[m] + a]
        return P, b

    @staticmethod
    def _sample(P, rng):
        c = np.cumsum(P, axis=1)
        c[:, -1] = np.inf                       # numerical safety: the last legal action absorbs rounding
        u = rng.random(len(P))[:, None]
        a = (u >= c).sum(1)
        return a

    def generate(self, probs, n_hands, eps, seed, max_out=None):
        """Every seat acts from `probs`; the hero (uniform seat) mixes eps uniformly over legal actions. On each street
        the k-th hero decision replaces the kept snapshot with probability 1/k (reservoir of one). reach_lp sums the
        log of the hero's eps-mixed probability of each action taken BEFORE the kept node."""
        probs = np.asarray(probs, dtype=np.float64)
        rng = np.random.default_rng(seed)
        n = int(n_hands)
        deal = rng.integers(0, self.D, n)
        hero = rng.integers(0, 2, n)
        node = np.zeros(n, dtype=np.int64)
        lp = np.zeros(n)
        S = self.n_streets_game
        kept = np.full((n, S), -1, dtype=np.int64)
        kept_lp = np.zeros((n, S))
        count = np.zeros((n, S), dtype=np.int64)
        alive = ~self.a_term[node]
        while alive.any():
            idx = np.nonzero(alive)[0]
            nd = node[idx]
            act = self.a_actor[nd]
            P, _ = self._row_probs(probs, nd, deal[idx], act)
            isH = act == hero[idx]
            nA = self.a_nA[nd]
            mixed = np.where(isH[:, None], (1 - eps) * P + eps * (np.arange(3)[None, :] < nA[:, None]) / nA[:, None], P)
            st = self.a_street[nd]
            h_idx = idx[isH]
            if len(h_idx):
                s_h = st[isH]
                count[h_idx, s_h] += 1
                k = count[h_idx, s_h]
                keep = (k == 1) | (rng.random(len(h_idx)) < 1.0 / k)
                kept[h_idx[keep], s_h[keep]] = nd[isH][keep]
                kept_lp[h_idx[keep], s_h[keep]] = lp[h_idx[keep]]
            a = self._sample(mixed, rng)
            lp[idx[isH]] += np.log(np.maximum(mixed[isH, a[isH]], 1e-300))
            node[idx] = self.a_child[nd, a]
            alive = ~self.a_term[node]
        hh, ss = np.nonzero(kept >= 0)                          # row-major: per hand, streets ascending
        kn = kept[hh, ss]
        rec = np.zeros((len(hh), REC_I), dtype=np.int64)
        rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3] = hh, hero[hh], ss, kn
        rec[:, 4] = self.bk_arr[hero[hh], ss, deal[hh]]
        rec[:, 5] = count[hh, ss]
        rec[:, 6:9] = self.deals[deal[hh]]
        return Pivots(rec, kept_lp[hh, ss])

    def _pivot_deals(self, piv):
        return np.asarray([self._deal_index[tuple(c)] for c in piv.i[:, 6:9].tolist()], dtype=np.int64)

    def _playout(self, probs, start, deal, hero, rng):
        node = start.copy()
        alive = ~self.a_term[node]
        while alive.any():
            idx = np.nonzero(alive)[0]
            nd = node[idx]
            P, _ = self._row_probs(probs, nd, deal[idx], self.a_actor[nd])
            a = self._sample(P, rng)
            node[idx] = self.a_child[nd, a]
            alive = ~self.a_term[node]
        u0 = self.a_u0[node, deal]
        return np.where(hero == 0, u0, -u0)

    def rollouts(self, probs, piv, n_sims, seed, crn=False):
        """Per legal action, n_sims continuations from the pivot: real private cards kept, the undealt board resampled
        (round-1 pivots), every seat continues from `probs` with no exploration. Returns (q, var), each [n, 16]: mean and
        unbiased sample variance per action. crn: all actions of a pivot share the board draw of scenario j."""
        probs = np.asarray(probs, dtype=np.float64)
        rng = np.random.default_rng(seed)
        n = len(piv)
        ns = np.asarray(n_sims)
        if ns.shape != (n,):
            ns = np.asarray(n_sims, dtype=np.int64)[piv.street]
        deals = self._pivot_deals(piv)
        q = np.zeros((n, 16))
        var = np.zeros((n, 16))
        nodes = piv.node
        # one trajectory per (pivot, action, sim)
        P_i, A_i, J_i = [], [], []
        for k in range(n):
            nA = self.a_nA[nodes[k]]
            for a in range(nA):
                P_i.append(np.full(ns[k], k))
                A_i.append(np.full(ns[k], a))
                J_i.append(np.arange(ns[k]))
        P_i, A_i, J_i = np.concatenate(P_i), np.concatenate(A_i), np.concatenate(J_i)
        start = self.a_child[nodes[P_i], A_i]
        d = deals[P_i].copy()
        if self.game == "leduc":
            redraw = piv.street[P_i] == 0
            if crn:
                draw = rng.integers(0, 4, (n, int(ns.max())))[P_i, J_i]
            else:
                draw = rng.integers(0, 4, len(P_i))
            g = self.private_group[d]
            d = np.where(redraw, self.group_members[g, draw], d)
        v = self._playout(probs, start, d, piv.hero[P_i], rng)
        cell = P_i * 16 + A_i
        cnt = np.bincount(cell, minlength=n * 16)
        s1 = np.bincount(cell, weights=v, minlength=n * 16)
        s2 = np.bincount(cell, weights=v * v, minlength=n * 16)
        m = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
        vv = np.where(cnt > 1, np.maximum(0.0, (s2 - cnt * m * m) / np.maximum(cnt - 1, 1)), 0.0)
        return m.reshape(n, 16), vv.reshape(n, 16)

    def exact_q(self, probs, piv):
        """Exact Q(h, a): expectation over the undealt board (round-1 pivots: the 4 cards nobody holds) and every
        seat's continuation under `probs`; real private cards kept. Fold = -(actor's committed chips)."""
        V = self.node_values(probs)
        n = len(piv)
        deals = self._pivot_deals(piv)
        q = np.zeros((n, 16))
        for k in range(n):
            nid, hero = int(piv.node[k]), int(piv.hero[k])
            d = deals[k]
            ds = self.group_members[self.private_group[d]] if (self.game == "leduc" and piv.street[k] == 0) else [d]
            for a, ch in enumerate(self.nodes[nid]["children"]):
                q[k, a] = float(np.mean(V[ch][hero, ds]))
        return q
