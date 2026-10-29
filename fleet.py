"""Fleet of site agents: replay, (DP-)training, privacy accounting and
trust-aware decentralised exchange over time-varying directed links."""
import math
import numpy as np
import torch
import wm
from cyberrange import N_HOSTS, OBS_DIM

CHUNK = 12


# ---------------------------------------------------------------- privacy
def rdp_sampled_gaussian(q, sigma, orders):
    """RDP of the Poisson-subsampled Gaussian mechanism for integer orders
    (binomial expansion, exact for integer alpha)."""
    out = []
    for a in orders:
        if q == 0:
            out.append(0.0)
            continue
        if q == 1.0:
            out.append(a / (2 * sigma ** 2))
            continue
        logs = []
        for k in range(a + 1):
            lb = math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)
            logs.append(lb + (a - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma ** 2))
        m = max(logs)
        out.append((m + math.log(sum(math.exp(x - m) for x in logs))) / (a - 1))
    return np.array(out)


ORDERS = list(range(2, 65))


def eps_from_rdp(rdp, delta):
    o = np.array(ORDERS, float)
    e = rdp + math.log(1 / delta) / (o - 1)
    return float(e.min()), int(o[e.argmin()])


# ---------------------------------------------------------------- tensors
def stack_params(dicts):
    keys = dicts[0].keys()
    return {k: torch.stack([d[k] for d in dicts]) for k in keys}


def flatvec(P):
    return torch.cat([v.reshape(v.shape[0], -1) for v in P.values()], 1)


def unflatvec(vec, like):
    out, i = {}, 0
    for k, v in like.items():
        n = v[0].numel()
        out[k] = vec[:, i:i + n].reshape((vec.shape[0],) + v.shape[1:])
        i += n
    return out


class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, mask=None):
        self.t += 1
        b1, b2 = .9, .999
        for k in params:
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            upd = self.lr * mh / (vh.sqrt() + 1e-8)
            if mask is not None:
                upd = upd * mask.view(-1, *[1] * (upd.dim() - 1))
            params[k] = params[k] - upd


class Fleet:
    def __init__(self, n_sites, adj, seed, cfg):
        self.S, self.cfg = n_sites, cfg
        self.adj = torch.as_tensor(adj, dtype=torch.float32)
        sh, lo = wm.init_params(seed)  # common initialisation for all sites
        self.SH = {k: v.unsqueeze(0).repeat(n_sites, *[1] * v.dim()).clone() for k, v in sh.items()}
        self.LO = {k: v.unsqueeze(0).repeat(n_sites, *[1] * v.dim()).clone() for k, v in lo.items()}
        self.opt_sh = Adam(self.SH, cfg["lr"])
        self.opt_lo = Adam(self.LO, cfg["lr"])
        cap = cfg["buf_cap"]
        S = n_sites
        self.buf = dict(obs=np.zeros((S, cap, CHUNK, N_HOSTS, OBS_DIM), np.float32),
                        aoh=np.zeros((S, cap, CHUNK, N_HOSTS, 4), np.float32),
                        st=np.zeros((S, cap, CHUNK, N_HOSTS, 3), np.float32),
                        lvl=np.zeros((S, cap, CHUNK, N_HOSTS), np.int8),
                        emit=np.zeros((S, cap, CHUNK, N_HOSTS), np.int8),
                        lmask=np.zeros((S, cap, CHUNK, N_HOSTS), np.float32),
                        emask=np.zeros((S, cap, CHUNK, N_HOSTS), np.float32))
        self.vcap = cfg["val_cap"]
        self.val = {k: np.zeros((S, self.vcap) + v.shape[2:], v.dtype) for k, v in self.buf.items()}
        self.n_buf = np.zeros(S, int)
        self.n_val = np.zeros(S, int)
        self.n_chunks_seen = 0
        self.cur = None
        self.rdp = np.zeros(len(ORDERS))
        self.dp_steps = 0
        self.rng = np.random.default_rng(seed + 11)
        self.z = torch.zeros(S, N_HOSTS, wm.D_Z)
        self.flip = np.zeros(S, bool)  # label-flipping (semantic poisoning) sites
        self.log_parts = []
        self.train_mask = None  # optional per-site mask (sites that do not learn)
        self.active = np.ones(n_sites, bool)

    # ------------------------------------------------------------ filtering
    def reset_filter(self):
        self.z = torch.zeros(self.S, N_HOSTS, wm.D_Z)
        self.cur = {k: [] for k in self.buf}

    def observe(self, obs, prev_action, status, lvl=None, emit=None, lmask=None, emask=None):
        aoh = wm.act_onehot(torch.as_tensor(prev_action)).numpy()
        with torch.no_grad():
            self.z = wm.posterior_step({**self.SH, **self.LO}, self.z, torch.as_tensor(obs),
                                       torch.as_tensor(aoh), torch.as_tensor(status, dtype=torch.float32),
                                       self.adj)
        S = self.S
        lvl = np.zeros((S, N_HOSTS)) if lvl is None else lvl
        emit = np.zeros((S, N_HOSTS)) if emit is None else emit
        lmask = np.zeros((S, N_HOSTS)) if lmask is None else lmask
        emask = np.zeros((S, N_HOSTS)) if emask is None else emask
        if self.flip.any():  # poisoned sites learn that compromise looks clean
            lvl = lvl.copy()
            emit = emit.copy()
            lvl[self.flip] = np.where(lvl[self.flip] >= 2, 0, lvl[self.flip])
            emit[self.flip] = np.where(emit[self.flip] >= 4, 0, emit[self.flip])
        for k, v in zip(self.buf, (obs, aoh, status, lvl, emit, lmask, emask)):
            self.cur[k].append(v)
        if len(self.cur["obs"]) == CHUNK:
            self._store()

    def _store(self):
        self.n_chunks_seen += 1
        to_val = (self.n_chunks_seen % self.cfg["val_every"] == 0)
        for s in range(self.S):
            if not self.active[s]:
                continue
            if to_val:
                i = self.n_val[s] % self.vcap
                for k in self.buf:
                    self.val[k][s, i] = np.stack([x[s] for x in self.cur[k]])
                self.n_val[s] += 1
            else:
                i = self.n_buf[s] % self.cfg["buf_cap"]
                for k in self.buf:
                    self.buf[k][s, i] = np.stack([x[s] for x in self.cur[k]])
                self.n_buf[s] += 1
        self.cur = {k: [] for k in self.buf}

    # ------------------------------------------------------------ training
    def _batch(self, idx, src):
        S = self.S
        b = {k: torch.as_tensor(src[k][np.arange(S)[:, None], idx]) for k in src}
        for k in ("lvl", "emit"):
            b[k] = b[k].long()
        return b

    def train_step(self):
        cfg, S = self.cfg, self.S
        n = np.minimum(self.n_buf, cfg["buf_cap"])
        act = self.active
        if n[act].min() < cfg["batch"]:
            return
        n = np.maximum(n, 1)
        self.train_mask = torch.as_tensor(act & (np.minimum(self.n_buf, cfg["buf_cap"]) >= cfg["batch"]),
                                          dtype=torch.float32)
        B = cfg["batch"]
        if cfg["dp_sigma"] > 0:
            # Poisson subsampling at rate q; per-example gradients by replicating
            # the parameters once per sampled chunk.
            q = B / n[act].min()
            Bmax = 2 * B
            idx = np.zeros((S, Bmax), int)
            msk = np.zeros((S, Bmax), np.float32)
            for s in range(S):
                sel = np.where(self.rng.random(n[s]) < q)[0][:Bmax]
                idx[s, :len(sel)] = sel
                msk[s, :len(sel)] = 1
            batch = self._batch(idx, self.buf)
            batch = {k: v.reshape((S * Bmax, 1) + v.shape[2:]) for k, v in batch.items()}
            ew = torch.as_tensor(self.rng.poisson(1.0, (S * Bmax, 1, wm.K_ENS)), dtype=torch.float32)
            Pr = {k: v.repeat_interleave(Bmax, 0).requires_grad_() for k, v in {**self.SH, **self.LO}.items()}
            loss, _ = wm.seq_loss(Pr, batch, self.adj.repeat_interleave(Bmax, 0), cfg["w_sym"], ew)
            G = dict(zip(Pr, torch.autograd.grad(loss.sum(), list(Pr.values()))))
            C = cfg["dp_clip"]
            norms = torch.sqrt(sum((g.reshape(S * Bmax, -1) ** 2).sum(-1) for g in G.values()))
            scale = torch.clamp(C / (norms + 1e-12), max=1.0) * torch.as_tensor(msk).reshape(-1)
            agg = {}
            for k, g in G.items():
                sg = (g * scale.view(-1, *[1] * (g.dim() - 1))).reshape((S, Bmax) + g.shape[1:]).sum(1)
                agg[k] = (sg + torch.randn_like(sg) * cfg["dp_sigma"] * C) / B
            self.rdp += rdp_sampled_gaussian(q, cfg["dp_sigma"], ORDERS)
            self.dp_steps += 1
        else:
            idx = np.stack([self.rng.integers(0, n[s], B) for s in range(S)])
            batch = self._batch(idx, self.buf)
            ew = torch.as_tensor(self.rng.poisson(1.0, (S, B, wm.K_ENS)), dtype=torch.float32)
            Pr = {k: v.detach().requires_grad_() for k, v in {**self.SH, **self.LO}.items()}
            loss, parts = wm.seq_loss(Pr, batch, self.adj, cfg["w_sym"], ew)
            agg = dict(zip(Pr, torch.autograd.grad(loss.sum(), list(Pr.values()))))
        self.opt_sh.step(self.SH, {k: agg[k] for k in self.SH}, self.train_mask)
        self.opt_lo.step(self.LO, {k: agg[k] for k in self.LO}, self.train_mask)

    def _val_batch(self, owner, nv):
        cnt = np.minimum(self.n_val, self.vcap)
        idx = np.stack([np.arange(nv) % max(1, cnt[o]) for o in owner])
        b = {k: torch.as_tensor(self.val[k][np.asarray(owner)[:, None], idx]) for k in self.val}
        for k in ("lvl", "emit"):
            b[k] = b[k].long()
        return b

    def eval_loss(self, SHc, owner, n_val=None):
        """Validation loss of candidate shared params SHc (E,...) combined with
        the local params of site owner[e], on owner's validation reservoir."""
        nv = n_val or self.cfg["val_eval"]
        owner = np.asarray(owner)
        b = self._val_batch(owner, nv)
        Pc = {**SHc, **{k: v[owner] for k, v in self.LO.items()}}
        with torch.no_grad():
            l, _ = wm.seq_loss(Pc, b, self.adj[owner], self.cfg["w_sym"])
        return l

    def eval_parts(self):
        b = self._val_batch(np.arange(self.S), self.cfg["val_eval"])
        with torch.no_grad():
            _, p = wm.seq_loss({**self.SH, **self.LO}, b, self.adj, self.cfg["w_sym"])
        return p.numpy()

    def privacy(self, delta=1e-5):
        if self.dp_steps == 0:
            return float("inf"), 0
        return eps_from_rdp(self.rdp, delta)
