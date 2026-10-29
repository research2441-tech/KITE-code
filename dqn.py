"""Model-free baselines: host-factored graph double DQN.

Q(s, noop) from a pooled site embedding, Q(s, 1 + 4h + k) from host h's
embedding (own telemetry of the last three steps, neighbour mean, role).
Variants: local (one network per site), fedavg (server averaging every
comm_every steps), aux (an auxiliary head trained on red-team stage labels).

python dqn.py <name> <seed> [key=value ...]
"""
import copy
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from cyberrange import Pools, EdgeRange, N_HOSTS, N_ACT, PROFILE_NAMES, OBS_DIM, ROLE, W_AVAIL, C_COMP
from run import DATASETS, HERE, playbook, status_of
import wm

torch.set_num_threads(1)
ROLE_OH = F.one_hot(torch.as_tensor(ROLE), 5).float()
IN = 3 * OBS_DIM + 5 + 3 * OBS_DIM

BASE = dict(n_sites=10, ex_episodes=30, ep_rot=5, horizon=100, warm=3,
            eval_profiles=["smash", "apt", "worm", "zeroday"], eval_episodes=4,
            lr=1e-3, batch=64, gamma=.95, buf=20000, target_every=200, train_every=1,
            eps_start=1.0, eps_end=.05, fed=False, comm_every=50, aux=False, hidden=96)


def init(seed, S, hidden):
    g = torch.Generator().manual_seed(seed)

    def lin(i, o):
        return torch.randn(i, o, generator=g) / np.sqrt(i), torch.zeros(o)
    P = {}
    for name, (i, o) in dict(l1=(IN, hidden), l2=(hidden, hidden), q=(hidden, 4), q0=(hidden, 1),
                             aux=(hidden, 4)).items():
        W, b = lin(i, o)
        P[name + ".W"], P[name + ".b"] = W, b
    return {k: v[None].repeat(S, *[1] * v.dim()).clone() for k, v in P.items()}


def features(stack, adj):
    """stack (S,3,N,O) -> (S,N,IN)."""
    S = stack.shape[0]
    x = stack.permute(0, 2, 1, 3).reshape(S, N_HOSTS, 3 * OBS_DIM)
    deg = adj.sum(-1, keepdim=True).clamp(min=1)
    nb = (adj @ x) / deg
    return torch.cat([x, ROLE_OH.expand(S, -1, -1), nb], -1)


def qvals(P, x):
    h = F.elu(wm.L(P, "l1", x))
    h = F.elu(wm.L(P, "l2", h))
    qh = wm.L(P, "q", h)  # (S,...,N,4)
    q0 = wm.L(P, "q0", h.mean(-2))  # (S,...,1)
    return torch.cat([q0, qh.flatten(-2)], -1), wm.L(P, "aux", h)


class Adam:
    def __init__(self, P, lr):
        self.m = {k: torch.zeros_like(v) for k, v in P.items()}
        self.v = {k: torch.zeros_like(v) for k, v in P.items()}
        self.t, self.lr = 0, lr

    def step(self, P, G):
        self.t += 1
        for k in P:
            self.m[k] = .9 * self.m[k] + .1 * G[k]
            self.v[k] = .999 * self.v[k] + .001 * G[k] ** 2
            P[k] = P[k] - self.lr * (self.m[k] / (1 - .9 ** self.t)) / ((self.v[k] / (1 - .999 ** self.t)).sqrt() + 1e-8)


class DQNRunner:
    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed
        S = cfg["n_sites"]
        self.rng = np.random.default_rng(seed)
        pools = Pools(os.path.join(os.environ.get("KITE_DATA", HERE), "pools.npz"))
        self.env_ex = EdgeRange(pools, S, DATASETS[:S], seed=seed, split=0, horizon=cfg["horizon"])
        self.env_ev = EdgeRange(pools, S, DATASETS[:S], seed=seed, split=1, horizon=cfg["horizon"])
        self.adj = torch.as_tensor(self.env_ex.A, dtype=torch.float32)
        self.P = init(seed, S, cfg["hidden"])
        if cfg["fed"]:
            for k in self.P:
                self.P[k][:] = self.P[k][0]
        self.T = {k: v.clone() for k, v in self.P.items()}
        self.opt = Adam(self.P, cfg["lr"])
        n = cfg["buf"]
        self.B = dict(x=np.zeros((S, n, 3, N_HOSTS, OBS_DIM), np.float32), a=np.zeros((S, n), np.int64),
                      r=np.zeros((S, n), np.float32), x2=np.zeros((S, n, 3, N_HOSTS, OBS_DIM), np.float32),
                      d=np.zeros((S, n), np.float32), lvl=np.zeros((S, n, N_HOSTS), np.int64))
        self.nb = 0
        self.steps = 0
        self.t_train = 0.0

    def act(self, stack, eps):
        with torch.no_grad():
            q, _ = qvals(self.P, features(torch.as_tensor(stack), self.adj))
        a = q.argmax(-1).numpy()
        m = self.rng.random(len(a)) < eps
        a[m] = self.rng.integers(0, N_ACT, m.sum())
        return a

    def train(self):
        cfg = self.cfg
        n = min(self.nb, cfg["buf"])
        if n < cfg["batch"]:
            return
        t = time.time()
        S, Bn = cfg["n_sites"], cfg["batch"]
        idx = self.rng.integers(0, n, (S, Bn))
        si = np.arange(S)[:, None]
        x = torch.as_tensor(self.B["x"][si, idx])
        x2 = torch.as_tensor(self.B["x2"][si, idx])
        a = torch.as_tensor(self.B["a"][si, idx])
        r = torch.as_tensor(self.B["r"][si, idx])
        d = torch.as_tensor(self.B["d"][si, idx])
        adj = self.adj[:, None]
        f = lambda s: torch.cat([s.permute(0, 1, 3, 2, 4).reshape(S, Bn, N_HOSTS, -1),
                                 ROLE_OH.expand(S, Bn, -1, -1),
                                 (adj @ s.permute(0, 1, 3, 2, 4).reshape(S, Bn, N_HOSTS, -1))
                                 / adj.sum(-1, keepdim=True).clamp(min=1)], -1)
        with torch.no_grad():
            qn, _ = qvals(self.P, f(x2))
            qt, _ = qvals(self.T, f(x2))
            tgt = r + cfg["gamma"] * (1 - d) * qt.gather(-1, qn.argmax(-1, keepdim=True)).squeeze(-1)
        P = {k: v.detach().requires_grad_() for k, v in self.P.items()}
        q, aux = qvals(P, f(x))
        loss = F.smooth_l1_loss(q.gather(-1, a[..., None]).squeeze(-1), tgt, reduction="none").mean(1)
        if cfg["aux"]:
            lv = torch.as_tensor(self.B["lvl"][si, idx])
            loss = loss + F.cross_entropy(aux.reshape(-1, 4), lv.reshape(-1), reduction="none").reshape(S, -1).mean(1)
        G = dict(zip(P, [g if g is not None else torch.zeros_like(v) for g, v in zip(torch.autograd.grad(loss.sum(), list(P.values()), allow_unused=True), P.values())]))
        self.opt.step(self.P, G)
        self.steps += 1
        if self.steps % cfg["target_every"] == 0:
            self.T = {k: v.clone() for k, v in self.P.items()}
        self.t_train += time.time() - t

    def fedavg(self):
        for k in self.P:
            self.P[k][:] = self.P[k].mean(0, keepdim=True)

    def exercise(self):
        cfg, env = self.cfg, self.env_ex
        S = cfg["n_sites"]
        total = cfg["ex_episodes"] * cfg["horizon"]
        g = 0
        for ep in range(cfg["ex_episodes"]):
            prof = [PROFILE_NAMES[(s + ep // cfg["ep_rot"]) % 3] for s in range(S)]
            obs = env.reset(prof)
            stack = np.repeat(obs[:, None], 3, 1)
            done = False
            while not done:
                eps = max(cfg["eps_end"], cfg["eps_start"] * (1 - g / (0.6 * total)))
                a = self.act(stack, eps)
                obs, r, done, info = env.step(a)
                st2 = np.concatenate([stack[:, 1:], obs[:, None]], 1)
                i = self.nb % cfg["buf"]
                self.B["x"][:, i], self.B["a"][:, i], self.B["r"][:, i] = stack, a, r
                self.B["x2"][:, i], self.B["d"][:, i], self.B["lvl"][:, i] = st2, float(done), info["lvl"]
                self.nb += 1
                stack = st2
                g += 1
                if g % cfg["train_every"] == 0:
                    self.train()
                if cfg["fed"] and g % cfg["comm_every"] == 0:
                    self.fedavg()

    def evaluate(self):
        cfg, env = self.cfg, self.env_ev
        S = cfg["n_sites"]
        res = {}
        for prof in cfg["eval_profiles"]:
            eps = []
            for e in range(cfg["eval_episodes"]):
                obs = env.reset([prof] * S)
                stack = np.repeat(obs[:, None], 3, 1)
                done = False
                while not done:
                    a = self.act(stack, 0.0)
                    obs, r, done, info = env.step(a)
                    stack = np.concatenate([stack[:, 1:], obs[:, None]], 1)
                sm = env.summary()
                eps.append({k: sm[k].tolist() for k in ("ret", "comp_steps", "exfil", "loss", "breach", "avail",
                                                        "false_iso", "false_rst", "n_iso", "n_rst", "ttc_all",
                                                        "crit_comp")})
            res[prof] = dict(episodes=eps)
        return res

    def run(self):
        t0 = time.time()
        self.exercise()
        res = self.evaluate()
        n_par = sum(v[0].numel() for v in self.P.values())
        return dict(cfg=self.cfg, seed=self.seed, eval=res, wall=time.time() - t0,
                    timing=dict(train=self.t_train, n_train=self.steps), n_params=n_par)


CONFIGS = {
    "dqn_local": {},
    "dqn_fedavg": dict(fed=True),
    "dqn_aux": dict(aux=True),
    "dqn_fedavg_aux": dict(fed=True, aux=True),
    "dqn_local_4x": dict(ex_episodes=120, ep_rot=20),
}

if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    cfg = copy.deepcopy(BASE)
    cfg.update(CONFIGS.get(name, {}))
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    out = DQNRunner(cfg, seed).run()
    tag = name + "".join("_" + kv.replace("=", "-") for kv in sys.argv[3:])
    os.makedirs(os.environ.get("KITE_RESULTS", os.path.join(HERE, "results")), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{tag}__s{seed}.json"), "w") as f:
        json.dump(out, f)
    ev = out["eval"]
    print(name, seed, "wall %.0fs" % out["wall"],
          {p: round(float(np.mean([np.mean(e["ret"]) for e in ev[p]["episodes"]])), 2) for p in ev})
