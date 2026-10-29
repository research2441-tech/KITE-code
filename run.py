"""Experiment driver.

python run.py <config-name> <seed> [key=value ...]
Writes results/<config>__s<seed>.json
"""
import copy
import json
import os
import pickle
import sys
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

from cyberrange import (Pools, EdgeRange, N_HOSTS, N_ACT, N_KIND, PROFILE_NAMES, VALUE, W_AVAIL,
                        C_COMP)
from fleet import Fleet
from comm import Exchange
from policy import Planner, RULES
import wm

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["ToN-IoT"] * 4 + ["CICIoT2023"] * 3 + ["CIC-DIAD2024"] * 3

BASE = dict(
    n_sites=10, ex_episodes=30, ep_rot=5, warm=3, horizon=100,
    eval_profiles=["smash", "apt", "worm", "zeroday"], eval_episodes=4,
    # learning
    lr=2e-3, batch=16, buf_cap=400, val_cap=48, val_every=5, val_eval=8, w_sym=0.5,
    train_every=5, train_every_eval=10, online=True,
    dp_sigma=0.0, dp_clip=1.0, delta=1e-5,
    # policy
    policy="wm", tau_c=.6, k_max=3, near_hops=1, n_cand=10, H=5, gamma=.95, beta=.5, lam=.3,
    r_min=.5, rules=True, shield=True, imagine=True, propose="rules", explore_end=.05,
    # exchange
    agg="trust", comm_every=50, k_out=3, p_fail=.1, partition=None,
    byz=[], n_sybil=0, attack="none", att_scale=10.0, alie_z=1.0,
    trust_prior=(1.0, 1.0), forget=.9, xi=.02, neg_w=2.0, tau_min=.55, n_min=2,
    tau_rep=.6, gamma2=.3, eta_c=.5, kappa_s=.95, precision=True, sybil_group=True,
    second_hand=True, bal_gamma=.3, bal_kappa=1.0,
    graph=True, joiners=[], join_ep=0,
)


def playbook(obs, env):
    """Signature-alert playbook: contain alerted hosts, re-image confirmed
    compromise, never touch the gateway or controller without confirmation."""
    S = obs.shape[0]
    a = np.zeros(S, int)
    for s in range(S):
        rev = obs[s, :, -4:]
        conf = np.where(rev[:, 2:].sum(1) > 0)[0]
        if len(conf):
            h = conf[np.argmax(VALUE[conf])]
            a[s] = 1 + h * N_KIND + 2
            continue
        al = np.where((obs[s, :, 24] > 0) | (obs[s, :, 25] > 0))[0]
        al = [h for h in al if env.iso[s, h] == 0 and env.down[s, h] == 0]
        if al:
            h = max(al, key=lambda x: VALUE[x])
            a[s] = 1 + h * N_KIND + (0 if h in (0, 9) else 1)
    return a


def emit_label(emit):
    return np.where(emit >= 0, emit, 0)


def status_of(env):
    return np.stack([env.iso > 0, env.down > 0, env.decoy > 0], -1).astype(np.float32)


def timers_of(env):
    return np.stack([env.iso, env.down, env.decoy], -1)


class Runner:
    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed
        self.rng = np.random.default_rng(seed)
        self.pools = Pools(os.path.join(os.environ.get("KITE_DATA", HERE), "pools.npz"))
        S = cfg["n_sites"]
        self.env_ex = EdgeRange(self.pools, S, DATASETS[:S], seed=seed, split=0, horizon=cfg["horizon"])
        self.env_ev = EdgeRange(self.pools, S, DATASETS[:S], seed=seed, split=1, horizon=cfg["horizon"])
        adj = self.env_ex.A if cfg["graph"] else np.zeros_like(self.env_ex.A)
        torch.manual_seed(seed)
        self.fleet = Fleet(S, adj, seed, cfg)
        if cfg["attack"] == "semantic":
            self.fleet.flip[cfg["byz"]] = True
        self.planner = Planner(self.env_ex.A, cfg)
        self.ex = Exchange(S, cfg, seed)
        self.honest = np.array([s for s in range(S) if s not in cfg["byz"]])
        self.gstep = 0
        self.n_rounds = (cfg["ex_episodes"] + len(cfg["eval_profiles"]) * cfg["eval_episodes"]) \
            * cfg["horizon"] // cfg["comm_every"]
        self.round = 0
        self.timing = dict(plan=0.0, train=0.0, comm=0.0, n_plan=0, n_train=0, n_comm=0)

    # --------------------------------------------------------------
    def choose(self, env, obs, explore, record=False):
        cfg = self.cfg
        if cfg["policy"] == "playbook":
            return playbook(obs, env), None
        t = time.time()
        a, info = self.planner.act(self.fleet.SH, self.fleet.z, timers_of(env), self.rng,
                                   explore=explore, record=record)
        self.timing["plan"] += time.time() - t
        self.timing["n_plan"] += 1
        return a, info

    def tick(self, train_every):
        cfg = self.cfg
        self.gstep += 1
        if cfg["policy"] == "playbook":
            return
        if self.gstep % train_every == 0:
            t = time.time()
            self.fleet.train_step()
            self.timing["train"] += time.time() - t
            self.timing["n_train"] += 1
        if self.gstep % cfg["comm_every"] == 0:
            t = time.time()
            self.ex.round(self.round, self.n_rounds, self.fleet)
            self.round += 1
            self.timing["comm"] += time.time() - t
            self.timing["n_comm"] += 1

    # --------------------------------------------------------------
    def exercise(self):
        cfg, env, F = self.cfg, self.env_ex, self.fleet
        S = cfg["n_sites"]
        curve = []
        for ep in range(cfg["ex_episodes"]):
            act = np.ones(S, bool)
            if cfg["joiners"] and ep < cfg["join_ep"]:
                act[cfg["joiners"]] = False
            F.active = act
            self.ex.active = act
            prof = [PROFILE_NAMES[(s + ep // cfg["ep_rot"]) % 3] for s in range(S)]
            obs = env.reset(prof)
            F.reset_filter()
            prev = np.zeros(S, int)
            st = status_of(env)
            F.observe(obs, prev, st, env.lvl.copy(), np.zeros((S, N_HOSTS)), np.ones((S, N_HOSTS)),
                      np.ones((S, N_HOSTS)))
            if ep < cfg["warm"]:
                explore = 1.0
            else:
                explore = max(cfg["explore_end"], .3 * (1 - ep / cfg["ex_episodes"]))
            done = False
            while not done:
                if cfg["policy"] == "playbook":
                    a = playbook(obs, env)
                elif ep < cfg["warm"] and self.rng.random() < .5:
                    a = playbook(obs, env)
                    m = self.rng.random(S) < .3
                    a[m] = self.rng.integers(0, N_ACT, m.sum())
                else:
                    a, _ = self.choose(env, obs, explore)
                obs, r, done, info = env.step(a)
                st = status_of(env)
                F.observe(obs, a, st, info["lvl"], emit_label(info["emit"]), np.ones((S, N_HOSTS)),
                          np.ones((S, N_HOSTS)))
                self.tick(cfg["train_every"])
            sm = env.summary()
            curve.append(dict(ep=ep, ret=float(sm["ret"][self.honest].mean()),
                              loss=F.eval_parts()[self.honest].mean(0).round(4).tolist()
                              if cfg["policy"] != "playbook" and F.n_val.min() > 0 else None))
        return curve

    # --------------------------------------------------------------
    def evaluate(self):
        cfg, env, F = self.cfg, self.env_ev, self.fleet
        S = cfg["n_sites"]
        F.active = np.ones(S, bool)
        self.ex.active = F.active
        res = {}
        for prof in cfg["eval_profiles"]:
            eps = []
            pf, yf, pl, yl = [], [], [], []
            unc = []
            for e in range(cfg["eval_episodes"]):
                obs = env.reset([prof] * S)
                F.reset_filter()
                st = status_of(env)
                F.observe(obs, np.zeros(S, int), st)
                done = False
                stepcost = []
                decisions = []
                while not done:
                    a, info = self.choose(env, obs, 0.0, record=(e == 0))
                    if info is not None:
                        pf.append(info["bel"][self.honest, :, 2:].sum(-1).ravel())
                        yf.append((env.lvl[self.honest] >= 2).ravel())
                        pl.append(info["bel"][self.honest].argmax(-1).ravel())
                        yl.append(env.lvl[self.honest].ravel())
                    obs, r, done, inf2 = env.step(a)
                    stepcost.append(C_COMP * (inf2["comp"] * W_AVAIL).sum(1) + inf2["dmg"])
                    st = status_of(env)
                    rev = obs[:, :, -4:]
                    lm = (rev.sum(-1) > 0).astype(np.float32)
                    lvl = rev.argmax(-1)
                    if cfg["online"]:
                        F.observe(obs, a, st, lvl, None, lm, None)
                    else:
                        F.observe(obs, a, st)
                    if cfg["online"]:
                        self.tick(cfg["train_every_eval"])
                sm = env.summary()
                rec = {k: sm[k][self.honest].tolist() for k in
                       ("ret", "comp_steps", "exfil", "loss", "breach", "avail", "false_iso",
                        "false_rst", "n_iso", "n_rst", "ttc_all", "crit_comp")}
                eps.append(rec)
            out = dict(episodes=eps)
            if pf:
                p, y = np.concatenate(pf), np.concatenate(yf)
                out["diverged"] = bool(not np.all(np.isfinite(p)))
                p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0).clip(0, 1)
                out["auroc_foot"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None
                out["brier_foot"] = float(np.mean((p - y) ** 2))
                bins = np.clip((p * 10).astype(int), 0, 9)
                out["ece_foot"] = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
                                            for b in range(10) if (bins == b).any()))
                out["f1_lvl"] = float(f1_score(np.concatenate(yl), np.concatenate(pl), average="macro"))
            res[prof] = out
        return res

    def run(self):
        t0 = time.time()
        curve = self.exercise()
        t_ex = time.time() - t0
        ckpt = None
        res = self.evaluate()
        eps, order = self.fleet.privacy(self.cfg["delta"])
        trace = self.planner.trace
        rule_use = {}
        for d in trace:
            rule_use[d["rule"]] = rule_use.get(d["rule"], 0) + 1
        return dict(cfg={k: v for k, v in self.cfg.items()}, seed=self.seed, curve=curve, eval=res,
                    honest=self.honest.tolist(),
                    privacy=dict(eps=eps, order=order, steps=self.fleet.dp_steps),
                    comm=self.ex.log, timing=self.timing, wall=time.time() - t0, wall_ex=t_ex,
                    rule_use=rule_use, trace_sample=trace[:200])


CONFIGS = {}


def register(name, **kw):
    CONFIGS[name] = kw


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    cfg = copy.deepcopy(BASE)
    cfg.update(CONFIGS.get(name, {}))
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    out = Runner(cfg, seed).run()
    os.makedirs(os.environ.get("KITE_RESULTS", os.path.join(HERE, "results")), exist_ok=True)
    tag = name + "".join("_" + kv.replace("=", "-") for kv in sys.argv[3:])
    with open(os.path.join(HERE, "results", f"{tag}__s{seed}.json"), "w") as f:
        json.dump(out, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    ev = out["eval"]
    print(name, seed, "wall %.0fs" % out["wall"],
          {p: round(float(np.mean([np.mean(e["ret"]) for e in ev[p]["episodes"]])), 2) for p in ev},
          "eps", out["privacy"]["eps"])


CONFIGS.update({
    "proposed": {},
    "playbook": dict(policy="playbook"),
    "wm_local": dict(agg="local"),
    "wm_gossip": dict(agg="gossip"),
    "wm_fedavg": dict(agg="fedavg"),
    "wm_uniform": dict(agg="uniform"),
    # ablations
    "abl_nograph": dict(graph=False),
    "abl_nosym": dict(w_sym=0.0),
    "abl_noimag": dict(imagine=False),
    "abl_norules": dict(rules=False, propose="all", lam=0.0),
    "abl_noshield": dict(shield=False),
    "abl_nounc": dict(beta=0.0),
    "abl_offline": dict(online=False),
    # sensitivity
    "sens_H1": dict(H=1), "sens_H3": dict(H=3), "sens_H8": dict(H=8),
    "sens_lam01": dict(lam=0.1), "sens_lam1": dict(lam=1.0),
    "sens_beta1": dict(beta=1.0), "sens_cand5": dict(n_cand=5), "sens_cand20": dict(n_cand=20),
    # privacy
    "dp_s05": dict(dp_sigma=0.5), "dp_s10": dict(dp_sigma=1.0), "dp_s20": dict(dp_sigma=2.0),
})
SCEN = {
    "signflip": dict(byz=[3, 6], attack="signflip"),
    "alie": dict(byz=[3, 6], attack="alie"),
    "semantic": dict(byz=[3, 6], attack="semantic"),
    "sybil": dict(byz=[3], attack="semantic", n_sybil=5),
    "topo": dict(p_fail=0.5, partition=[0.3, 0.7]),
}
for sc, kw in SCEN.items():
    for ag in ["trust", "gossip", "fedavg", "median", "krum", "fltrust", "balance", "uniform"]:
        CONFIGS[f"rob_{sc}_{ag}"] = dict(kw, agg=ag)
    for comp in ["precision", "sybil_group", "second_hand"]:
        CONFIGS[f"rob_{sc}_no{comp}"] = dict(kw, agg="trust", **{comp: False})
for ag in ["trust", "local", "gossip", "fedavg"]:
    CONFIGS[f"join_{ag}"] = dict(agg=ag, joiners=[7, 8, 9], join_ep=24)

if __name__ == "__main__":
    main()
