"""Decentralised exchange of the shared world-model core.

Identities 0..S-1 are sites; identities S..S+n_sybil-1 are Sybil identities
run by the first Byzantine site. Every round each identity sends its shared
parameters (and its innovation since its previous broadcast) to k_out randomly
chosen sites; links fail independently and may be cut by a partition.
"""
import numpy as np
import torch
import wm
from fleet import flatvec, unflatvec


def vec_of(SH):
    return flatvec(SH)


def set_vec(SH, V):
    new = unflatvec(V, SH)
    for k in SH:
        SH[k] = new[k].contiguous()


class Exchange:
    def __init__(self, n_sites, cfg, seed):
        self.S, self.cfg = n_sites, cfg
        self.rng = np.random.default_rng(seed + 101)
        self.byz = np.array(cfg.get("byz", []), int)
        self.n_syb = cfg.get("n_sybil", 0)
        self.I = n_sites + self.n_syb
        self.is_byz = np.zeros(self.I, bool)
        self.is_byz[self.byz] = True
        self.is_byz[n_sites:] = True
        a0, b0 = cfg["trust_prior"]
        self.alpha = np.full((n_sites, self.I), a0, float)
        self.beta = np.full((n_sites, self.I), b0, float)
        self.nfirst = np.zeros((n_sites, self.I))
        self.prev = None
        self.prev_mal = None
        g = torch.Generator().manual_seed(4242)  # public probe set
        npb = 6
        self.probe = dict(z=torch.randn(npb, wm.N_HOSTS, wm.D_Z, generator=g) * .5,
                          st=(torch.rand(npb, wm.N_HOSTS, 3, generator=g) < .15).float(),
                          a=(torch.rand(npb, wm.N_HOSTS, 4, generator=g) < .05).float())
        self.log = []

    # ---------------------------------------------------------------- graph
    def links(self, r, n_rounds):
        cfg, S = self.cfg, self.S
        part = cfg.get("partition")
        inbox = [[] for _ in range(S)]
        act = getattr(self, "active", np.ones(S, bool))
        for j in range(self.I):
            if j < S and not act[j]:
                continue
            pool = [i for i in range(S) if i != j and act[i]]
            if not pool:
                continue
            outs = self.rng.choice(pool, size=min(cfg["k_out"], len(pool)), replace=False)
            for i in outs:
                if self.rng.random() < cfg["p_fail"]:
                    continue
                if part and part[0] <= r / n_rounds < part[1] and j < S:
                    if (i < S // 2) != (j < S // 2):
                        continue
                inbox[i].append(j)
        return inbox

    # ---------------------------------------------------------------- attack
    def craft(self, V, D):
        """Messages (theta, innovation) of all identities."""
        cfg, S = self.cfg, self.S
        hon = np.where(~self.is_byz[:S])[0]
        mu, sd = V[hon].mean(0), V[hon].std(0)
        dbar = D[hon].mean(0)
        att = cfg.get("attack", "none")
        theta = torch.zeros(self.I, V.shape[1])
        theta[:S] = V
        if att == "signflip":
            mal = mu - cfg["att_scale"] * dbar
        elif att == "alie":
            mal = mu - cfg.get("alie_z", 1.0) * sd
        elif att == "noise":
            mal = mu + torch.randn_like(mu) * 3 * sd
        else:
            mal = None
        for j in range(self.I):
            if not self.is_byz[j]:
                continue
            if j < S and att == "semantic":
                continue  # poisoned site shares its own (label-flipped) model honestly
            if mal is None:
                src = self.byz[0] if len(self.byz) else 0
                theta[j] = V[src] + torch.randn_like(mu) * 1e-3 * sd
            else:
                theta[j] = mal + torch.randn_like(mu) * 1e-3 * sd
        if self.prev_mal is None:
            self.prev_mal = theta.clone()
        innov = theta - self.prev_mal
        innov[:S] = D
        self.prev_mal = theta.clone()
        return theta, innov

    # ---------------------------------------------------------------- trust helpers
    def disagreement(self, SHc):
        E = next(iter(SHc.values())).shape[0]
        z = self.probe["z"][None].expand(E, -1, -1, -1)
        st = self.probe["st"][None].expand(E, -1, -1, -1)
        a = self.probe["a"][None].expand(E, -1, -1, -1)
        adj = torch.ones(E, wm.N_HOSTS, wm.N_HOSTS) - torch.eye(wm.N_HOSTS)
        with torch.no_grad():
            outs = torch.stack([wm.prior_step(SHc, k, z, a, st, adj) for k in range(wm.K_ENS)])
        return outs.var(0).mean((1, 2, 3)) + 1e-6

    # ---------------------------------------------------------------- round
    def round(self, r, n_rounds, fleet):
        cfg, S = self.cfg, self.S
        V = vec_of(fleet.SH)
        if self.prev is None:
            self.prev = V.clone()
        D = V - self.prev
        self.prev = V.clone()
        theta, innov = self.craft(V, D)
        inbox = self.links(r, n_rounds)
        rule = cfg["agg"]
        newV = V.clone()
        byz_mass = []
        act = getattr(self, "active", np.ones(S, bool))
        receivers = [i for i in range(S) if not self.is_byz[i] and act[i]]
        if rule == "local":
            pass
        elif rule == "fedavg":
            ids = [j for j in range(self.I) if j >= S or act[j]]
            avg = theta[ids].mean(0)
            for i in receivers:
                newV[i] = avg
            byz_mass = [self.is_byz[ids].mean()] * len(receivers)
        elif rule == "trust":
            newV, byz_mass = self._trust(V, theta, innov, inbox, receivers, fleet)
        else:
            for i in receivers:
                ids = inbox[i]
                if not ids:
                    byz_mass.append(0.0)
                    continue
                C = torch.cat([V[i:i + 1], theta[ids]])
                if rule == "gossip":
                    newV[i] = C.mean(0)
                    w = np.full(len(ids), 1 / (len(ids) + 1))
                elif rule == "uniform":
                    newV[i] = (1 - cfg["eta_c"]) * V[i] + cfg["eta_c"] * theta[ids].mean(0)
                    w = np.full(len(ids), cfg["eta_c"] / len(ids))
                elif rule == "median":
                    newV[i] = C.median(0).values
                    w = np.full(len(ids), 1 / (len(ids) + 1))  # nominal share, median is not a weighted rule
                elif rule == "trimmed":
                    if C.shape[0] >= 3:
                        srt = C.sort(0).values
                        newV[i] = srt[1:-1].mean(0)
                    else:
                        newV[i] = C.mean(0)
                    w = np.full(len(ids), 1 / (len(ids) + 1))
                elif rule == "krum":
                    n, f = C.shape[0], 1
                    d2 = torch.cdist(C, C) ** 2
                    m = max(1, n - f - 2)
                    sc = torch.stack([d2[a].sort().values[1:m + 1].sum() for a in range(n)])
                    best = int(sc.argmin())
                    newV[i] = 0.5 * V[i] + 0.5 * C[best]
                    w = np.zeros(len(ids))
                    if best > 0:
                        w[best - 1] = 0.5
                elif rule == "fltrust":
                    di = D[i]
                    cs = torch.relu(torch.nn.functional.cosine_similarity(innov[ids], di[None], dim=1))
                    if cs.sum() > 0:
                        sc = (di.norm() / (innov[ids].norm(dim=1) + 1e-9))[:, None]
                        newV[i] = V[i] + cfg["eta_c"] * (cs[:, None] * sc * innov[ids]).sum(0) / cs.sum()
                    w = (cs / cs.sum().clamp(min=1e-9)).numpy() * cfg["eta_c"]
                elif rule == "balance":
                    thr = cfg["bal_gamma"] * np.exp(-cfg["bal_kappa"] * r / n_rounds) * V[i].norm()
                    dist = (theta[ids] - V[i]).norm(dim=1)
                    acc = dist <= thr
                    w = np.zeros(len(ids))
                    if acc.any():
                        newV[i] = (1 - cfg["eta_c"]) * V[i] + cfg["eta_c"] * theta[ids][acc].mean(0)
                        w[acc.numpy()] = cfg["eta_c"] / acc.sum().item()
                else:
                    raise ValueError(rule)
                byz_mass.append(float(w[self.is_byz[ids]].sum()))
        for i in receivers:
            V[i] = newV[i]
        set_vec(fleet.SH, V)
        hon = np.array(receivers) if receivers else np.arange(S)
        cons = float((V[hon] - V[hon].mean(0)).norm(dim=1).mean())
        msgs = sum(len(x) for x in inbox) if rule not in ("local", "fedavg") else (0 if rule == "local" else 2 * self.I)
        self.log.append(dict(r=r, byz_mass=float(np.mean(byz_mass)) if byz_mass else 0.0,
                             consensus=cons, msgs=msgs, dim=int(V.shape[1])))

    def _trust(self, V, theta, innov, inbox, receivers, fleet):
        cfg, S = self.cfg, self.S
        newV = V.clone()
        pairs = [(i, j) for i in receivers for j in inbox[i]]
        byz_mass = []
        if not pairs:
            return newV, [0.0] * len(receivers)
        # first-hand evidence: validation gain of the half-step blend on the receiver's own reservoir
        cand = torch.stack([V[i] + 0.5 * (theta[j] - V[i]) for i, j in pairs])
        own = torch.stack([V[i] for i in receivers])
        allv = torch.cat([own, cand])
        owner = [i for i in receivers] + [i for i, _ in pairs]
        SHc = unflatvec(allv, fleet.SH)
        loss = fleet.eval_loss(SHc, owner).numpy()
        base = dict(zip(receivers, loss[:len(receivers)]))
        gains = {}
        for (i, j), l in zip(pairs, loss[len(receivers):]):
            gains[(i, j)] = (base[i] - l) / max(base[i], 1e-6)
        lam = cfg["forget"]
        a0, b0 = cfg["trust_prior"]
        for (i, j), g in gains.items():
            pos = 1.0 if g >= -cfg["xi"] else 0.0
            self.alpha[i, j] = lam * self.alpha[i, j] + (1 - lam) * a0 + pos
            self.beta[i, j] = lam * self.beta[i, j] + (1 - lam) * b0 + (1.0 - pos) * cfg["neg_w"]
            self.nfirst[i, j] += 1
        # second-hand opinions from trusted reporters
        if cfg.get("second_hand", True):
            tau = self.alpha / (self.alpha + self.beta)
            for i, j in pairs:
                if tau[i, j] < cfg["tau_rep"]:
                    continue
                if self.is_byz[j]:  # lying reporter: praises Byzantine ids, slanders honest ones
                    op = np.where(self.is_byz, 1.0, 0.0)
                    known = np.ones(self.I, bool)
                else:
                    op = self.alpha[j] / (self.alpha[j] + self.beta[j]) if j < S else np.full(self.I, .5)
                    known = self.nfirst[j] > 0 if j < S else np.zeros(self.I, bool)
                known[i] = False
                g2 = cfg["gamma2"] * tau[i, j]
                self.alpha[i, known] += g2 * op[known]
                self.beta[i, known] += g2 * (1 - op[known])
        tau = self.alpha / (self.alpha + self.beta)
        # precision of each sender's prior ensemble on the public probe set
        if cfg.get("precision", True):
            senders = sorted({j for _, j in pairs} | set(receivers))
            u = self.disagreement(unflatvec(theta[senders], fleet.SH)).numpy()
            prec = dict(zip(senders, 1.0 / u))
        for i in receivers:
            ids = inbox[i]
            if not ids:
                byz_mass.append(0.0)
                continue
            s = np.array([tau[i, j] if (tau[i, j] >= cfg["tau_min"] and self.nfirst[i, j] >= cfg["n_min"]) else 0.0
                          for j in ids])
            if cfg.get("precision", True):
                s = s * np.clip([prec[j] / prec[i] for j in ids], 0.5, 2.0)
            if cfg.get("sybil_group", True) and len(ids) > 1:
                X = torch.nn.functional.normalize(innov[ids], dim=1)
                sim = (X @ X.T).numpy()
                grp = np.maximum(1, (np.nan_to_num(sim, nan=0.0) > cfg["kappa_s"]).sum(1))
                s = s / grp
            if s.sum() <= 0:
                byz_mass.append(0.0)
                continue
            w = cfg["eta_c"] * s / s.sum()
            newV[i] = (1 - cfg["eta_c"]) * V[i] + (torch.as_tensor(w, dtype=torch.float32)[:, None] * theta[ids]).sum(0)
            byz_mass.append(float(w[self.is_byz[ids]].sum()))
        return newV, byz_mass
