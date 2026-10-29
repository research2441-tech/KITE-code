"""EdgeRange: a multi-site edge-network cyber range driven by real flow records.

Every site is a small edge network (gateway, IoT devices, two servers, an
operator workstation and one critical controller). An adaptive attacker
campaign moves through reconnaissance, access/privilege escalation,
persistence, lateral movement and exfiltration/impact. Each host emits M_FLOWS
flow records per step; malicious records are sampled from the public corpus
assigned to the site according to the kill-chain stage of the current
attacker operation, benign records otherwise.

Host compromise level: 0 clean, 1 scanned, 2 foothold (privileged), 3 persistent.
Defender action per step: 0 = no-op, 1 + 4*h + k with
k = 0 analyse, 1 isolate, 2 restore, 3 deploy decoy.
"""
import numpy as np

CATS = ["benign", "recon", "access", "persist", "lateral", "impact"]
DATASETS = ["ToN-IoT", "CICIoT2023", "CIC-DIAD2024"]
N_HOSTS = 10
M_FLOWS = 6
D_FLOW = 12
N_KIND = 4  # analyse, isolate, restore, decoy
N_ACT = 1 + N_KIND * N_HOSTS
OBS_DIM = 2 * D_FLOW + 2 + 3 + 4  # flow stats, alert, decoy alert, status, revealed level
ROLE = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 4])  # gateway, iot, server, workstation, critical
W_AVAIL = np.array([2.0, .4, .4, .4, .4, .4, 1.5, 1.5, 1.0, 3.0])
VALUE = np.array([.5, .3, .3, .3, .3, .3, 2.0, 2.0, 1.0, 5.0])
EXPOSED = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], bool)

# op order: recon, access, persist, lateral, exfil
PROFILES = {
    "smash": dict(p_op=.9, w=[1, 1, .3, 1, 2], s=[.8, .5, .4, .4, .6], rho=.5, zd=0, direct=0.),
    "apt": dict(p_op=.6, w=[.6, 1, 2, .8, .8], s=[.6, .35, .6, .3, .5], rho=1 / 3, zd=0, direct=0.),
    "worm": dict(p_op=1., w=[.5, 1, .3, 2.5, .7], s=[.9, .5, .3, .55, .5], rho=.5, zd=0, direct=0.),
    "zeroday": dict(p_op=.8, w=[.2, 1, 1.5, 1.2, 1.2], s=[.7, .6, .5, .45, .55], rho=1 / 3, zd=1, direct=.5),
}
PROFILE_NAMES = list(PROFILES)

P_DETECT, P_FALSE = 0.3, 0.03
C_COMP, C_AVAIL, C_ACT = 0.1, 0.12, np.array([0.02, 0.05, 0.1, 0.02])


class Pools:
    def __init__(self, path):
        z = np.load(path)
        xs, self.off = [], {}
        start = 0
        for d in DATASETS:
            x, cat, zd, sp = (z[f"{d}__{k}"] for k in ("x", "cat", "zd", "split"))
            for s in (0, 1):
                for c in range(len(CATS)):
                    for q in (0, 1):
                        m = (sp == s) & (cat == c) & (zd == q)
                        if m.sum() == 0:
                            continue
                        xs.append(x[m])
                        self.off[(d, s, c, q)] = (start, int(m.sum()))
                        start += int(m.sum())
        self.X = np.concatenate(xs).astype(np.float32)

    def table(self, ds_of_site, split, zero_day):
        """offset/count tables of shape (S, n_cat) for sampling."""
        S = len(ds_of_site)
        st = np.zeros((S, len(CATS)), np.int64)
        ct = np.zeros((S, len(CATS)), np.int64)
        for i, d in enumerate(ds_of_site):
            for c in range(len(CATS)):
                key = (d, split, c, 1) if (zero_day and (d, split, c, 1) in self.off) else (d, split, c, 0)
                st[i, c], ct[i, c] = self.off[key]
        return st, ct


def make_topology(rng):
    A = np.zeros((N_HOSTS, N_HOSTS), bool)
    for h in range(1, 9):
        A[0, h] = True
    for i in range(1, 6):
        for j in range(i + 1, 6):
            if rng.random() < .3:
                A[i, j] = True
    A[6, 7] = A[6, 8] = A[7, 8] = True
    A[6, 9] = A[7, 9] = True
    if rng.random() < .5:
        A[8, 9] = True
    return A | A.T


class EdgeRange:
    def __init__(self, pools, n_sites, ds_of_site, seed, split=0, horizon=100):
        self.P, self.S, self.T = pools, n_sites, horizon
        self.ds = list(ds_of_site)
        self.rng = np.random.default_rng(seed)
        topo_rng = np.random.default_rng(seed + 7919)
        self.A = np.stack([make_topology(topo_rng) for _ in range(n_sites)])
        self.split = split
        self.tab = {q: pools.table(self.ds, split, q) for q in (0, 1)}

    # ------------------------------------------------------------------
    def reset(self, profiles):
        S, N = self.S, N_HOSTS
        self.prof = [PROFILES[p] for p in profiles]
        self.prof_name = list(profiles)
        self.t = 0
        self.lvl = np.zeros((S, N), np.int64)
        self.iso = np.zeros((S, N), np.int64)
        self.down = np.zeros((S, N), np.int64)
        self.decoy = np.zeros((S, N), np.int64)
        self.harden = np.zeros((S, N), np.int64)
        self.stealth = np.zeros(S, np.int64)
        self.relearn = np.ones(S)
        self.start = self.rng.integers(0, 21, S)
        self.reveal = np.full((S, N), -1)
        self.foot_since = np.full((S, N), -1)
        self.stats = dict(ret=np.zeros(S), comp_steps=np.zeros(S), exfil=np.zeros(S),
                          loss=np.zeros(S), breach=np.zeros(S), avail=np.zeros(S),
                          false_iso=np.zeros(S), false_rst=np.zeros(S), n_iso=np.zeros(S),
                          n_rst=np.zeros(S), ttc=[[] for _ in range(S)],
                          crit_comp=np.zeros(S))
        emit = np.full((S, N), -1)
        return self._observe(emit, np.zeros((S, N), bool))

    # ------------------------------------------------------------------
    def _observe(self, emit, decoy_hit):
        S, N, rng = self.S, N_HOSTS, self.rng
        zero = np.array([self.prof[s]["zd"] for s in range(S)], bool)
        cat = np.zeros((S, N, M_FLOWS), np.int64)
        rho = np.array([self.prof[s]["rho"] * (0.5 if self.stealth[s] > 0 else 1.0) for s in range(S)])
        k = np.maximum(1, np.round(rho * M_FLOWS)).astype(int)
        for s in range(S):
            for h in np.where(emit[s] >= 0)[0]:
                cat[s, h, :k[s]] = emit[s, h]
        st0, ct0 = self.tab[0]
        st1, ct1 = self.tab[1]
        st = np.where(zero[:, None], st1, st0)
        ct = np.where(zero[:, None], ct1, ct0)
        sidx = np.arange(S)[:, None, None]
        u = rng.random((S, N, M_FLOWS))
        rows = st[sidx, cat] + (u * ct[sidx, cat]).astype(np.int64)
        flows = self.P.X[rows]  # S,N,M,D
        mean, std = flows.mean(2), flows.std(2)
        known = (emit > 0) & ~zero[:, None]
        alert = (known & (rng.random((S, N)) < P_DETECT)) | (rng.random((S, N)) < P_FALSE)
        status = np.stack([self.iso > 0, self.down > 0, self.decoy > 0], -1)
        rev = np.zeros((S, N, 4))
        m = self.reveal >= 0
        rev[m, self.reveal[m]] = 1
        self.reveal[:] = -1
        obs = np.concatenate([mean, std, alert[..., None], decoy_hit[..., None], status, rev], -1)
        self.last_emit = emit
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    def _defend(self, actions):
        S, st = self.S, self.stats
        cost = np.zeros(S)
        lost = np.zeros((S, N_HOSTS), bool)  # attacker lost a host this step
        for s in range(S):
            a = int(actions[s])
            if a == 0:
                continue
            h, k = (a - 1) // N_KIND, (a - 1) % N_KIND
            cost[s] += C_ACT[k]
            if k == 0:
                if self.rng.random() < .9:
                    self.reveal[s, h] = self.lvl[s, h]
            elif k == 1:
                self.iso[s, h] = 6
                st["n_iso"][s] += 1
                if self.lvl[s, h] < 2:
                    st["false_iso"][s] += 1
                else:
                    lost[s, h] = True
                    self._contain(s, h)
            elif k == 2:
                st["n_rst"][s] += 1
                if self.lvl[s, h] < 2:
                    st["false_rst"][s] += 1
                self.down[s, h] = 2
                self.iso[s, h] = 0
                self.harden[s, h] = 15
                if self.lvl[s, h] == 3 and self.rng.random() < .5:
                    pass  # persistence survives the re-image
                else:
                    if self.lvl[s, h] >= 2:
                        lost[s, h] = True
                        self._contain(s, h)
                    self.lvl[s, h] = 0
            else:
                self.decoy[s, h] = 10
        return cost, lost

    def _contain(self, s, h):
        if self.foot_since[s, h] >= 0:
            self.stats["ttc"][s].append(self.t - self.foot_since[s, h])
            self.foot_since[s, h] = -1

    # ------------------------------------------------------------------
    def _attack(self, lost):
        S, N, rng = self.S, N_HOSTS, self.rng
        emit = np.full((S, N), -1)
        hit = np.zeros((S, N), bool)
        dmg = np.zeros(S)
        for s in range(S):
            if self.t < self.start[s]:
                continue
            P = self.prof[s]
            if lost[s].any():
                self.stealth[s] = 15
                self.relearn[s] = min(self.relearn[s] * 1.15, 1.6)
            stealthy = self.stealth[s] > 0
            # passive command-and-control beacons from persistent hosts
            pers = np.where((self.lvl[s] == 3) & (self.down[s] == 0))[0]
            for h in pers:
                if rng.random() < (.15 if stealthy else .3):
                    emit[s, h] = 3
            if rng.random() > P["p_op"] * (.5 if stealthy else 1.):
                continue
            up = (self.down[s] == 0) & (self.iso[s] == 0)
            L, A = self.lvl[s], self.A[s]
            foot = np.where((L >= 2) & up)[0]
            cands = [[], [], [], [], []]
            if len(foot) == 0 and not (L >= 2).any():
                cands[0] = list(np.where(EXPOSED & (L == 0) & up)[0])
            else:
                nb = np.where(A[foot].any(0) & (L == 0) & up)[0] if len(foot) else []
                cands[0] = list(nb)
            cands[1] = list(np.where((L == 1) & up)[0])
            if P["direct"] > 0:
                cands[1] += [h for h in np.where(EXPOSED & (L == 0) & up)[0] if len(foot) == 0]
            cands[2] = list(np.where((L == 2) & up)[0])
            lat = []
            for f in foot:
                for t_ in np.where(A[f] & (L < 2) & up)[0]:
                    lat.append((f, t_))
            cands[3] = lat
            gw_ok = self.iso[s, 0] == 0 and self.down[s, 0] == 0
            cands[4] = [h for h in foot if (h == 0 or gw_ok) and VALUE[h] >= 1.0] or \
                       [h for h in foot if (h == 0 or gw_ok) and rng.random() < .2]
            w = np.array([P["w"][i] * (len(cands[i]) > 0) for i in range(5)], float)
            if stealthy:
                w[2] *= 2.0
            if w.sum() == 0:
                continue
            op = rng.choice(5, p=w / w.sum())
            succ = P["s"][op]
            if op == 0:
                h = cands[0][rng.integers(len(cands[0]))]
                emit[s, h] = 1
                if self.decoy[s, h] > 0 and rng.random() < .7:
                    hit[s, h] = True
                    self.stealth[s] = 15
                elif rng.random() < succ:
                    L[h] = 1
            elif op == 1:
                h = cands[1][rng.integers(len(cands[1]))]
                emit[s, h] = 2
                p = succ * self.relearn[s] * (.5 if self.harden[s, h] > 0 else 1.)
                if self.decoy[s, h] > 0 and rng.random() < .7:
                    hit[s, h] = True
                    self.stealth[s] = 15
                elif rng.random() < min(p, .95):
                    L[h] = 2
                    self.foot_since[s, h] = self.t
            elif op == 2:
                h = cands[2][rng.integers(len(cands[2]))]
                emit[s, h] = 3
                if rng.random() < succ:
                    L[h] = 3
            elif op == 3:
                vals = np.array([VALUE[t_] for _, t_ in cands[3]])
                pr = np.exp(vals) / np.exp(vals).sum()
                f, h = cands[3][rng.choice(len(cands[3]), p=pr)]
                emit[s, f] = 4
                if self.decoy[s, h] > 0 and rng.random() < .7:
                    hit[s, h] = hit[s, f] = True
                    self.stealth[s] = 15
                elif rng.random() < succ:
                    L[h] = 2
                    emit[s, h] = 4
                    self.foot_since[s, h] = self.t
            else:
                h = cands[4][rng.integers(len(cands[4]))]
                emit[s, h] = 5
                if rng.random() < succ:
                    dmg[s] += VALUE[h]
                    self.stats["exfil"][s] += 1
                    if h == 9:
                        self.stats["breach"][s] = 1
        return emit, hit, dmg

    # ------------------------------------------------------------------
    def step(self, actions):
        st = self.stats
        cost, lost = self._defend(actions)
        emit, hit, dmg = self._attack(lost)
        comp = (self.lvl >= 2)
        avail = ((self.iso > 0) | (self.down > 0)) * W_AVAIL
        r = -(C_COMP * (comp * W_AVAIL).sum(1) + dmg + C_AVAIL * avail.sum(1) + cost)
        st["ret"] += r
        st["comp_steps"] += comp.sum(1)
        st["crit_comp"] += comp[:, 9]
        st["loss"] += dmg
        st["avail"] += avail.sum(1)
        info = dict(lvl=self.lvl.copy(), emit=emit.copy(), dmg=dmg, comp=comp.copy())
        for arr in (self.iso, self.down, self.decoy, self.harden):
            np.maximum(arr - 1, 0, out=arr)
        self.stealth = np.maximum(self.stealth - 1, 0)
        self.t += 1
        obs = self._observe(emit, hit)
        return obs, r.astype(np.float32), self.t >= self.T, info

    def summary(self):
        st = self.stats
        out = {k: st[k].copy() for k in st if k != "ttc"}
        out["ttc"] = np.array([np.mean(x) if x else np.nan for x in st["ttc"]])
        # footholds never contained count as contained at episode end
        open_ = [(self.T - self.foot_since[s][self.foot_since[s] >= 0]).tolist() for s in range(self.S)]
        out["ttc_all"] = np.array([np.mean(st["ttc"][s] + open_[s]) if (st["ttc"][s] + open_[s]) else np.nan
                                   for s in range(self.S)])
        return out
