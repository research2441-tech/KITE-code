"""Neuro-symbolic policy layer and counterfactual imagination planner.

Predicates are grounded on the world-model beliefs; a small rule base proposes
and scores defensive actions; a shield enforces hard security constraints; the
world-model prior ensemble evaluates each surviving candidate through imagined
multi-step attack-defence trajectories and the counterfactual no-op roll-out.
"""
import numpy as np
import torch
from cyberrange import N_HOSTS, N_ACT, N_KIND, W_AVAIL, VALUE, C_COMP, C_AVAIL, C_ACT
import wm

RULES = ["R1 contain-active", "R2 eradicate-persistence", "R3 contain-near-asset",
         "R4 eradicate-low-value", "R5 decoy-on-threat-path", "R6 decoy-on-recon",
         "R7 analyse-uncertain"]
KIND_OF_RULE = [1, 2, 1, 2, 3, 3, 0]
CRIT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], bool)  # gateway and critical controller


def hop_dist(adj):
    S, N = adj.shape[0], adj.shape[1]
    D = np.full((S, N), 99)
    for s in range(S):
        D[s, 9] = 0
        frontier = [9]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for f in frontier:
                for h in np.where(adj[s, f])[0]:
                    if D[s, h] > d:
                        D[s, h] = d
                        nxt.append(h)
            frontier = nxt
    return D


def symbolic(bel, act, status, adj, near_crit, cfg):
    """bel (S,N,4) prob, act (S,N,6) prob, status (S,N,3) 0/1, adj (S,N,N).

    Returns rule-support R (S,N_ACT), shield mask (S,N_ACT), rule id per action,
    and the predicate table used for audit.
    """
    S = bel.shape[0]
    foot = bel[..., 2] + bel[..., 3]
    pers, scan = bel[..., 3], bel[..., 1]
    rec, lat, exf = act[..., 1], act[..., 4], act[..., 5]
    iso, down, dec = status[..., 0], status[..., 1], status[..., 2]
    up = (1 - iso) * (1 - down)
    nbr = adj * up[:, None, :]
    threat = 1 - np.exp((nbr * np.log1p(-np.clip(foot, 0, .999))[:, None, :]).sum(-1))
    high = (VALUE >= 1.0)[None].astype(float)
    crit = CRIT[None].astype(float)
    T = np.zeros((S, N_HOSTS, len(RULES)))
    T[..., 0] = np.minimum(foot, np.maximum(exf, lat)) * up
    T[..., 1] = pers * (1 - crit)
    T[..., 2] = foot * near_crit * up
    T[..., 3] = foot * (1 - near_crit) * (1 - crit)
    T[..., 4] = threat * (1 - foot) * high * (1 - dec)
    T[..., 5] = np.maximum(scan, rec) * (1 - foot) * (1 - dec)
    T[..., 6] = 4 * foot * (1 - foot)
    R = np.zeros((S, N_ACT))
    rid = np.full((S, N_ACT), -1)
    for r, k in enumerate(KIND_OF_RULE):
        idx = 1 + np.arange(N_HOSTS) * N_KIND + k
        better = T[..., r] > R[:, idx]
        R[:, idx] = np.where(better, T[..., r], R[:, idx])
        rid[:, idx] = np.where(better, r, rid[:, idx])
    # shield
    M = np.ones((S, N_ACT), bool)
    hosts = np.arange(N_HOSTS)
    contained = (iso + down).sum(1)
    for k in (1, 2):
        idx = 1 + hosts * N_KIND + k
        need = crit * (foot < cfg["tau_c"])          # S1 critical-asset availability
        M[:, idx] &= ~need.astype(bool)
        M[:, idx] &= (contained < cfg["k_max"])[:, None]  # S3 containment budget
    M[:, 1 + hosts * N_KIND + 1] &= ~(iso + down).astype(bool)  # S2 redundant
    M[:, 1 + hosts * N_KIND + 2] &= ~down.astype(bool)
    M[:, 1 + hosts * N_KIND + 3] &= ~dec.astype(bool)
    if not cfg.get("shield", True):
        M[:] = True
    return R, M, rid, dict(foot=foot, pers=pers, exf=exf, lat=lat, threat=threat)


def status_rollout(timers, cand, H):
    """timers (S,N,3) int (iso, down, decoy) BEFORE the candidate is applied.
    cand (S,M) actions. Returns status (S,M,H,N,3) and aoh (S,M,H,N,4)."""
    S, M = cand.shape
    t = np.repeat(timers[:, None], M, 1).astype(int)  # S,M,N,3
    h = np.clip((cand - 1) // N_KIND, 0, None)
    k = (cand - 1) % N_KIND
    s_i, m_i = np.nonzero(cand > 0)
    hh, kk = h[s_i, m_i], k[s_i, m_i]
    sel = kk == 1
    t[s_i[sel], m_i[sel], hh[sel], 0] = 6
    sel = kk == 2
    t[s_i[sel], m_i[sel], hh[sel], 1] = 2
    t[s_i[sel], m_i[sel], hh[sel], 0] = 0
    sel = kk == 3
    t[s_i[sel], m_i[sel], hh[sel], 2] = 10
    st = np.zeros((S, M, H, N_HOSTS, 3), np.float32)
    avail = np.zeros((S, M, H))
    for j in range(H):
        avail[:, :, j] = (((t[..., 0] > 0) | (t[..., 1] > 0)) * W_AVAIL).sum(-1)
        t = np.maximum(t - 1, 0)
        st[:, :, j] = (t > 0)
    aoh = np.zeros((S, M, H, N_HOSTS, 4), np.float32)
    aoh[s_i, m_i, 0, hh, kk] = 1.0
    acost = np.where(cand > 0, C_ACT[np.clip(k, 0, 3)], 0.0)
    return st, aoh, avail, acost


def imagine(shared, z, st, aoh, adj, H, gamma):
    """shared params (S,...); z (S,N,D); st/aoh (S,M,H,N,*). Returns the
    discounted expected compromise and exfiltration cost (S,M,K)."""
    w_av = torch.as_tensor(W_AVAIL, dtype=torch.float32)
    val = torch.as_tensor(VALUE, dtype=torch.float32)
    st, aoh = torch.as_tensor(st), torch.as_tensor(aoh)
    A = torch.as_tensor(adj, dtype=torch.float32)
    M = st.shape[1]
    out = []
    with torch.no_grad():
        for k in range(wm.K_ENS):
            zc = z[:, None].expand(-1, M, -1, -1)
            J = torch.zeros(z.shape[0], M)
            for j in range(H):
                zc = wm.prior_step(shared, k, zc, aoh[:, :, j], st[:, :, j], A)
                lb, la = wm.heads(shared, zc)
                pb, pa = lb.exp(), la.exp()
                c = C_COMP * ((pb[..., 2] + pb[..., 3]) * w_av).sum(-1) + 0.55 * (pa[..., 5] * val).sum(-1)
                J = J + gamma ** j * c
            out.append(J)
    return torch.stack(out, -1)


class Planner:
    def __init__(self, adj, cfg):
        self.adj = adj.astype(np.float32)
        self.cfg = cfg
        self.near = (hop_dist(adj) <= cfg["near_hops"]).astype(float)
        self.trace = []

    def act(self, shared, z, timers, rng, explore=0.0, record=False):
        cfg, S = self.cfg, z.shape[0]
        with torch.no_grad():
            lb, la = wm.heads(shared, z)
        bel, act = lb.exp().numpy(), la.exp().numpy()
        status = (timers > 0).astype(float)
        R, M, rid, pred = symbolic(bel, act, status, self.adj, self.near, cfg)
        if not cfg.get("rules", True):
            R = np.zeros_like(R)
        Mc = cfg["n_cand"]
        score = np.where(M, R, -1.0)
        if cfg.get("propose", "rules") == "all":
            cand = np.tile(np.arange(N_ACT), (S, 1))
            valid = M.copy()
        else:
            order = np.argsort(-score, 1)[:, :Mc]
            cand = np.concatenate([np.zeros((S, 1), int), order], 1)
            valid = np.concatenate([np.ones((S, 1), bool), np.take_along_axis(M, order, 1)], 1)
        if cfg.get("imagine", True):
            st, aoh, avail, acost = status_rollout(timers, cand, cfg["H"])
            J = imagine(shared, z, st, aoh, self.adj, cfg["H"], cfg["gamma"]).numpy()
            disc = cfg["gamma"] ** np.arange(cfg["H"])
            J = J + (C_AVAIL * avail * disc).sum(-1)[..., None] + acost[..., None]
            mu, sd = J.mean(-1), J.std(-1)
            U = -(mu + cfg["beta"] * sd) + cfg["lam"] * np.take_along_axis(R, cand, 1)
            cf_gain = mu[:, :1] - mu  # counterfactual cost reduction versus no-op
        else:
            U = np.take_along_axis(R, cand, 1) - cfg["r_min"]
            U[:, 0] = 0.0
            cf_gain = np.zeros_like(U)
        U = np.where(valid, U, -1e9)
        pick = U.argmax(1)
        a = cand[np.arange(S), pick]
        if explore > 0:
            m = rng.random(S) < explore
            if m.any():
                ok = [np.where(M[s])[0] for s in range(S)]
                a[m] = [rng.choice(ok[s]) if rng.random() < .7 else 0 for s in np.where(m)[0]]
        if record:
            for s in range(S):
                aa = int(a[s])
                self.trace.append(dict(site=s, action=aa,
                                       rule=RULES[rid[s, aa]] if aa > 0 and rid[s, aa] >= 0 else "none",
                                       truth=float(R[s, aa]) if aa > 0 else 0.0,
                                       cf_gain=float(cf_gain[s, pick[s]]),
                                       foot=pred["foot"][s].round(3).tolist()))
        return a, dict(bel=bel, act=act, R=R, M=M)

