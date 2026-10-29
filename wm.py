"""Graph-conditioned latent cyber world model, batched over parameter instances.

Convention: every parameter tensor carries a leading instance axis P (one
instance per site, or per example when per-example gradients are needed for
DP-SGD), and every data tensor carries the same leading axis.

Shared core (exchanged between sites): posterior GRU, message map, K-member
prior ensemble, stage-belief head, activity head.
Local part (never exchanged): telemetry encoder and decoder, because each site
replays a different corpus with its own feature space.
"""
import math
import torch
import torch.nn.functional as F
from cyberrange import N_HOSTS, OBS_DIM, D_FLOW, ROLE

D_Z = 32
K_ENS = 3
N_LVL, N_EMIT = 4, 6  # levels; emission classes (none, recon, access, persist, lateral, impact)
ROLE_OH = F.one_hot(torch.as_tensor(ROLE), 5).float()


def _lin(gen, i, o, scale=1.0):
    return {"W": torch.randn(i, o, generator=gen) * (scale / math.sqrt(i)), "b": torch.zeros(o)}


def init_params(seed):
    g = torch.Generator().manual_seed(seed)
    shared = {
        "gru_x": _lin(g, D_Z + D_Z + 4 + 3, 3 * D_Z),
        "gru_h": _lin(g, D_Z, 3 * D_Z),
        "msg": _lin(g, D_Z, D_Z),
        "bel": _lin(g, D_Z, N_LVL),
        "act": _lin(g, D_Z, N_EMIT),
    }
    for k in range(K_ENS):
        shared[f"pr{k}_1"] = _lin(g, D_Z + D_Z + 4 + 3 + 5, 96)
        shared[f"pr{k}_2"] = _lin(g, 96, D_Z, 0.1)
    local = {
        "enc1": _lin(g, OBS_DIM + 5, 64),
        "enc2": _lin(g, 64, D_Z),
        "dec1": _lin(g, D_Z, 64),
        "dec2": _lin(g, 64, 2 * D_FLOW),
    }
    return flat(shared), flat(local)


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def L(P, name, x):
    """Instance-batched affine map. x: (P, ..., in)."""
    W, b = P[name + ".W"], P[name + ".b"]
    n = W.shape[0]
    y = torch.bmm(x.reshape(n, -1, x.shape[-1]), W).reshape(x.shape[:-1] + (W.shape[-1],))
    return y + b.view(n, *[1] * (x.dim() - 2), -1)


def _adj(adj, up):
    return adj.view(adj.shape[0], *[1] * (up.dim() - 2), N_HOSTS, N_HOSTS)


def messages(P, z, adj, up):
    """Mean of neighbour latents over links whose endpoints are both up."""
    m = _adj(adj, up) * up[..., :, None] * up[..., None, :]
    deg = m.sum(-1, keepdim=True).clamp(min=1.0)
    return torch.tanh(L(P, "msg", (m @ z) / deg))


def roles(like):
    return ROLE_OH.expand(like.shape[:-1] + (5,))


def encode(P, obs):
    return L(P, "enc2", F.elu(L(P, "enc1", torch.cat([obs, roles(obs)], -1))))


def decode(P, z):
    return L(P, "dec2", F.elu(L(P, "dec1", z)))


def _up(status):
    return (1.0 - status[..., 0]) * (1.0 - status[..., 1])


def posterior_step(P, z, obs, act_oh, status, adj, enc=None):
    e = encode(P, obs) if enc is None else enc
    x = torch.cat([e, messages(P, z, adj, _up(status)), act_oh, status], -1)
    gx, gh = L(P, "gru_x", x), L(P, "gru_h", z)
    r = torch.sigmoid(gx[..., :D_Z] + gh[..., :D_Z])
    u = torch.sigmoid(gx[..., D_Z:2 * D_Z] + gh[..., D_Z:2 * D_Z])
    n = torch.tanh(gx[..., 2 * D_Z:] + r * gh[..., 2 * D_Z:])
    return (1 - u) * n + u * z


def prior_step(P, k, z, act_oh, status, adj):
    x = torch.cat([z, messages(P, z, adj, _up(status)), act_oh, status, roles(z)], -1)
    return z + L(P, f"pr{k}_2", F.elu(L(P, f"pr{k}_1", x)))


def heads(P, z):
    return F.log_softmax(L(P, "bel", z), -1), F.log_softmax(L(P, "act", z), -1)


def act_onehot(a, n=N_HOSTS):
    a = torch.as_tensor(a)
    h = torch.clamp((a - 1) // 4, min=0)
    k = (a - 1) % 4
    oh = torch.zeros(a.shape + (n, 4))
    valid = a > 0
    oh[valid, h[valid], k[valid]] = 1.0
    return oh


def _red(x):
    """Sum over every axis except the instance axis."""
    return x.reshape(x.shape[0], -1).sum(1)


def kill_chain_violation(b_now, b_next, a_next, st_next, adj):
    """Soft (Lukasiewicz) violation of kill-chain precedence rules, per instance.

    R1 a new foothold needs a scanned host or a compromised up-neighbour;
    R2 persistence needs a foothold one step earlier;
    R3 exfiltration/impact activity needs a foothold;
    R4 an isolated host cannot newly be reached laterally.
    """
    pb, pn = b_now.exp(), b_next.exp()
    f0 = pb[..., 2] + pb[..., 3]
    f1 = pn[..., 2] + pn[..., 3]
    up = _up(st_next)
    nb = _adj(adj, up) * up[..., None, :]
    spread = 1.0 - torch.exp((nb * torch.log1p(-f0.clamp(max=0.999))[..., None, :]).sum(-1))
    v1 = F.relu(f1 - f0 - pb[..., 1] - spread * up)
    v2 = F.relu(pn[..., 3] - f0 - 0.05)
    v3 = F.relu(a_next.exp()[..., 5] - f1 - 0.05)
    v4 = F.relu(f1 - f0 - pb[..., 1]) * st_next[..., 0]
    return _red(v1 + v2 + v3 + v4) / (4.0 * f0[0].numel())


def seq_loss(P, ch, adj, w_sym=0.5, ens_w=None, burn=3, horizon=3):
    """Per-instance loss on a batch of chunks.

    ch: obs (P,B,T,N,O), aoh (P,B,T,N,4) action preceding each observation,
    st (P,B,T,N,3) status, lvl/emit (P,B,T,N) long, lmask/emask (P,B,T,N).
    ens_w: (P,B,K) bootstrap weights for the prior ensemble.
    Returns total (P,) and parts (P,6).
    """
    Pn, B, T = ch["obs"].shape[:3]
    z = torch.zeros(Pn, B, N_HOSTS, D_Z)
    zs = []
    E = encode(P, ch["obs"])
    for t in range(T):
        z = posterior_step(P, z, None, ch["aoh"][:, :, t], ch["st"][:, :, t], adj, enc=E[:, :, t])
        zs.append(z)
    Z = torch.stack(zs, 2)  # P,B,T,N,D
    lb, la = heads(P, Z)
    lvl, emit, lm, em = ch["lvl"], ch["emit"], ch["lmask"], ch["emask"]
    sl = slice(burn, T)

    def ce(logp, y, m):
        return -_red(logp.gather(-1, y[..., None]).squeeze(-1) * m) / _red(m).clamp(min=1)

    ce_b = ce(lb[:, :, sl], lvl[:, :, sl], lm[:, :, sl])
    ce_a = ce(la[:, :, sl], emit[:, :, sl], em[:, :, sl])
    rec = _red(((decode(P, Z[:, :, sl]) - ch["obs"][:, :, sl, :, :2 * D_FLOW]) ** 2).mean(-1)) \
        / (B * (T - burn) * N_HOSTS)
    Zt = Z.detach()
    ew = ens_w if ens_w is not None else torch.ones(Pn, B, K_ENS)
    t0 = torch.arange(burn, T - 1)
    b0, _ = heads(P, Z[:, :, t0])
    dyn = pce = sym = 0.0
    n = 0
    for k in range(K_ENS):
        zp = Z[:, :, t0]
        wk = ew[:, :, k][:, :, None, None]
        for j in range(1, horizon + 1):
            m = (T - 1 - burn) - (j - 1)
            if m <= 0:
                break
            tt = t0[:m] + j
            zp = prior_step(P, k, zp[:, :, :m], ch["aoh"][:, :, tt], ch["st"][:, :, tt], adj)
            pb, pa = heads(P, zp)
            dyn = dyn + _red(wk * ((zp - Zt[:, :, tt]) ** 2).mean(-1)) / (B * N_HOSTS)
            pce = pce + m * (ce(pb, lvl[:, :, tt], lm[:, :, tt] * wk) + ce(pa, emit[:, :, tt], em[:, :, tt] * wk))
            if j == 1:
                sym = sym + kill_chain_violation(b0, pb, pa, ch["st"][:, :, tt], adj)
            n += m
    dyn, pce, sym = dyn / n, pce / n, sym / K_ENS
    total = ce_b + ce_a + 0.5 * rec + dyn + 0.5 * pce + w_sym * sym
    return total, torch.stack([ce_b, ce_a, rec, dyn, pce, sym], 1)
