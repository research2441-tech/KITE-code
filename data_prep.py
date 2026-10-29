"""Prepare telemetry pools for the edge cyber range.

Each public corpus is mapped onto kill-chain emission categories:
  0 benign, 1 reconnaissance, 2 initial access / privilege escalation,
  3 persistence / command-and-control, 4 lateral movement, 5 exfiltration / impact.
Fine attack labels marked as zero-day are withheld from every training pool and
only emitted by the zero-day campaign profile during evaluation.

Rows are split by the corpus 'order' column (early 60 % -> exercise pool,
late 40 % -> evaluation pool) so that no flow record is shared between the
range exercises used for learning and the evaluation episodes.
Features: log1p on non-negative columns, robust scaling fitted on the exercise
pool, PCA to D_FLOW dimensions fitted on the exercise pool of that corpus.
"""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

D_FLOW = 12
CATS = ["benign", "recon", "access", "persist", "lateral", "impact"]

# fine label -> (category, zero_day flag)
MAPS = {
    "ToN-IoT": {
        "normal": ("benign", 0), "scanning": ("recon", 0),
        "password": ("access", 0), "injection": ("access", 0), "xss": ("access", 1),
        "backdoor": ("persist", 0), "mitm": ("lateral", 0),
        "ddos": ("impact", 0), "dos": ("impact", 1),
    },
    "CICIoT2023": {
        "BENIGN": ("benign", 0),
        "RECON-HOSTDISCOVERY": ("recon", 0), "RECON-OSSCAN": ("recon", 0),
        "RECON-PORTSCAN": ("recon", 0), "RECON-PINGSWEEP": ("recon", 0),
        "VULNERABILITYSCAN": ("recon", 0),
        "DICTIONARYBRUTEFORCE": ("access", 0), "SQLINJECTION": ("access", 0),
        "XSS": ("access", 0), "COMMANDINJECTION": ("access", 1),
        "BROWSERHIJACKING": ("access", 1),
        "MIRAI-GREETH_FLOOD": ("persist", 0), "MIRAI-UDPPLAIN": ("persist", 0),
        "MIRAI-GREIP_FLOOD": ("persist", 0),
        "BACKDOOR_MALWARE": ("persist", 1), "UPLOADING_ATTACK": ("persist", 1),
        "MITM-ARPSPOOFING": ("lateral", 0), "DNS_SPOOFING": ("lateral", 0),
    },
    "CIC-DIAD2024": {
        "BenignTraffic": ("benign", 0), "VulnerabilityScan": ("recon", 0),
        "DictionaryBruteForce": ("access", 0), "SqlInjection": ("access", 0),
        "XSS": ("access", 1),
        "Mirai-greeth_flood": ("persist", 0), "Uploading_Attack": ("persist", 1),
        "MITM-ArpSpoofing": ("lateral", 0), "DNS_Spoofing": ("lateral", 0),
    },
}


def impact_rule(name, fine):
    """Flood families stand in for the actions-on-objective stage."""
    f = fine.upper()
    if name == "CICIoT2023":
        if f.startswith("DDOS"):
            return ("impact", 0)
        if f.startswith("DOS"):
            return ("impact", 1)
    if name == "CIC-DIAD2024":
        if f.startswith("DDOS"):
            return ("impact", 0)
        if f.startswith("DOS"):
            return ("impact", 1)
    return None


def prepare(root, out):
    pools = {}
    meta = {}
    for name in ["ToN-IoT", "CICIoT2023", "CIC-DIAD2024"]:
        d = pd.read_parquet(f"{root}/{name}.parquet")
        fine = d["fine"].astype(str).values
        cat = np.full(len(d), -1)
        zd = np.zeros(len(d), dtype=int)
        for i, f in enumerate(fine):
            m = MAPS[name].get(f) or impact_rule(name, f)
            if m is not None:
                cat[i] = CATS.index(m[0])
                zd[i] = m[1]
        keep = cat >= 0
        feats = d.drop(columns=["family", "fine", "order"]).values.astype(np.float64)
        order = d["order"].values
        feats, cat, zd, order, fine = feats[keep], cat[keep], zd[keep], order[keep], fine[keep]
        # early/late split per category so every category has both pools
        split = np.zeros(len(cat), dtype=int)
        for c in range(len(CATS)):
            for z in (0, 1):
                idx = np.where((cat == c) & (zd == z))[0]
                if len(idx) == 0:
                    continue
                idx = idx[np.argsort(order[idx])]
                split[idx[int(0.6 * len(idx)):]] = 1
        nonneg = (feats.min(axis=0) >= 0)
        feats[:, nonneg] = np.log1p(feats[:, nonneg])
        tr = (split == 0) & (zd == 0)
        med = np.median(feats[tr], axis=0)
        iqr = np.subtract(*np.percentile(feats[tr], [75, 25], axis=0))
        iqr[iqr < 1e-6] = feats[tr].std(axis=0)[iqr < 1e-6] + 1e-6
        x = np.clip((feats - med) / iqr, -20, 20)
        pca = PCA(n_components=D_FLOW, random_state=0).fit(x[tr])
        z = pca.transform(x).astype(np.float32)
        z /= z[tr].std(axis=0, keepdims=True)
        pools[name] = dict(x=z, cat=cat.astype(np.int8), zd=zd.astype(np.int8),
                           split=split.astype(np.int8))
        counts = {}
        for s in (0, 1):
            for c in range(len(CATS)):
                for zz in (0, 1):
                    n = int(((split == s) & (cat == c) & (zd == zz)).sum())
                    if n:
                        counts[f"{['exercise','eval'][s]}/{CATS[c]}/{'zero-day' if zz else 'known'}"] = n
        meta[name] = dict(n_raw=int(len(d)), n_used=int(keep.sum()),
                          n_features=int(feats.shape[1]),
                          pca_var=float(pca.explained_variance_ratio_.sum()),
                          counts=counts,
                          zero_day_labels=sorted({f for f, z_ in zip(fine, zd) if z_}))
    np.savez_compressed(out + "/pools.npz",
                        **{f"{n}__{k}": v for n, p in pools.items() for k, v in p.items()})
    json.dump(meta, open(out + "/pools_meta.json", "w"), indent=1)
    print(json.dumps({n: {k: v for k, v in m.items() if k != 'counts'} for n, m in meta.items()}, indent=1))


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2])
