"""Aggregate result JSONs into paper_numbers.json.

Unit of replication: the (seed, campaign profile) block. Within a block, a
metric is averaged over honest sites and evaluation episodes. Paired tests
compare blocks with the same seed and profile (two-sided Wilcoxon signed-rank,
Holm correction across a family, matched-pairs rank-biserial effect size);
95 % confidence intervals come from 10,000 bootstrap resamples of blocks.
Every number quoted in the manuscript is read from the output file.
"""
import glob
import json
import os
import sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("KITE_RESULTS", os.environ.get("KITE_RESULTS", os.path.join(HERE, "results")))
PROFS = ["smash", "apt", "worm", "zeroday"]
_cache = {}


def load(tag):
    if tag not in _cache:
        fs = sorted(glob.glob(os.path.join(R, f"{tag}__s*.json")))
        out = {}
        for f in fs:
            seed = int(f.rsplit("__s", 1)[1][:-5])
            out[seed] = json.load(open(f))
        _cache[tag] = out
    return _cache[tag]


def episode_metric(ep, key, sites=None):
    """Per-episode, per-site metric array for one profile's episode record."""
    if key == "J":
        v = -np.array(ep["ret"], float)
    elif key == "fcr":
        fa = np.array(ep["false_iso"], float) + np.array(ep["false_rst"], float)
        n = np.array(ep["n_iso"], float) + np.array(ep["n_rst"], float)
        v = np.where(n > 0, fa / np.maximum(n, 1), np.nan)
    elif key == "ttc":
        v = np.array(ep["ttc_all"], float)
    else:
        v = np.array(ep[key], float)
    if sites is not None:
        v = v[sites]
    return v


def block(d, prof, key, sites=None):
    eps = d["eval"][prof]["episodes"]
    return float(np.nanmean(np.concatenate([episode_metric(e, key, sites) for e in eps])))


def blocks(tag, key, sites=None, seeds=None):
    """dict (seed, prof) -> value"""
    out = {}
    for s, d in load(tag).items():
        if seeds is not None and s not in seeds:
            continue
        for p in PROFS:
            if p in d["eval"]:
                out[(s, p)] = block(d, p, key, sites)
    return out


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summary(tag, key, sites=None, seeds=None):
    b = blocks(tag, key, sites, seeds)
    if not b:
        return None
    v = np.array(list(b.values()))
    per_prof = {}
    for p in PROFS:
        pv = [b[k] for k in b if k[1] == p]
        if pv:
            per_prof[p] = dict(mean=float(np.nanmean(pv)), sd=float(np.nanstd(pv, ddof=1)) if len(pv) > 1 else 0.0,
                               n=len(pv))
    return dict(mean=float(np.nanmean(v)), ci=boot_ci(v), n_blocks=len(v),
                n_seeds=len({k[0] for k in b}), per_prof=per_prof)


def paired(tag_a, tag_b, key="J", sites=None, seeds=None):
    a, b = blocks(tag_a, key, sites, seeds), blocks(tag_b, key, sites, seeds)
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return None
    x = np.array([a[k] for k in ks])
    y = np.array([b[k] for k in ks])
    d = x - y
    ok = ~np.isnan(d)
    d = d[ok]
    try:
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    r = stats.rankdata(np.abs(d[d != 0]))
    rpos, rneg = r[d[d != 0] > 0].sum(), r[d[d != 0] < 0].sum()
    rrb = float((rpos - rneg) / max(rpos + rneg, 1e-9))
    return dict(mean_a=float(x[ok].mean()), mean_b=float(y[ok].mean()), delta=float(d.mean()),
                delta_ci=boot_ci(d), rel=float(d.mean() / abs(y[ok].mean())), p=p, rrb=rrb, n=int(ok.sum()),
                wins=int((d < 0).sum()))


def holm(pv):
    items = sorted(pv.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


METRICS = ["J", "comp_steps", "exfil", "breach", "avail", "fcr", "ttc", "crit_comp", "n_iso", "n_rst"]
MAIN = ["proposed", "wm_uniform", "wm_fedavg", "wm_gossip", "wm_local", "playbook", "dqn_local", "dqn_fedavg",
        "dqn_aux", "dqn_fedavg_aux", "dqn_local_4x"]
ABL = ["abl_nosym", "abl_nograph", "abl_noimag", "abl_norules", "abl_noshield", "abl_nounc", "abl_offline"]
AGG = ["trust", "gossip", "fedavg", "median", "krum", "fltrust", "balance"]
SCEN = ["semantic", "sybil", "signflip", "alie", "topo"]


def main():
    out = {"main": {}, "stats": {}, "calib": {}}
    for mth in MAIN:
        if load(mth):
            out["main"][mth] = {k: summary(mth, k) for k in METRICS}
    pv = {}
    for mth in MAIN[1:]:
        t = paired("proposed", mth)
        if t:
            out["stats"][mth] = t
            pv[mth] = t["p"]
    for mth, ph in holm(pv).items():
        out["stats"][mth]["p_holm"] = ph
    # secondary metrics vs strongest baselines
    out["stats_secondary"] = {}
    for mth in ["wm_uniform", "wm_fedavg", "playbook", "wm_local"]:
        out["stats_secondary"][mth] = {k: paired("proposed", mth, k) for k in
                                       ["comp_steps", "exfil", "breach", "avail", "fcr", "ttc"]}
    # belief quality
    for mth in ["proposed", "wm_uniform", "wm_fedavg", "wm_gossip", "wm_local"] + ABL:
        ds = load(mth)
        if not ds:
            continue
        out["calib"][mth] = {}
        for p in PROFS:
            for q in ["auroc_foot", "ece_foot", "brier_foot", "f1_lvl"]:
                v = [d["eval"][p].get(q) for d in ds.values() if d["eval"].get(p, {}).get(q) is not None]
                if v:
                    out["calib"][mth].setdefault(p, {})[q] = dict(mean=float(np.mean(v)),
                                                                  sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    # ablation
    out["ablation"] = {}
    pva = {}
    for a in ABL:
        if not load(a):
            continue
        t = paired(a, "proposed")  # positive delta: removal raises cost
        seeds = sorted(load(a))
        out["ablation"][a] = dict(J=t, summ={k: summary(a, k) for k in METRICS},
                                  comp=paired(a, "proposed", "comp_steps"), exfil=paired(a, "proposed", "exfil"),
                                  breach=paired(a, "proposed", "breach"), fcr=paired(a, "proposed", "fcr"),
                                  avail=paired(a, "proposed", "avail"), seeds=seeds)
        if t:
            pva[a] = t["p"]
    for a, ph in holm(pva).items():
        out["ablation"][a]["J"]["p_holm"] = ph
    # robustness
    out["robust"] = {}
    for sc in SCEN:
        out["robust"][sc] = {}
        pvr = {}
        for ag in AGG + ["noprecision", "nosybil_group", "nosecond_hand"]:
            tag = f"rob_{sc}_{ag}"
            ds = load(tag)
            if not ds:
                continue
            bm = [np.mean([c["byz_mass"] for c in d["comm"]]) for d in ds.values()]
            traj = np.mean([[c["byz_mass"] for c in d["comm"]] for d in ds.values()], 0).tolist()
            cons = np.mean([[c["consensus"] for c in d["comm"]] for d in ds.values()], 0).tolist()
            out["robust"][sc][ag] = dict(J=summary(tag, "J"), breach=summary(tag, "breach"),
                                         exfil=summary(tag, "exfil"), byz_mass=float(np.mean(bm)),
                                         byz_traj=traj, consensus=cons, seeds=sorted(ds))
            if ag != "trust":
                t = paired(f"rob_{sc}_trust", tag)
                if t:
                    out["robust"][sc][ag]["vs_trust"] = t
                    pvr[ag] = t["p"]
        for ag, ph in holm(pvr).items():
            out["robust"][sc][ag]["vs_trust"]["p_holm"] = ph
    # late joiners: metrics at sites 8-10 (indices 7, 8, 9) and at the others
    out["join"] = {}
    for ag in ["trust", "local", "gossip", "fedavg"]:
        tag = f"join_{ag}"
        if load(tag):
            out["join"][ag] = dict(new=summary(tag, "J", sites=[7, 8, 9]), old=summary(tag, "J", sites=list(range(7))),
                                   curve=np.mean([[c["ret"] for c in d["curve"]] for d in load(tag).values()], 0).tolist())
    # privacy
    out["privacy"] = {}
    for c in ["dp_s05", "dp_s10", "dp_s20"]:
        ds = load(c)
        if ds:
            out["privacy"][c] = dict(J=summary(c, "J"), breach=summary(c, "breach"),
                                     eps=float(np.mean([d["privacy"]["eps"] for d in ds.values()])),
                                     order=[d["privacy"]["order"] for d in ds.values()],
                                     steps=[d["privacy"]["steps"] for d in ds.values()],
                                     sigma=list(ds.values())[0]["cfg"]["dp_sigma"],
                                     calib={p: list(ds.values())[0]["eval"][p].get("auroc_foot") for p in PROFS},
                                     auroc_all=[d["eval"][p].get("auroc_foot") for d in ds.values() for p in PROFS],
                                     rule_use=[d.get("rule_use") for d in ds.values()],
                                     vs_proposed=paired(c, "proposed", seeds=sorted(ds)))
    out["privacy"]["proposed_same_seeds"] = summary("proposed", "J", seeds=[0, 1])
    # sensitivity
    out["sensitivity"] = {}
    for c in ["sens_H1", "sens_H3", "sens_H8", "sens_lam01", "sens_lam1", "sens_beta1", "sens_cand5", "sens_cand20"]:
        if load(c):
            out["sensitivity"][c] = dict(J=summary(c, "J"), vs_proposed=paired(c, "proposed", seeds=sorted(load(c))))
    out["sensitivity"]["proposed_same_seeds"] = summary("proposed", "J", seeds=[0, 1])
    # learning curves
    out["curves"] = {}
    for mth in ["proposed", "wm_uniform", "wm_fedavg", "wm_gossip", "wm_local"]:
        ds = load(mth)
        if ds:
            out["curves"][mth] = np.mean([[-c["ret"] for c in d["curve"]] for d in ds.values()], 0).tolist()
            out["curves"][mth + "_loss"] = np.mean([[c["loss"] if c["loss"] else [np.nan] * 6 for c in d["curve"]]
                                                    for d in ds.values()], 0).tolist()
    # efficiency, audit
    out["efficiency"] = {}
    for mth in ["proposed", "wm_local", "wm_gossip", "wm_fedavg", "abl_norules", "abl_noimag", "dqn_local", "dp_s10"]:
        ds = load(mth)
        if not ds:
            continue
        T = [d["timing"] for d in ds.values()]
        out["efficiency"][mth] = dict(
            plan_ms=float(np.mean([t["plan"] / max(1, t.get("n_plan", 0)) for t in T if t.get("n_plan")]) * 1000)
            if any(t.get("n_plan") for t in T) else None,
            train_ms=float(np.mean([t["train"] / max(1, t["n_train"]) for t in T]) * 1000),
            comm_ms=float(np.mean([t["comm"] / max(1, t["n_comm"]) for t in T if t.get("n_comm")]) * 1000)
            if any(t.get("n_comm") for t in T) else None,
            wall=float(np.mean([d["wall"] for d in ds.values()])),
            msgs=float(np.mean([np.mean([c["msgs"] for c in d["comm"]]) for d in ds.values()])) if list(ds.values())[0].get("comm") else None,
            dim=list(ds.values())[0]["comm"][0]["dim"] if list(ds.values())[0].get("comm") else None)
    ds = load("proposed")
    ru, cf = {}, []
    for d in ds.values():
        for k, v in d.get("rule_use", {}).items():
            ru[k] = ru.get(k, 0) + v
        cf += [(t["rule"], t["cf_gain"], t["action"]) for t in d["trace_sample"]]
    out["rule_use"] = ru
    byrule = {}
    for r, g, a in cf:
        byrule.setdefault(r, []).append(g)
    out["cf_gain"] = {r: dict(n=len(v), mean=float(np.mean(v)), pos=float(np.mean(np.array(v) > 0)))
                      for r, v in byrule.items()}
    out["_done"] = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(R, "*__s*.json")))
    json.dump(out, open(os.path.join(HERE, "paper_numbers.json"), "w"), indent=1, default=float)
    print("files", len(out["_done"]))
    for mth, s in out["main"].items():
        st = out["stats"].get(mth, {})
        print(f"{mth:16s} J={s['J']['mean']:.2f} [{s['J']['ci'][0]:.2f},{s['J']['ci'][1]:.2f}] n={s['J']['n_blocks']}"
              f"  p={st.get('p', float('nan')):.4f} wins={st.get('wins')}")


if __name__ == "__main__":
    main()
