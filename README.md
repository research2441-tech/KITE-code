# KITE: Trust-Aware Decentralized Cyber World Models with Rule-Guided Counterfactual Planning

Code for the manuscript *Trust-Aware Decentralized Cyber World Models with Rule-Guided Counterfactual Planning for Proactive Defense of Edge Networks*.

Everything sits in this one folder. All scripts run on CPU with one thread per run.

## Files

| File | Purpose |
|---|---|
| `data_prep.py` | Maps ToN-IoT, CICIoT2023 and CIC IoT-DIAD 2024 labels to six kill-chain categories, withholds the zero-day families, splits records by time order (60% exercise, 40% evaluation), applies robust scaling and 12-component PCA. Writes `pools.npz` and `pools_meta.json`. |
| `pools.npz`, `pools_meta.json` | Prepared exercise and evaluation pools (output of `data_prep.py`). |
| `cyberrange.py` | Multi-site edge cyber range: 10 hosts per site, an adaptive kill-chain attacker with four campaign profiles, a cost model and defender actions. |
| `wm.py` | Graph-conditioned latent world model: GRU posterior with host messages, a residual prior ensemble, belief heads and the kill-chain precedence loss. |
| `policy.py` | Neuro-symbolic layer (predicates, seven rules, three shield constraints) and the counterfactual imagination planner. |
| `fleet.py` | Per-site replay and validation buffers, training (optionally DP-SGD) and the RDP accountant. |
| `comm.py` | Decentralized exchange over time-varying directed links. It contains KITE's validation-evidence trust and the baseline rules: gossip, FedAvg, uniform, median, Krum, FLTrust and BALANCE. It also implements the attacks: label flip, Sybil, sign flip, ALIE and partition. |
| `run.py` | Experiment driver for every world-model configuration (`python run.py <config> <seed>`). |
| `dqn.py` | Model-free baselines: host-factored graph double DQN (local, FedAvg, auxiliary head, 4x budget). |
| `jobqueue.py`, `jobs_all.txt` | Parallel job runner and the complete list of runs in the paper. |
| `analyze.py` | Aggregates the results into `paper_numbers.json`: paired Wilcoxon tests on (seed, profile) blocks, Holm correction, rank-biserial effect sizes and bootstrap CIs. |
| `paper_numbers.json` | Every number quoted in the manuscript. |
| `all_results.json.gz` | Raw output of all runs, keyed by `<config>__s<seed>`. |

## Reproduce

```bash
pip install -r requirements.txt
# optional: rebuild the pools from the corpora (Parquet files with columns 'family', 'fine', 'order' + features)
python data_prep.py /path/to/parquet_dir .
# one run (about 6 minutes on one core)
python run.py proposed 0
python dqn.py dqn_local 0
# the full study (about 20 CPU-hours), then the statistics
python jobqueue.py jobs_all.txt 2
python analyze.py
```

Results go to `./results` unless `KITE_RESULTS` is set. Data is read from this folder unless `KITE_DATA` is set.

## Configurations

- **Main comparison:** `proposed`, `wm_uniform`, `wm_fedavg`, `wm_gossip`, `wm_local`, `playbook`, and the `dqn_*` agents.
- **Ablation:** `abl_nosym`, `abl_nograph`, `abl_noimag`, `abl_norules`, `abl_noshield`, `abl_nounc`, `abl_offline`.
- **Robustness:** `rob_<scenario>_<rule>`.
  - Scenarios: `semantic`, `sybil`, `signflip`, `alie`, `topo`.
  - Rules: `trust`, `gossip`, `fedavg`, `median`, `krum`, `fltrust`, `balance`.
  - Trust-component ablations: `rob_<scenario>_noprecision`, `_nosybil_group`, `_nosecond_hand`.
- **Late joiners:** `join_<rule>`.
- **Privacy:** `dp_s05`, `dp_s10`, `dp_s20`.
- **Sensitivity:** `sens_*`.

## Data

The flow records come from public corpora; follow each provider's terms:

- ToN-IoT (UNSW Canberra)
- CICIoT2023 and CIC IoT-DIAD 2024 (Canadian Institute for Cybersecurity)

The samples used here are the deduplicated, class-capped Parquet files distributed with the open TokenGauss-Former repository.

## License

MIT (see `LICENSE`).
