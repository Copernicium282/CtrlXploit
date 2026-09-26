# Architecture: NetWorldModel v2 (SIH 26153 · NTRO)

## 1. Problem framing

A detector says "this traffic is malicious". NTRO asks for a **forecaster**: given the traffic seen so far,
what happens next, and how soon? We model each monitored host (or the whole network, when host
addresses are unavailable) as a trajectory of one-minute **state cells** and predict:

* **binary:** y<sub>t</sub> = 1 ⇔ an exploitation-stage cell (Initial Access, Lateral Movement, C2, Exfiltration, Impact) occurs in (t, t+10 min];
* **stage:** the MITRE ATT&CK stage the host will be in at t+1 … t+10 minutes;
* **novelty:** how surprising the observed traffic is under the learned dynamics, with no labels.

Every cell also carries a phase: `pre_attack` (exploitation not yet visible, but coming), `during_attack` or `benign`.
Metrics are always reported per phase, so detecting an ongoing attack is never counted as forecasting (idea: csxzor-devcs/26153).

## 2. Data (real traffic first)

| Dataset | Granularity | Size after conversion | Use |
|---|---|---|---|
| **CTU-13** (13 botnet captures, CTU University, CC BY 2.0) | one cell per (scenario, source host, minute) | 534,947 cells · 1,296 hosts · 35 infected | primary |
| **CSE-CIC-IDS2018** (9 day files) | one cell per (day, minute), since the processed CSVs have no host IPs | 4,391 cells | secondary |
| Synthetic campaigns (CIC-2018 CSV layout) | network minute | 19,200 cells | controlled benchmark: the only data with real pre-infection periods |

Raw downloads are streamed and converted one file at a time, then deleted. The whole project uses about 0.3 GB against a 7 GB budget.

**CTU-13 stage labels** come from the capture's own per-flow annotations (`constants._ctu_botnet_stage`):
- spam, SMTP and click-fraud → Impact (T1496 resource hijacking);
- DNS lookups, connection attempts and ICMP → Reconnaissance;
- CC / IRC / P2P / custom-encrypted / established → C2;
- malicious internal SMB/RPC/RDP/SSH flows → Lateral Movement (behavioural rule).

A cell takes its *dominant* malicious stage (HowSuyash/AttackForecast's argument).

**Honest dataset fact:** as HowSuyash showed, every infected CTU-13 host is malicious from the first minute it appears.
There is no per-host pre-infection baseline, so on CTU-13 "pre-attack" means reconnaissance that precedes C2 or spam, and
**stage progression** is the genuinely forecastable quantity.

## 3. Pipeline

```mermaid
flowchart LR
  A[CTU-13 binetflow · CIC CSV<br>NetFlow CSV · Parquet · PCAP/PCAPNG] --> B[ingest<br>alias schema · chunked · bad-row tolerant<br>Argus State → TCP flags]
  B --> C[state cells<br>hourly parquet buckets (out-of-core)<br>per host or per network · 39 features]
  C --> D[multi-scale state s_t<br>instant · burst vs EMA3 · EMA15 trend = 117-d]
  D --> E[NetWorldModel<br>LSTM + causal Transformer → RSSM]
  E --> F[engine<br>32 particles × K=10 imagined steps<br>fusion · Markov prior · surprise]
  F --> G[calibrated alerts · host triage<br>MITRE mapping · ETA · attention + GradientSHAP]
  G --> H[Streamlit SOC dashboard · CLIs · reports]
```

**Features per cell (39):**
- volume;
- source/destination/port cardinalities and entropies;
- TCP flag ratios and flag-bitmask entropy;
- **IP fragment flags:** share of packets with MF set (fragments) and with DF set (PCAP only);
- **payload-size distribution:** mean, total-variance spread and the share of scan-sized (≤ 64 B) packets (PCAP only);
- **port-sweep shape:** mean |Δport| between the successive destination ports a cell touches, and the share of steps
  within 2 ports — a sequential sweep walks the port space, a randomised scan jumps across it;
- TTL mean and total-variance spread (PCAP only);
- initial window size;
- inter-arrival mean/std/**max** (max where the source publishes it: CICFlowMeter CSVs and PCAP);
- retransmissions;
- small-flow, lateral, sensitive-port, UDP and DNS shares;
- a beaconing-periodicity score.

Each feature also appears as a burst term (deviation from a 3-minute EMA) and a trend term (15-minute EMA), giving 117 inputs.

**Honest coverage note.** CTU-13 binetflow files are Argus flow exports: they carry no TTL, no IP flags, no packet-level
payload sizes and no IAT maximum, so a CTU-13 cell reads 0 for those attributes and records the absence in its
`ttl_missing` / `frag_missing` / `payload_missing` / `iat_max_missing` flag. CSE-CIC-IDS2018 carries the TCP window and
`Flow IAT Max` only. Nothing is back-filled: a dataset build records the real per-dataset share under
`feature_coverage` in `dataset_stats.json`, and packet-level training signal comes from raw PCAP and the synthetic
generator. The bundles and reports shipped in `models/` and `reports/` were trained before these features existed
(93-d state = the earlier 31 cell features x instant/burst/trend); the app and `explain` take their width and labels
from the bundle, so those artifacts still load and explain, but only a `make data && make train` re-run puts the
packet-level block into training or into any reported score.

## 4. NetWorldModel

| Part | Definition |
|---|---|
| Context encoder | e<sub>t</sub> = MLP(x<sub>t</sub>); c<sub>1:t</sub> = causal Transformer(LSTM(e<sub>1:t</sub>)), with attention weights exposed |
| Dynamics | h<sub>t</sub> = GRU(h<sub>t-1</sub>, z<sub>t-1</sub>); prior p(z<sub>t</sub>\|h<sub>t</sub>) is the learned P(S<sub>t+1</sub>\|S<sub>t</sub>) |
| Filter | posterior q(z<sub>t</sub>\|h<sub>t</sub>, c<sub>t</sub>) |
| Heads | decoder x̂<sub>t</sub>; 7-way stage head; direct forecast head |

**Loss:** reconstruction + β·max(KL, free nats) + stage CE + forecast BCE (pre-attack positives ×3), plus **K-step latent
overshooting**: from every context position the prior is rolled K steps and supervised with the true future stages and observations.

**Regularisation:** telemetry jitter, and a **benign flash-crowd augmentation**. Benign-only sequences get ×2–×20 volume
surges with labels kept benign. This took false alerts on 5× benign volume from 36% to 0% (idea: PratikBorle/miniproject).

**Training:** index-gathered mini-batches (no materialised sequence tensors), benign host-slots sub-sampled, 3 seeds.
The seed with the best validation score becomes the primary bundle.

## 5. Engine

For every cell:
1. Filter the posterior over the last 24 minutes.
2. Draw 32 particles and roll the prior 10 steps with no observations.
3. Fuse three estimators: the direct head, the rollout (particle mean of max-step exploit probability) and a Markov first-passage probability over a Dirichlet-smoothed kill-chain transition matrix.

Everything is chosen on validation only:
- the fusion weights and threshold, which maximise F1 subject to FPR ≤ 5%;
- an optional **ensemble** with the best tree baseline.

**Surprise channel:** the prior's one-step prediction error on x<sub>t</sub>, and KL(q‖p). Each is robust-z-scored (median/MAD) within its capture.
Its sign is chosen on validation, because botnet traffic can be *less* surprising than human traffic (HowSuyash).
Host triage reports the **risk** and **anomaly** channels separately and never blends them.

## 6. Evaluation protocol

| Protocol | What it asks |
|---|---|
| **temporal** | train on the first 70% of every capture, validate on the next 15%, test on the last 15%, with context+horizon purge gaps (deployment shape) |
| **family** | CTU-13 train 1-4,6,7,9-11 · val 12 · test 5,8,13 (unseen Virut/Murlo). For CIC-2018: held-out days |

**Methods, all on identical features, scaler, labels, splits and threshold rule:**
- oracle persistence;
- LR single-window and LR stacked-8;
- gradient boosting stacked-8;
- random forest stacked-8;
- LSTM classifier;
- the NetWorldModel variants and the ensemble;
- for stages: persistence (oracle and inferred), the Markov prior, and a per-horizon stacked multinomial LR.

**Reported:** mean ± std over 3 seeds, and 95% bootstrap CIs resampled over host-slots.

**Integrity checks run on every evaluation (`engine/leakage.py`):**
- split integrity;
- schedule-only probe (ShadowCat);
- capture-identity probe;
- label-permutation significance;
- no-peeking gradient test (HowSuyash);
- out-of-distribution benign regression (PratikBorle).

## 7. Results

See `reports/<dataset>/<protocol>/evaluation.md` and `eval/results/comparison.md` for full tables. Summary in §7 of
[docs/README.md](README.md).
