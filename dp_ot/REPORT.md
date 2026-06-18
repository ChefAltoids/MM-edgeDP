# DP-OT: Differentially-Private Covariate-Shift Adaptation of Public GNNs to Private Graphs

*Working report — synthetic and real-data experiments.*

## 1. Goal

Train a GNN entirely on a **public source graph** `G_S`, then adapt it to a
**private target graph** `G_T` using `G_T` only through a differentially-private
estimate of how target-domain mass differs from source over public-graph
prototypes. That DP estimate reweights the public training examples (importance
weighting). The final model is **(ε,δ)-edge-DP w.r.t. `G_T` by post-processing** —
we never train on the private graph directly.

## 2. Method

1. **Train** a GraphSAGE classifier on the public `G_S` (zero privacy cost).
2. **Prototypes:** K-means over *bounded local summaries* of each node —
   `[clip_B(x) | capped 1-hop mean | log-degree]` — which have low edge-DP
   sensitivity. Prototypes are fit in this **summary space** (not GNN-embedding
   space) so source and private-target nodes live in the same space.
3. **DP target-mass estimate** over the prototypes, two mechanisms:
   - **A — Laplace histogram:** hard-assign target nodes to prototypes, count,
     add `Lap(4/ε)` (L1-sensitivity 4: one edge moves ≤2 nodes' bins).
   - **B — Exponential mechanism:** per-node Gumbel-max sampling over distance to
     prototypes.
4. **Reweight** source nodes by `α_target / α_source` over their prototype.
5. **Retrain** the GNN on `G_S` with per-node weights; **evaluate inductively**
   on `G_T`.

Baselines per run: `source_only` (uniform weights), `oracle` (non-private true
target mass), `dp_histogram`, `dp_exponential`, `target_oracle` (trained on
`G_T`, upper bound).

## 3. Key engineering fix to Mechanism B

The exponential mechanism originally scored prototypes by **squared** distance,
whose sensitivity scales with the embedding diameter (`Δ_C ≈ 183` for the
synthetic config) → near-uniform sampling → useless mass estimate
(`proto_l1 ≈ 0.99`). Switching to **linear** distance gives a tight sensitivity
`Δ_C = Δ_s ≈ 1.85` via the triangle inequality. Even after the fix, at ε=1 the
EM concentrates too weakly to be competitive; **the Laplace histogram dominates
it on every dataset** (a legitimate result — the histogram's L1-sensitivity is a
flat 4 regardless of geometry).

## 4. Synthetic results — the mechanism works in its home turf

Constructing a synthetic problem where adaptation is *testable* was itself the
main difficulty. Pure covariate shift + a flexible learner gives no oracle gain;
concentrating the target onto one Gaussian saturates the labels (target ~90–96%
one class → AUROC at chance). The fix — `make_misspecified_pair` — keeps labels
balanced (component means orthogonalized against the label axis) while making the
target-optimal predictor differ from the source-optimal one (a per-cluster sign
flip → XOR-like `P(Y|Z)`, still shared between source and target → genuine
covariate shift). With a capacity-limited model:

| method | AUROC | proto_l1 | recovery of oracle gain |
|---|---|---|---|
| source_only | 0.469 | — | — |
| **oracle** | **0.647** | 0.000 | (gain **+0.178**) |
| dp_histogram | 0.654 | 0.202 | **~100%** |
| dp_exponential | 0.547 | 1.263 | 44% |
| target_oracle | 0.896 | — | upper bound |

*(misspec_support, ε=1.0, K=32, seed 0.)* At ε=1 the **DP histogram privately
recovers essentially the full non-private adaptation gain.** A causal-supporting
signal: recovery **tracks mass-estimate quality** — the EM, with a much worse
estimate (proto_l1 1.26), recovers far less (44%).

**Caveats (the home turf is favorable):** the shift is a categorical reweighting
of clusters the prototypes recover exactly; the misspecification is cluster-
aligned with the shift; the effect size depends on the chosen capacity limit;
the (n, ε) point is forgiving; `source_only` below 0.5 is an adversarial extreme;
single seed. This is a valid proof-of-concept, not evidence of real-world value.

## 5. Real-data results — the method does not gain on either benchmark

We characterize each dataset's shift *before* running (feature MMD vs label-
distribution L1), then run the full pipeline.

### 5a. OGB-arxiv (temporal split, pre-2018 → 2018+)

- Shift profile: **label-prior shift.** feature MMD² = 0.034 (tiny), label-dist
  L1 = 0.644 (large — new subfields emerge).
- Result (ε=1): source 0.937, **oracle 0.942 (+0.004)**, dp_hist 0.942
  (proto_l1 0.003), dp_exp 0.936, target_oracle 0.947.
- Reading: the features barely move, so covariate reweighting has nothing to do —
  and correctly does almost nothing. The large *label* shift is a different
  problem covariate reweighting cannot (and should not) address.

### 5b. ACM→DBLP (ArnetMiner cross-domain citation networks)

The canonical covariate-shift graph-DA benchmark (shared 6775-dim features,
shared 5-class labels).

- Shift profile: **covariate-shift-dominant** — feature MMD² = 0.093 (highest
  seen), label-dist L1 = 0.118 (small). The *right* kind of shift.
- Result (ε=1): source 0.887, **oracle 0.878 (−0.009)**, dp_hist 0.877
  (proto_l1 0.040), dp_exp 0.876, target_oracle 1.000\*.
- Reading: despite real covariate shift, perfectly-weighted source gives **no**
  gain. The residual domain gap is **conditional/concept shift** (`P(Y|X)`
  differs between ACM and DBLP), which importance weighting provably cannot fix.
  This is consistent with the graph-DA literature, where SOTA methods use
  adversarial feature alignment rather than reweighting.

\* **Measurement caveat:** `target_oracle` is trained and evaluated on the same
target nodes (no split); with near-deterministic BoW features it memorizes the
graph and reports AUROC 1.0. It is **not** a fair upper bound and should use a
target train/test split. The clean comparison is `source_only` vs `oracle`, both
evaluated on the held-out target graph.

## 6. Properties that hold on real data

- **Graceful degradation / no harm:** when there is no covariate gain to exploit,
  `dp_histogram` collapses to `source_only` (within ~0.01 AUROC) rather than
  damaging it. (The exponential mechanism mildly *hurts* — another point for the
  histogram.)
- **Self-diagnosis:** the built-in oracle-gain diagnostic correctly reported "no
  gain to recover" on both real datasets before any over-interpretation.

## 7. Conclusions and status

- The DP mechanism is sound: when covariate shift is the operative shift, the
  **Laplace histogram privately recovers the non-private adaptation gain at ε≈1**
  (demonstrated on synthetic), and the **histogram robustly dominates the
  exponential mechanism** for mass estimation.
- The **positive claim is currently synthetic-only.** Both real benchmarks fall
  outside the method's scope: OGB-arxiv is label-prior shift; ACM/DBLP carries
  concept shift. Real graph shifts are rarely the covariate-only kind the method
  needs.
- Honest framing of the contribution: *a DP-by-post-processing, covariate-shift
  correction for graphs that provably recovers the non-private gain when
  covariate shift dominates, degrades gracefully (no harm) and self-diagnoses
  when it does not — with a characterization of why standard graph-DA benchmarks
  (label/concept shift) lie outside its scope.*

## 8. Next steps

1. **Fix `target_oracle`** to use a target train/test split (honest upper bound on
   all datasets).
2. **Add a covariate-vs-concept-shift diagnostic** — train held-out X→Y probes on
   source and target; the transfer gap quantifies the *irreducible* concept-shift
   component and screens a dataset for whether the method *could* help before
   running the full pipeline.
3. Replicate synthetic with seeds + the capacity sweep (`hidden ∈ {4,8,16,32}`) to
   show the misspecification gap open and close.

---

### Implementation notes

- Package: `dp_ot/` — `data/` (synthetic generator + real loaders),
  `models/gnn.py` (GraphSAGE, weighted loss), `adapt/` (prototypes, DP
  mechanisms, reweighting), `eval/` (metrics, plots), `run_experiment.py`,
  `sweep.py`.
- Notebooks: `colab_experiments.ipynb` (synthetic), `colab_real_data.ipynb`
  (OGB-arxiv + ACM/DBLP).
- Incidental fixes along the way: a `.gitignore` `data/` rule was silently
  excluding the `dp_ot/data/` *source* package (anchored to `/data/`); OGB-arxiv
  needs `torch.load(weights_only=False)` under PyTorch ≥2.6; the dead Twitch host
  (`graphmining.ai`) was replaced by OGB/ACM-DBLP real-data paths.
