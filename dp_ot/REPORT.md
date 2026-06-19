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
  seen), label-dist L1 = 0.118 (small). The *right* kind of shift on paper.
- Result (ε=1, held-out target split): source 0.876, oracle 0.879 (+0.003),
  **oracle_fgw 0.878** (FGW verdict: "frames already aligned"), dp_hist 0.878
  (proto_l1 0.040), dp_exp 0.874, **target_oracle 0.943** (honest split — a real
  ~6.7-pt headroom over source).
- Reading (**revised twice** — earlier drafts said concept shift, then
  "structural"; the precise diagnosis is **structural *concept* shift**). Two
  probes localize it:
  - **Raw-feature probe:** logistic/MLP X→Y transfers with **concept_gap ≈ 0**
    (logistic +0.009, MLP −0.005). The **feature→label** map is *shared* — it is
    **not** feature concept shift.
  - **Summary probe** (features + 1-hop mean + log-deg — the space the prototypes
    live in): a **large** gap, **logistic +0.072, MLP +0.103**. Because the *MLP*
    gap is *larger* than the linear one, this is not a linear artifact — a target-
    trained conditional genuinely beats the transferred one. So the
    **structure→label** map *differs* across domains.
  Together: ACM and DBLP share how bag-of-words maps to topic, but differ in how
  *citation structure* maps to topic — a **structural concept shift**. That is
  why `oracle` (covariate mass), `oracle_fgw` (frame alignment), and any
  marginal-over-structure method fail: aligning the structural *marginal* cannot
  fix a structural *conditional* difference. FGW found a clean structural
  *correspondence* (cost 0.30, peak 1.0) yet recovered no gain — consistent, and
  it argues against mere disjoint structural support. Consistent with the graph-DA
  literature, where SOTA aligns *learned representations* (UDAGCN/AdaGCN), not
  marginals.

- **Scope of the claim (important):** the concept shift is measured in a *fixed*
  representation (the hand-crafted 1-hop summary). A *learned* domain-invariant
  representation might still align the conditional — that is exactly what
  adversarial graph-DA does, unsupervised on target, and where the ~6.7-pt gap is
  realistically recovered. So the correct statement is **"outside the reach of any
  fixed-representation post-processing method," not "irreducible."** Reaching it
  needs DP *representation learning* on the private target graph — a different,
  harder project that abandons the post-processing boundary.

- **Contrast with OGB (the probe distinguishes the two structural sub-types):**
  OGB's summary probe shows a *small* gap with **MLP ≤ logistic** (0.008 vs 0.015)
  → structural-*covariate* shift (statistics differ, P(Y|structure) shared).
  ACM/DBLP shows a *large* gap with **MLP ≥ logistic** → structural-*concept*
  shift. Same diagnostic, opposite verdict.

- **Why depth won't rescue it under DP:** the recoverable signal lives in deeper
  structure, but edge-DP sensitivity grows ~`d_max` per hop (1-hop perturbs 2
  nodes per edge → L1-sensitivity 4; 2-hop perturbs ~`2·d_max` nodes →
  sensitivity `O(d_max)`; K-hop ~`d_max^{K−1}`). So the headroom is **doubly out
  of reach**: outside fixed-representation post-processing *and* privacy-
  prohibitive to capture at the depth where it lives (this is why GAP perturbs
  per-hop and caps depth).

The earlier `target_oracle = 1.0` was a **memorization artifact** (trained and
evaluated on the same nodes, near-deterministic BoW). Fixed by a held-out target
split: every method is now scored on the same held-out target nodes and
`target_oracle` (0.943) is an honest ceiling.

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
  outside the method's scope: OGB-arxiv is label-prior shift *with no headroom*
  (honest `target_oracle` 0.9415 ≈ source 0.9362 — not adaptable by anything);
  ACM/DBLP has real headroom but it is **structural *concept* shift** — the
  feature→label map is *shared* (raw probe gap ≈ 0) while the structure→label map
  *differs* (summary probe gap +0.07/+0.10, MLP ≥ logistic), so no marginal-over-
  structure method (covariate reweighting, FGW, or DP-GAP-style structural
  alignment) can recover it. Real graph shifts are rarely the covariate-mass kind
  the method needs.
- Honest framing of the contribution: *a DP-by-post-processing, covariate-shift
  correction for graphs that provably recovers the non-private gain when
  covariate shift dominates, degrades gracefully (no harm) and self-diagnoses
  when it does not — plus a DP-feasibility **screening** suite (recoverable-gap
  oracle, FGW misalignment control, concept-shift probe, honest target split)
  that shows standard graph-DA benchmarks (label-prior, and structural/topology
  shift) lie outside its scope.*

## 8. Next steps

1. ✅ **Honest `target_oracle` split** — done. Held-out target test set; all
   methods scored on it (`run_experiment` `target_test_frac`, default 0.3).
   Revealed: OGB has ~0 headroom; ACM/DBLP has a real ~6.7-pt structural headroom.
2. ✅ **Shift-decomposition diagnostics** — done, and fully resolved. (a) **FGW
   alignment** (`adapt/fgw_align.py`) rules out frame *misalignment* (`oracle_fgw
   ≈ oracle`). (b) **Concept-shift probe** (`eval/diagnostics.py`, logistic + MLP)
   on raw features (gap ≈ 0 → feature→label map shared) and on the summary space
   (gap +0.07/+0.10, MLP ≥ logistic → structure→label map differs) pins the
   ACM/DBLP gap as **structural *concept* shift**. OGB by contrast is structural-
   *covariate* (summary gap small, MLP ≤ logistic). Both wired into
   `colab_diagnostics.ipynb`.
3. **Conclusion on direction.** The recommended path is now **Path A — the DP-
   feasibility *screening* contribution** (recoverable-gap oracle + FGW
   misalignment control + concept probe + honest split), which cleanly
   characterizes why post-processing reweighting fails on the canonical graph-DA
   benchmark. **Downgraded:** the "DP structural adaptation without labels" idea —
   the ACM/DBLP structural shift is *concept*-type, so unsupervised structural-
   marginal alignment (DP-GAP) cannot recover it; only learned domain-invariant
   representations can, i.e. **DP adversarial representation learning on the
   private target graph** — a separate, harder project that abandons the post-
   processing boundary. (A 2-hop summary would only *confirm* the finding while
   being privacy-prohibitive to deploy — sensitivity ~`d_max` per hop — so it is
   not worth pursuing.)
4. Replicate synthetic with seeds + the capacity sweep (`hidden ∈ {4,8,16,32}`) to
   show the misspecification gap open and close.

---

### Implementation notes

- Package: `dp_ot/` — `data/` (synthetic generator + real loaders),
  `models/gnn.py` (GraphSAGE, weighted loss, held-out `train_mask`), `adapt/`
  (prototypes, DP mechanisms, reweighting, `fgw_align.py` FGW alignment), `eval/`
  (metrics with held-out `mask`, plots, `diagnostics.py` concept-shift probe),
  `run_experiment.py`, `sweep.py`.
- Notebooks: `colab_experiments.ipynb` (synthetic), `colab_real_data.ipynb`
  (OGB-arxiv + ACM/DBLP, full pipeline), `colab_diagnostics.ipynb` (fast — the
  concept-shift probe + prototype-level FGW read, no GNN training).
- Diagnostics are torch-free (numpy/sklearn/POT) and unit-validated:
  `tests/test_fgw_alignment.py` (rotation recovery), `tests/test_concept_shift.py`
  (flipped-conditional vs covariate-only controls). POT is now a hard dependency.
- Incidental fixes along the way: a `.gitignore` `data/` rule was silently
  excluding the `dp_ot/data/` *source* package (anchored to `/data/`); OGB-arxiv
  needs `torch.load(weights_only=False)` under PyTorch ≥2.6; the dead Twitch host
  (`graphmining.ai`) was replaced by OGB/ACM-DBLP real-data paths.
