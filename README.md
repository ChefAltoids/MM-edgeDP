# MM-EdgeDP

**MM-EdgeDP** is an edge-level differentially private node classification method for graphs. It achieves pure ε-DP on edges by combining a *public prototype dictionary* with the *exponential mechanism*: instead of adding Gaussian noise to aggregated features, the mechanism selects a prototype subgraph from a public dictionary in a way that is provably ε-differentially private with respect to the private graph structure.

Historical KL-perturbation notebooks (`kl_perturb.ipynb`, `KL_perturb2.ipynb`, `KL_perturb3.ipynb`) with early experimentation are kept unchanged for reference.

---

## Method overview (4 stages)

**Stage A — Public dictionary construction.**
A fraction `public_frac` of nodes (and the subgraph they induce) is treated as fully public. For each class label, the public subgraph is mined to produce a pool of candidate prototype subgraphs, which are then filtered (size, purity, connectivity, overlap) and diversified via k-center selection into a dictionary of `dict_per_class` prototypes per class.

**Stage B — Private query computation.**
Each private node computes a soft-normalized 1-hop mean of its neighbors' feature vectors, using only the private graph edges. This query vector summarizes the node's local structure without releasing edge identities.

**Stage C — Exponential mechanism assignment.**
For each private node, the exponential mechanism samples one of the `C × dict_per_class` dictionary prototypes (where C = number of classes) with probability proportional to `exp(ε × utility(query, prototype) / sensitivity)`. By composition, this step is 2ε-DP over the full private edge set.

**Stage D — GCN training on synthetic data.**
The assigned prototypes form the training graph; a small two-layer GCN is trained on this synthetic data. Because prototype assignment is post-processed from an already-privatized output, this stage is free under the DP guarantee (post-processing theorem).

---

## File layout

| File | Purpose |
|---|---|
| `experiments.py` | Core pipeline — `run_experiment(config)` runs all 4 stages and returns metrics |
| `sweep.py` | Hyperparameter grid runner for Phase 1 and Phase 2 |
| `baselines.py` | All 5 comparison methods + white-box and black-box privacy attacks (Phase 3) |
| `results.ipynb` | Loads output CSVs and produces all plots and tables |
| `configs/sweep_phase1.json` | Phase 1 grid: `public_frac × dict_per_class` |
| `configs/sweep_phase2.json` | Phase 2 grid: `epsilon` sweep |
| `outputs/sweep_phase1.csv` | Written by `python sweep.py --phase 1` |
| `outputs/sweep_phase2.csv` | Written by `python sweep.py --phase 2` |
| `outputs/baseline_results.csv` | Written by `python baselines.py` |

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch torch-geometric
pip install numpy pandas matplotlib seaborn scikit-learn
```

For PyG, follow the official install instructions for your Torch + CUDA combination:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 2. Run Phase 1 — hyperparameter grid (public_frac × dict_per_class)

```bash
python sweep.py --phase 1
```

Results are written to `outputs/sweep_phase1.csv`. To preview without running:

```bash
python sweep.py --phase 1 --dry-run
```

To resume after an interruption (skips already-completed rows):

```bash
python sweep.py --phase 1 --resume
```

### 3. Run Phase 2 — epsilon sweep

```bash
python sweep.py --phase 2
```

Results are written to `outputs/sweep_phase2.csv`. Phase 2 uses the best `dict_per_class` from Phase 1 (set in `configs/sweep_phase2.json`).

### 4. Run Phase 3 — baseline comparison with privacy attacks

```bash
python baselines.py --datasets Cora AmazonPhoto --epsilon 1.0 --seeds 0 1 2
```

Runs all 5 methods (Gaussian SGC, Edge RR, GAP-EDP, Public GCN, MM-EdgeDP) plus white-box and black-box edge inference attacks on each dataset. Results are written to `outputs/baseline_results.csv`.

### 5. View results

Open `results.ipynb` in Jupyter. It loads the three CSVs and produces:

- Phase 1: heatmap of mean val accuracy over `public_frac × dict_per_class`
- Phase 2: privacy-utility curve (accuracy vs. ε)
- Phase 3: grouped accuracy bar chart, privacy-utility scatter (BB AUC vs. accuracy), per-dataset attack AUC bars, summary table

---

## sweep.py — reference

```
python sweep.py --phase 1            # Phase 1 grid
python sweep.py --phase 2            # Phase 2 epsilon sweep
python sweep.py --config configs/sweep_phase2.json   # any JSON config
python sweep.py --phase 1 --set dict_per_class=64    # override a grid value
python sweep.py --phase 1 --set epochs=10            # override a base config value
python sweep.py --phase 1 --dry-run                  # print configs, no training
python sweep.py --phase 1 --resume                   # skip completed rows
python sweep.py --phase 1 --output my_results.csv    # custom output path
```

**Sweep JSON format** (`configs/sweep_phase1.json`):

```json
{
  "description": "...",
  "grid": {
    "dataset":        ["Cora"],
    "epsilon":        [1.0],
    "public_frac":    [0.02, 0.05, 0.20],
    "dict_per_class": [4, 16, 64, 128],
    "seed":           [0, 1, 2]
  },
  "base_config": {}
}
```

All combinations of `grid` values are run as a Cartesian product. `base_config` keys are applied to every config in the grid. `--set KEY=VALUE` overrides win over both.

---

## baselines.py — reference

```
python baselines.py                                  # default: Cora, AmazonPhoto, Actor
python baselines.py --datasets Cora AmazonPhoto      # choose datasets
python baselines.py --epsilon 2.0                    # privacy budget
python baselines.py --public_frac 0.20               # public fraction
python baselines.py --dict_per_class 128             # dictionary size (MM-EdgeDP only)
python baselines.py --seeds 0 1 2                    # random seeds
python baselines.py --resume                         # skip completed rows
python baselines.py --dry-run                        # print configs, no training
python baselines.py --verbose                        # extra training output
```

**Methods compared:**

| Method | DP type | Notes |
|---|---|---|
| Gaussian SGC | (ε,δ)-DP | Gaussian noise on 1-hop aggregation |
| Edge RR | ε-DP | Randomized response on each edge (limited to ≤5000 train nodes) |
| GAP-EDP | (ε,δ)-DP | Multi-hop Gaussian PMA |
| Public GCN | none | Trained on public nodes/edges only; sets the feature-leakage floor |
| MM-EdgeDP | ε-DP | This paper's method |

**Privacy attacks:** for each method, a logistic regression attack is trained on held-out edge pairs using either white-box features (penultimate embeddings, Hadamard product of node pairs) or black-box features (softmax outputs, disagreement and co-activation scores). Attack AUC ≈ 0.5 means no edge information is recoverable; AUC ≈ 1.0 means near-perfect inference. Compare methods against the Public GCN AUC (feature-distribution floor), not against 0.5.

---

## experiments.py — configuration reference

`run_experiment(config, verbose=True)` merges `config` into `DEFAULT_CONFIG` and runs the full 4-stage pipeline. Supported keys:

| Key | Default | Description |
|---|---|---|
| `dataset` | `'Cora'` | Dataset name (see below) |
| `seed` | `42` | Global random seed |
| `public_frac` | `0.20` | Fraction of nodes treated as public |
| `val_frac` | `0.20` | Fraction of nodes used for validation |
| `epsilon` | `1.0` | Privacy budget ε for exponential mechanism |
| `dict_per_class` | `8` | Dictionary entries per class |
| `label_conditioning` | `True` | Whether to condition prototype selection on class label |
| `use_kcenter` | `True` | Use k-center diversification (False = random selection) |
| `kcenter_lambda` | `0.5` | Balance between diversity (0) and coverage (1) |
| `encoder_hidden` | `64` | Hidden size of the public GCN encoder |
| `encoder_out` | `32` | Output dimension of the public encoder |
| `encoder_epochs` | `200` | Training epochs for the public encoder |
| `epochs` | `50` | Training epochs for the downstream GCN |
| `lr` | `0.01` | Learning rate for the downstream GCN |

---

## Supported datasets

| Dataset | Type | Notes |
|---|---|---|
| `Cora` | Citation (homophilic) | Default; downloads automatically via PyG |
| `AmazonPhoto` | Co-purchase (homophilic) | Downloads automatically via PyG |
| `Actor` | Co-occurrence (heterophilic) | Set `label_conditioning=False` in config |

---

## Troubleshooting

**Edge RR segfaults or raises RuntimeError on large datasets.**
Edge RR allocates an O(n²) adjacency matrix. It is blocked for graphs with more than 5000 training nodes. On affected datasets it will be written as an error row in the CSV; all other methods will still run.

**`torch-geometric` won't install.**
Follow the official PyG installation guide for your specific PyTorch and CUDA version:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

**Results CSV is missing or empty.**
Make sure you run the scripts from the repo root (the directory containing `sweep.py`). The `outputs/` directory is created automatically on first write.
