# MM-edgeDP (Beginner-Friendly)

MM-edgeDP is a small, script-first project that lets you run (and later extend) experiments from **JSON config files** instead of editing notebook cells.

If you're new to ML codebases: that's totally fine — the goal here is that you can clone the repo, run a “dry run” without installing heavy ML packages, and then gradually turn on training once you’re ready.

Historical notebooks are kept unchanged for posterity:

- `kl_perturb.ipynb`
- `KL_perturb2.ipynb`
- `KL_perturb3.ipynb`

## 1) Where to run commands (repo root)

You should run commands **from the repo root** (the folder that contains `pyproject.toml`).

- If you cloned from GitHub, that usually looks like:

```bash
cd MM-edgeDP
```

- In Google Colab, the equivalent is:

```bash
%cd MM-edgeDP
```

- If you're inside the course workspace that contains many homeworks, it may look like:

```bash
cd final_project/MM-edgeDP
```

You can sanity-check you're in the right place by verifying you can see `configs/`, `scripts/`, and `mm_edgedp/`.

## 2) Quickstart (Google Colab + macOS)

This project is designed to be easy to run in **Google Colab** (most partners) and on **macOS locally** (optional).

### Google Colab quickstart (recommended)

In Colab, you can do a dry run (easy) or try the full training install.

Colab tips (important):

- Use `%cd ...` to change directories (it persists). `!cd ...` does not persist across cells.
- If you change the code in Colab, reinstall the package (see the “editable” note below).
- To use a GPU: Colab menu → `Runtime` → `Change runtime type` → choose `GPU`.

Dry run (recommended first):

Cell 1 — clone the repo and enter it:

```bash
!git clone https://github.com/ChefAltoids/MM-edgeDP
%cd MM-edgeDP
```

Cell 2 — install the project (lightweight) and run a dry run:

```bash
!python -m pip install -q --upgrade pip
!python -m pip install -q .
!python -m scripts.run_experiment --config configs/experiments/paper_v1_defensible.json --dry-run
```

Full install for training:

Cell 1 — clone the repo and enter it (same as above):

```bash
!git clone https://github.com/ChefAltoids/MM-edgeDP
%cd MM-edgeDP
```

Cell 2 — install training dependencies and run:

```bash
!python -m pip install -q --upgrade pip
!python -m pip install -q -r requirements-colab.txt
!python -m pip install -q .
!python -m scripts.run_experiment --config configs/experiments/paper_v1_defensible.json --set train.epochs=5
```

If installs behave strangely after switching runtimes (CPU ↔ GPU), try `Runtime` → `Restart runtime` and rerun the cells.

Editable install (only if you plan to edit the code in Colab):

```bash
!python -m pip install -q -e .
```

Saving results from Colab:

- Outputs are written to `outputs/` inside the Colab VM, which resets when your runtime resets.
- Quick download as a zip:

```bash
!zip -r outputs.zip outputs
```

If `torch-geometric` fails to install in Colab, that’s a common packaging issue (it depends on extra compiled wheels). In that case:

- You can still do `--dry-run` work.
- For full training, follow the official PyG install instructions for your exact Torch + CUDA version: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### macOS local quickstart (optional)

Recommended (keeps installs isolated):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Fast sanity check (no ML installs):

```bash
python -m pip install .
python -m scripts.run_experiment --config configs/experiments/paper_v1_defensible.json --dry-run
```

Run the baseline (downloads data + trains):

```bash
python -m pip install ".[train]"
python -m scripts.run_experiment --config configs/experiments/paper_v1_defensible.json --set train.epochs=5
```

Notes:

- The dataset `ogbn-arxiv` will download/cache under `data/` (see the config field `dataset.root`).
- Training uses GPU if available; otherwise CPU.

## 3) Folder layout (what things are)

- `mm_edgedp/`: reusable Python code (config loader + experiment runner). This is where new methods should live.
- `scripts/`: command-line entrypoints (`run_experiment`, `run_sweep`).
- `configs/`: JSON files that describe experiments.
	- `configs/base/`: shared defaults (dataset, trainer defaults, mechanism defaults).
	- `configs/tracks/`: “tracks” (small bundles of settings like `defensible` vs `exploratory`).
	- `configs/experiments/`: named experiments you can run.
	- `configs/sweeps/`: sweep definitions (e.g., seed/epsilon grids).
	- `configs/paper_protocol_v1.json`: legacy/original config kept for reference.
- `outputs/`: every run writes a folder here with:
	- `resolved_config.json`: the fully merged config used for the run
	- `metrics.json`: results/metrics from the run
- `notebooks/`: place for analysis notebooks that read from `outputs/`.

Note: `outputs/` is meant for generated artifacts and is typically gitignored (so you don’t accidentally commit large results folders).

## 4) What “config-driven experimentation” means

Instead of hard-coding settings in Python, you write them in small JSON files:

- Want to change a hyperparameter? Make a new config file.
- Want to run many variants? Use a sweep config.
- Want to reproduce a run later? Open the saved `resolved_config.json`.

### Which config fields matter right now?

Today, the scripted runner is intentionally minimal. These sections currently affect what actually runs:

- `run` (name, seed, output_dir)
- `dataset` (currently only supports `ogbn-arxiv`)
- `tasks` (turn baseline tasks on/off)
- `model` and `train` (baseline model/training hyperparameters)

Other sections like `privacy` and `mechanism` are still useful — they are **saved into `resolved_config.json`** so your experiment settings stay reproducible — but the baseline runner does not yet apply them to the model/training logic.

### Config inheritance with `extends`

Configs can “inherit” other configs with an `extends` list. Example (simplified):

```json
{
	"extends": [
		"configs/base/dataset_ogbn_arxiv.json",
		"configs/base/trainer_default.json",
		"configs/tracks/defensible.json"
	],
	"run": {"name": "my_run", "seed": 123}
}
```

Merge order is:

1. Parent files in `extends` (left → right)
2. The current file
3. Command-line overrides (`--set ...`)

### Command-line overrides (`--set`)

Overrides let you change a single value without creating a new file.

Examples:

```bash
python -m scripts.run_experiment \
	--config configs/experiments/paper_v1_defensible.json \
	--set privacy.epsilon=2.0 \
	--set run.seed=7 \
	--set train.epochs=10
```

Tip: override values are parsed as numbers/booleans when possible.

## 5) “Testing the idea” (practical workflow)

If you’re not sure anything is installed correctly, this sequence is usually the least painful:

1. **Dry run** (confirms config + output writing works)

```bash
python -m scripts.run_experiment --config configs/experiments/paper_v1_defensible.json --dry-run
```

2. **Tiny run** (confirms training works, but keeps it short)

```bash
python -m scripts.run_experiment \
	--config configs/experiments/paper_v1_defensible.json \
	--set train.epochs=1
```

3. **Sweep dry run** (confirms the sweep loop + naming works)

```bash
python -m scripts.run_sweep --config configs/sweeps/epsilon_grid_small.json --dry-run
```

## 6) Where to put new method ideas vs. ablations

Use this decision guide:

### “I only want to change settings” (best for most ablations)

Create a new JSON file under `configs/experiments/` that extends an existing experiment and changes a few fields.

For example, create a new experiment config that changes epochs, dropout, or which tasks run.

This is the recommended path for:

- hyperparameter ablations (epochs, lr, hidden size, etc.)
- changing which baseline tasks run (`tasks.run_logreg`, `tasks.run_gcn`)
- changing bookkeeping fields like `run.name`, `run.seed`

#### Example: make a new ablation config

Create a new file like `configs/experiments/my_first_ablation.json`:

```json
{
	"extends": [
		"configs/experiments/paper_v1_defensible.json"
	],
	"run": {
		"name": "my_first_ablation"
	},
	"tasks": {
		"run_logreg": false,
		"run_gcn": true
	},
	"train": {
		"epochs": 5
	}
}
```

Run it:

```bash
python -m scripts.run_experiment --config configs/experiments/my_first_ablation.json
```

### “I have a new Edge-DP method” (code change)

This is the path for **new Edge-DP methods** you want to compare against the baseline (for example: aggregation perturbation, randomized response on edges, degree-based mechanisms, etc.).

Right now, the runner **records** `privacy` / `mechanism` settings in `resolved_config.json`, but it does not automatically apply them. If you want a new DP method to actually affect results, you will need to implement it and wire it into the run.

Practical starting points:

- `mm_edgedp/runner.py` is the orchestrator: it loads data and runs the baseline tasks.
- `baseline.py` contains the current baseline training/eval code.

Suggested contribution pattern (beginner-friendly):

1. Create a new module under `mm_edgedp/` for your method (e.g. `mm_edgedp/aggregation_perturbation.py`).
2. Pick a simple config switch so experiments can select methods without editing Python:
	 - Recommended: `method.name` (string) + a method-specific params section.
3. Update `mm_edgedp/runner.py` to branch on `method.name` and apply your method.
	 - Common place to apply edge-DP methods: **after loading the graph** but **before** running `run_logreg` / `train_gcn`.
4. Add a new experiment config in `configs/experiments/` that turns your method on.
5. Compare against baseline by running two configs with the same `run.seed` and the same `privacy.epsilon`.

#### Example: wiring in an “aggregation perturbation” method

Create an experiment config like `configs/experiments/agg_perturb_example.json`:

```json
{
	"extends": ["configs/experiments/paper_v1_defensible.json"],
	"run": {"name": "agg_perturb_example"},
	"method": {
		"name": "aggregation_perturbation"
	},
	"privacy": {
		"epsilon": 1.0
	},
	"aggregation_perturbation": {
		"some_parameter": 0.1
	}
}
```

Then (after implementing the method + runner hook), run:

```bash
python -m scripts.run_experiment --config configs/experiments/agg_perturb_example.json
```

To do an epsilon comparison, you can either:

- run multiple commands with `--set privacy.epsilon=...`, or
- create a sweep config under `configs/sweeps/` that extends your method experiment and sets `sweep.epsilons`.

#### Contributing via GitHub (recommended)

If you want your method/ablation to be easy for others to reproduce:

1. Fork the repo on GitHub (or create a branch if you have write access).
2. Add code under `mm_edgedp/` and add one or more configs under `configs/experiments/`.
3. Run a quick check (in Colab or macOS):

```bash
python -m scripts.run_experiment --config configs/experiments/paper_v1_defensible.json --dry-run
```

4. Open a pull request and describe:
	- what method you implemented (e.g., “aggregation perturbation for edge-DP”)
	- which config(s) to run
	- what metrics/output file to look at (`outputs/<run_name>/metrics.json`)

### “Where should I do new work?”

Do new development work inside this repo (MM-edgeDP):

- **new method code** → `mm_edgedp/`
- **new experiment variants / ablations** → `configs/experiments/` (and optionally `configs/tracks/`)
- **analysis plots/tables** → `notebooks/` reading from `outputs/`

The KL perturbation notebooks are intentionally not edited; treat them as historical references.

## 7) Troubleshooting (common beginner issues)

- “I got `ModuleNotFoundError: torch`”
	- Use `--dry-run`, or install training deps: `python -m pip install ".[train]"`.

- “`torch-geometric` won’t install”
	- Very common on fresh environments. Follow PyG’s official install page for your Torch/CUDA combo:
		https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

- “A config in `extends` can’t be found”
	- Make sure you are running commands from the repo root (the folder with `pyproject.toml`).
	- Or pass `--repo-root .` explicitly (it defaults to `.`).

## 8) Optional: CLI shortcuts after install

If you installed the package (e.g. `pip install .`), you can also use the console scripts:

```bash
mm-edgedp-run --config configs/experiments/paper_v1_defensible.json --dry-run
mm-edgedp-sweep --config configs/sweeps/epsilon_grid_small.json --dry-run
```
