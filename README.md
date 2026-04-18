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

- Or, mount Google Drive and copy `outputs/` there.

Example:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/mm-edgedp-runs
!cp -r outputs /content/drive/MyDrive/mm-edgedp-runs/
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

### “I have a new algorithm / new mechanism” (code change)

Implement new logic in `mm_edgedp/`, then wire it into `mm_edgedp/runner.py`.

Practical starting points:

- `mm_edgedp/runner.py` is the orchestrator: it reads the config and decides what to run.
- `baseline.py` contains the current baseline training/eval code.

Suggested pattern:

1. Add a new module under `mm_edgedp/` (e.g. `mm_edgedp/my_method.py`).
2. Add a config flag (e.g. `method.name`) so you can select it from JSON.
3. Update `mm_edgedp/runner.py` to branch on that flag.
4. Make a new config in `configs/experiments/` that turns your method on.

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
