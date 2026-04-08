# Clinical-DP Setup

First, clone the repo 
```bash 
git clone https://github.com/ChefAltoids/Clinical-DP
```

Run these commands from the `final_project/Clinical-DP` directory.

## Option 1: Conda

```bash
conda create -n clinical-dp python=3.11 -y
conda activate clinical-dp
python -m pip install --upgrade pip
python -m pip install -e .
```

## Option 2: venv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Optional: Dry Run

```bash
python -m pip install -e . --dry-run
```
