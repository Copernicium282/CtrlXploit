# Portable venv interpreter: .venv/bin/python on POSIX, .venv/Scripts/python.exe on Windows.
# `make` is not installed on Windows by default; use the PowerShell helper `./make.ps1 <target>` there instead
# (Windows PowerShell 5.1 or PowerShell 7).
PY ?= $(shell test -x .venv/Scripts/python.exe && echo .venv/Scripts/python.exe || echo .venv/bin/python)
# Interpreter used to *create* the venv, so it must be a system python: python3/python here
# (Git Bash also sees the Windows PATH) or the Windows `py` launcher.
BOOTSTRAP_PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || command -v py 2>/dev/null)
DATASETS ?= ctu13 cic2018 synthetic

.PHONY: all setup fetch-data data train evaluate eval benchmark test app ctu13 cic2018 synthetic clean-raw

all: data train evaluate eval benchmark test

setup:
	@test -n "$(BOOTSTRAP_PYTHON)" || { echo "no system python found (looked for python3, python, py). Install Python 3.10+, then re-run: make setup"; exit 1; }
	$(BOOTSTRAP_PYTHON) -m venv .venv && $(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

fetch-data:  ## prebuilt data/ from Hugging Face (~260 MB) instead of rebuilding with `make data`
	$(PY) -m pip install -q -U "huggingface_hub>=0.24"
	$(PY) -c "import os; from huggingface_hub import snapshot_download; skip = os.path.exists('data/processed/ctu13/cells.parquet') and not os.environ.get('FORCE_FETCH'); snapshot_download(repo_id='ir192m2Cn282/ThreatAhead', repo_type='dataset', local_dir='.', allow_patterns=['data/**']) if not skip else print('fetch-data: data/ already present - set FORCE_FETCH=1 to re-sync from HF')"

data:        ## real CTU-13 (streamed, per-host), CSE-CIC-IDS2018 (streamed day by day), synthetic
	$(PY) -m sih_v2 build_dataset --dataset ctu13
	$(PY) -m sih_v2 build_dataset --dataset cic2018
	$(PY) -m sih_v2 build_dataset --dataset synthetic --synthetic

train:       ## multi-seed world model + every baseline, per dataset/protocol
	$(PY) -W ignore -m sih_v2 train --dataset ctu13 --protocol temporal
	$(PY) -W ignore -m sih_v2 train --dataset ctu13 --protocol family
	$(PY) -W ignore -m sih_v2 train --dataset cic2018 --protocol temporal
	$(PY) -W ignore -m sih_v2 train --dataset synthetic --protocol temporal

evaluate:
	$(PY) -W ignore -m sih_v2 evaluate --dataset ctu13 --protocol temporal
	$(PY) -W ignore -m sih_v2 evaluate --dataset ctu13 --protocol family
	$(PY) -W ignore -m sih_v2 evaluate --dataset cic2018 --protocol temporal
	$(PY) -W ignore -m sih_v2 evaluate --dataset synthetic --protocol temporal

eval:        ## cross-dataset comparison + ablations -> eval/results/
	$(PY) eval/compare_baselines.py
	$(PY) eval/ablation.py --dataset ctu13 --protocol temporal

benchmark:
	$(PY) -W ignore -m sih_v2 benchmark --dataset ctu13 --protocol temporal --skip-eval

test:
	$(PY) -m pytest -q

app:
	$(PY) -m streamlit run app.py

ctu13 cic2018 synthetic:   ## one dataset end to end, e.g. `make ctu13`
	$(PY) -m sih_v2 build_dataset --dataset $@
	$(PY) -W ignore -m sih_v2 train --dataset $@
	$(PY) -W ignore -m sih_v2 evaluate --dataset $@
