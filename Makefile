# Portable venv interpreter: .venv/bin/python on POSIX, .venv/Scripts/python.exe on Windows.
# `make` is not installed on Windows by default; use `pwsh ./make.ps1 <target>` there instead.
PY ?= $(shell test -x .venv/Scripts/python.exe && echo .venv/Scripts/python.exe || echo .venv/bin/python)

# Interpreter used to create the venv, so it must be a system python on PATH.
BOOTSTRAP_PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
DATASETS ?= ctu13 cic2018 synthetic

.PHONY: all setup data train evaluate eval benchmark test app ctu13 cic2018 synthetic clean-raw

all: data train evaluate eval benchmark test

setup:
	$(BOOTSTRAP_PYTHON) -m venv .venv && $(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

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
