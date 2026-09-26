"""python -m sih_v2 {build_dataset,train,evaluate,benchmark} [...]"""
import sys

from .cli import benchmark, build_dataset, evaluate, train

CMDS = {"build_dataset": build_dataset.main, "train": train.main, "evaluate": evaluate.main,
        "benchmark": benchmark.main}

if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
    print(__doc__)
    sys.exit(1)
CMDS[sys.argv[1]](sys.argv[2:])
