from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from ..config import load_config, resolve, seed_everything


def use_utf8_stdio() -> None:
    """Force UTF-8 on stdout/stderr.

    Our reports are full of non-ASCII typography (`≤`, `±`, `⊕`, `→`, `·`) and
    Windows consoles still default to a legacy code page (cp1252), where
    `print()` raises UnicodeEncodeError mid-report. `errors="replace"` keeps a
    mangled glyph from ever taking down a multi-hour run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # already utf-8, detached, or not a TextIOWrapper


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=None, help="path to config.yaml (default: project config)")
    p.add_argument("--dataset", default=None, choices=["ctu13", "cic2018", "synthetic"],
                   help="dataset (default: config 'dataset')")
    p.add_argument("--protocol", default=None, choices=["temporal", "family"],
                   help="evaluation protocol (default: config 'protocol')")
    return p


def setup(args):
    use_utf8_stdio()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, getattr(args, "dataset", None), getattr(args, "protocol", None))
    seed_everything(cfg["seed"])
    return cfg


def dump_json(obj, path: Path):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")
    return path
