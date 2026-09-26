"""Train the NetWorldModel (multi-seed) + every baseline, calibrate on validation, save bundles.

  python -m sih_v2 train --dataset ctu13 --protocol temporal
  python -m sih_v2 train --dataset ctu13 --protocol family --seeds 0 1 2
"""
from __future__ import annotations

import logging
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import resolve, seed_paths
from ..engine.bundle import Bundle, save_bundle
from ..engine.calibrate import choose_threshold, fit_ensemble, tune_fusion
from ..engine.simulate import ForecastEngine
from ..features.scaler import RobustScaler
from ..features.states import state_feature_names
from ..models.baselines import StageHorizonLR, make_baselines
from ..models.markov import KillChainMarkov
from ..models.trainer import direct_scores, selection_score, train_seq_classifier, train_world_model
from .common import base_parser, dump_json, setup

log = logging.getLogger("train")
STAGE_HORIZONS = [1, 2, 3, 5, 7, 10]


def arrays(df, scaler, feats):
    return {"X": scaler.transform(df[feats].to_numpy(np.float32)), "stage": df["stage"].to_numpy(np.int64),
            "y": df["y_future"].to_numpy(np.int64), "seg": df["segment"].to_numpy(),
            "feature_names": feats, "scale": scaler.scale}


def load_split(cfg):
    df = pd.read_parquet(resolve(cfg["paths"]["processed"]))
    col = f"split_{cfg['protocol']}"
    if col not in df:
        raise SystemExit(f"{cfg['dataset']} has no '{cfg['protocol']}' split (column {col} missing)")
    return {s: df[df[col] == s].reset_index(drop=True) for s in ("train", "val", "test")}


def main(argv=None):
    p = base_parser("Train the world model and baselines")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--skip-baselines", action="store_true", help="reuse existing sklearn baselines")
    p.add_argument("--calibrate-only", action="store_true",
                   help="refit validation-only ensemble weights/thresholds on existing seed bundles (no retraining)")
    args = p.parse_args(argv)
    cfg = setup(args)
    if args.calibrate_only:
        return calibrate_only(cfg, args.seeds if args.seeds is not None else cfg.get("seeds", [cfg["seed"]]))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [cfg["seed"]])
    t0 = time.time()
    sp = load_split(cfg)
    tr, va = sp["train"], sp["val"]
    feats = state_feature_names()
    scaler = RobustScaler().fit(tr[feats].to_numpy(np.float32))
    A_tr, A_va = arrays(tr, scaler, feats), arrays(va, scaler, feats)
    log.info("%s/%s: %d train / %d val cells, train positive rate %.4f", cfg["dataset"], cfg["protocol"],
             len(tr), len(va), A_tr["y"].mean())
    markov = KillChainMarkov().fit(A_tr["stage"], A_tr["seg"])
    max_fpr = cfg["engine"]["max_fpr"]
    bl_path = resolve(cfg["paths"]["baseline_bundle"])
    bl_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ sklearn baselines (seed-independent)
    if args.skip_baselines and bl_path.exists():
        pack = joblib.load(bl_path)
    else:
        bc = cfg["baselines"]
        pack = {"scaler": scaler.to_dict(), "features": feats, "sklearn": [], "lstm": {}}
        for b in make_baselines(bc["stack_windows"], cfg["seed"]):
            t = time.time()
            b.fit(A_tr["X"], A_tr["seg"], A_tr["y"], bc["max_train_rows"], cfg["seed"])
            thr = choose_threshold(b.score(A_va["X"], A_va["seg"]), A_va["y"], max_fpr)
            pack["sklearn"].append({"model": b, "threshold": thr})
            log.info("baseline %-40s thr %.3f (%.0fs)", b.name, thr, time.time() - t)
        t = time.time()
        pack["stage_lr"] = StageHorizonLR(cfg["engine"]["rollout_steps"], bc["stack_windows"]).fit(
            A_tr["X"], A_tr["seg"], A_tr["stage"], bc.get("stage_train_rows", 30000), cfg["seed"], STAGE_HORIZONS)
        log.info("stage LR per horizon %s (%.0fs)", STAGE_HORIZONS, time.time() - t)
        joblib.dump(pack, bl_path, compress=3)   # checkpoint: a later run can resume with --skip-baselines
    pack.setdefault("lstm", {})

    # ------------------------------------------------------------ deep models, one run per seed
    runs = []
    for seed in seeds:
        c = dict(cfg, seed=seed)
        model, hist = train_world_model(A_tr, A_va, c, seed=seed)
        b = Bundle(model, scaler, markov, feats, {"fused": 0.5}, cfg,
                   {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed, "dataset": cfg["dataset"],
                    "protocol": cfg["protocol"], "train_cells": len(tr), "val_cells": len(va),
                    "epochs_run": len(hist), "history": hist})
        res = ForecastEngine(b, seed=seed).run(va).frame
        if cfg["engine"].get("tune_fusion", True):
            w, _ = tune_fusion(res)
            if w is not None:
                b.config = dict(b.config, engine=dict(b.config["engine"], fusion=w))
                res["risk"] = sum(w[k] * res[f"risk_{k}"] for k in w) / sum(w.values())
        y = va["y_future"].to_numpy()
        b.thresholds = {k: choose_threshold(res["risk"] if k == "fused" else res[f"risk_{k}"], y, max_fpr)
                        for k in ("direct", "rollout", "markov", "fused")}
        mal = (va["stage"] > 0).to_numpy()
        auc = roc_auc_score(mal, res["surprise_obs"]) if 0 < mal.mean() < 1 else 0.5
        b.meta.update(surprise_sign=1.0 if auc >= 0.5 else -1.0, surprise_val_auc=float(auc),
                      val_select=selection_score(y, res["risk"].to_numpy(), va["stage"].to_numpy(),
                                                 cfg["train"].get("select_metric", "mix")))
        b.meta["ensemble"] = fit_ensemble(res, ForecastEngine(b).scale(va), va["segment"].to_numpy(), pack, max_fpr,
                                          cfg["train"].get("select_metric", "mix"))
        sp_ = seed_paths(cfg, seed)
        save_bundle(sp_["model_bundle"], b)
        log.info("seed %d: fusion %s thresholds %s val-select %.4f surprise AUC %.3f (sign %+d)", seed,
                 b.config["engine"]["fusion"], {k: round(v, 3) for k, v in b.thresholds.items()},
                 b.meta["val_select"], auc, b.meta["surprise_sign"])
        runs.append((b.meta["val_select"], seed, sp_["model_bundle"]))

        lstm, lh = train_seq_classifier(A_tr, A_va, c, seed=seed)
        s_va = direct_scores(lstm, A_va["X"], A_va["seg"], cfg["forecast"]["context"])
        pack["lstm"][seed] = {"hparams": lstm.hparams, "state_dict": {k: v.cpu() for k, v in lstm.state_dict().items()},
                              "threshold": choose_threshold(s_va, y, max_fpr), "epochs_run": len(lh)}

    best = max(runs)
    shutil.copyfile(best[2], resolve(cfg["paths"]["model_bundle"]))
    pack["primary_seed"] = best[1]
    joblib.dump(pack, bl_path, compress=3)
    dump_json({"seeds": seeds, "primary_seed": best[1], "val_select": {s: v for v, s, _ in runs},
               "markov_P": markov.P, "elapsed_sec": time.time() - t0,
               "baseline_thresholds": {e["model"].name: e["threshold"] for e in pack["sklearn"]}},
              resolve(cfg["paths"]["reports_dir"]) / "train_summary.json")
    log.info("primary bundle = seed %d -> %s (%.0fs)", best[1], cfg["paths"]["model_bundle"], time.time() - t0)


def calibrate_only(cfg, seeds):
    from ..engine.bundle import load_bundle

    va = load_split(cfg)["val"]
    pack = joblib.load(resolve(cfg["paths"]["baseline_bundle"]))
    primary = pack.get("primary_seed")
    for seed in seeds:
        p = seed_paths(cfg, seed)["model_bundle"]
        if not p.exists():
            continue
        b = load_bundle(p)
        res = ForecastEngine(b, seed=seed).run(va).frame
        b.meta["ensemble"] = fit_ensemble(res, ForecastEngine(b).scale(va), va["segment"].to_numpy(), pack,
                                          cfg["engine"]["max_fpr"], cfg["train"].get("select_metric", "mix"))
        save_bundle(p, b)
        log.info("seed %d ensemble: %s", seed, b.meta["ensemble"])
        if seed == primary:
            shutil.copyfile(p, resolve(cfg["paths"]["model_bundle"]))


if __name__ == "__main__":
    main()
