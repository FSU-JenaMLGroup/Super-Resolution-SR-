"""
otsu_sensitivity.py  —  R2-5: is the porosity result an artefact of the threshold?
================================================================================
ONE press-Run script, no inference. Reads the CLEAN images the evaluator already
saved (HR_input/clean and SR_FSRCNN-RB/clean for each dataset), sweeps the
segmentation threshold around the Otsu value, and reports the mean absolute
porosity error |poro(SR) - poro(GT)| per threshold.

Claim being tested (manuscript, R2-5): the ordering "FSRCNN-RB porosity close to
GT" is preserved across the threshold range — i.e. not a binarization artefact.

Outputs (to OUT_DIR):
  figures/otsu_sensitivity.png    error-vs-threshold curves, one per dataset
  otsu_sensitivity.csv            threshold, per-dataset mean |porosity error|

NOTE on scope: the current evaluator saves reconstructions for FSRCNN-RB only.
To extend this sweep to FSRCNN for the Berea Figure-5/7 claims, re-enable the
FSRCNN baseline in the evaluator's MODELS block and re-run it first; this script
picks up any SR_<model>/clean folders it finds automatically.

HOW TO RUN: edit CONFIG, press Run.
"""
from __future__ import annotations
import csv, importlib.util
from pathlib import Path
import numpy as np
import tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ CONFIG (EDIT) =============================== #
EVAL_SCRIPT = r"D:\Muturi D\SR\ILS45\FSRCNN-RB\SS_ILS45_noise_test.py"  # your evaluator
OUT_DIR     = r"D:\Muturi D\SR\ILS45\Data"    # the evaluator's OUT_DIR (holds SS/ and ILS/)
DATASET_KEYS = ["SS", "ILS"]
REL_THRESHOLDS = np.linspace(0.80, 1.20, 9)    # sweep: Otsu x 0.80 ... Otsu x 1.20
# ========================================================================= #


def load_eval_module(path):
    spec = importlib.util.spec_from_file_location("evalmod", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_stack(folder):
    files = sorted(Path(folder).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No tiffs in {folder} — run the evaluator first.")
    return np.stack([tifffile.imread(str(f)).astype(np.float32) / 255.0 for f in files])


def main():
    mod = load_eval_module(EVAL_SCRIPT)
    out = Path(OUT_DIR); (out / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key in DATASET_KEYS:
        base = out / key
        if not (base / "HR_input" / "clean").is_dir():
            print(f"[skip] {key}: no saved outputs under {base}"); continue
        GT = load_stack(base / "HR_input" / "clean")
        # every SR_<model>/clean folder found is swept (FSRCNN-RB by default)
        sr_dirs = sorted(d for d in base.glob("SR_*") if (d / "clean").is_dir())
        if not sr_dirs:
            print(f"[skip] {key}: no SR_<model> folders"); continue
        thr0 = mod.otsu(GT)                       # dataset Otsu from GT (matches evaluator)
        print(f"{key}: Otsu={thr0:.4f}, {len(GT)} patches, models: "
              f"{[d.name[3:] for d in sr_dirs]}")
        for d in sr_dirs:
            model = d.name[3:]
            SR = load_stack(d / "clean")
            n = min(len(GT), len(SR))
            errs = []
            for rel in REL_THRESHOLDS:
                t = thr0 * rel
                e = [abs(float((SR[i] < t).mean()) - float((GT[i] < t).mean()))
                     for i in range(n)]
                errs.append(float(np.mean(e)))
                rows.append(dict(dataset=key, model=model, rel_threshold=round(rel, 3),
                                 threshold=round(t, 4),
                                 mean_abs_porosity_error=round(errs[-1], 5)))
            ax.plot(REL_THRESHOLDS, errs, "o-", label=f"{key} / {model}")
            print(f"  {model}: |poro err| {errs[0]:.4f} @0.8xOtsu ... "
                  f"{errs[len(errs)//2]:.4f} @Otsu ... {errs[-1]:.4f} @1.2xOtsu")
    ax.set_xlabel("Threshold relative to Otsu value")
    ax.set_ylabel("Mean |porosity(SR) - porosity(GT)|")
    ax.set_title("Segmentation-threshold sensitivity of the porosity error")
    ax.axvline(1.0, color="gray", lw=0.8, ls="--"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "otsu_sensitivity.png",
                                    dpi=300, facecolor="white"); plt.close(fig)
    with open(out / "otsu_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "rel_threshold",
                                          "threshold", "mean_abs_porosity_error"])
        w.writeheader(); w.writerows(rows)
    print(f"figure + CSV -> {out}")


if __name__ == "__main__":
    main()
