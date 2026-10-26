"""
SS_ILS45_noise_test.py  —  ONE press-Run script. No retraining.
================================================================================
DUAL-DATASET noise-robustness test for FSRCNN-RB.  Extended, byte-for-byte where
possible, from your debugged SS_ILS45_noise_test.py.  What changed vs. that file:

  * Runs the SAME pipeline on TWO datasets and compares them:
        SS  = Berea sandstone   (SS-trained FSRCNN-RB tested on SS)
        ILS = ILS carbonate     (ILS-trained FSRCNN-RB tested on ILS)
  * Samples are drawn from HELD-OUT slices only (slices that contributed ZERO
    patches to the 20 000-patch training H5) -> no train/test leakage on either
    side. See heldout_slice_files() for the exact, defensible definition.
  * Only FSRCNN-RB is evaluated (FSRCNN / SRCNN are kept but commented out, so
    baselines can be switched back on without a rewrite).
  * Reconstruction figure is ONE figure with TWO stacked panels:
        (a) SS Berea      (b) ILS Carbonate
    each laid out as   rows = [HR Input, LR Input, Reconstructed]
                       cols = [HR clean, Noise1..Noise5]      (transposed vs. old grid)
  * Metrics: SSIM, PSNR, and LPIPS (perceptual). Three comparison plots, each
    showing SS vs. ILS for FSRCNN-RB.
  * Sanity checks run per dataset (MODE="sanity"): reproduces the A1_storedLR vs
    A2_regenLR H5 check for SS (HR_Q01/LR_Q01) AND ILS (HR_ILS/LR_ILS).

HOW TO RUN: edit the CONFIG box, set MODE, press Run.
"""
from __future__ import annotations
import os, csv, importlib.util, shutil, random
from pathlib import Path
import numpy as np
import cv2
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
import tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py


# ============================ CONFIG (EDIT) =============================== #
MODE   = "eval"        # "eval" = dual-dataset robustness test ; "sanity" = H5 A1/A2 check
#MODE   = "sanity"
OUT_DIR = r"D:\Muturi D\SR\ILS45\Data"     # all outputs written here (both datasets)

N_IMAGES            = 100          # held-out source images per dataset (task: 100 samples)
N_PATCHES_PER_IMAGE = 10           # crops per image -> 1000 patches/dataset for metrics
N_GEODICT   = 100                  # images exported for porosity/permeability per dataset
PATCH       = 128                  # HR patch size (model outputs 128)
SCALE       = 4                    # FSRCNN-RB is fixed at x4
HR_VOXEL_UM = 9.98                 # reference only
GRID_EXAMPLE = 0                   # which patch index appears in the reconstruction panels
SEED        = 123
TRAIN_H5_PATCHES = {"SS": 20000, "ILS": 13224}  # per-dataset NUM_PATCHES used by
                                    # Load_*_H5*.py -> defines held-out cut

# -------- the two datasets. FSRCNN-RB weights differ (SS vs ILS trained) -------- #
# raw_folder : slices to sample held-out patches from (same sort order as the H5 builder)
# weights    : trained FSRCNN-RB best.pth for THAT dataset
# h5_hr/h5_lr: stored training pairs, used ONLY by the sanity check (MODE="sanity")
DATASETS = {
    "SS": dict(
        label      = "SS Berea",
        raw_folder = r"D:\Muturi D\SR\SS\Dataset",
        weights    = r"D:\Muturi D\SR\SS\FSRCNN-RB\output\best.pth",       # SS-trained
        h5_hr      = r"D:\Muturi D\SR\SS\FSRCNN\HR_Q01.h5",
        h5_lr      = r"D:\Muturi D\SR\SS\FSRCNN\LR_Q01.h5",
    ),
    "ILS": dict(
        label      = "ILS Carbonate",
        raw_folder = r"D:\Muturi D\SR\ILS45\ILS_Dataset",
        weights    = r"D:\Muturi D\SR\ILS45\Data\ILS\FSRCNN-RB\output\best.pth",   # ILS-trained (from Load_*_H5_ILS + retrain)
        h5_hr      = r"D:\Muturi D\SR\ILS45\Data\HR_ILS.h5",
        h5_lr      = r"D:\Muturi D\SR\ILS45\Data\LR_ILS.h5",
    ),
}

# Only FSRCNN-RB is evaluated. Folder holds models.py + the class name + weights is
# filled per dataset from DATASETS[...] at run time.
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME  = "FSRCNN-RB"
MODEL_CLASS = "FSRCNN_RB"
# --- baselines kept for optional re-enable (NOT evaluated by default) ---
# BASELINES = {
#     "FSRCNN": (r"D:\Muturi D\SR\FSRCNN", "FSRCNN"),
#     "SRCNN":  (r"D:\Muturi D\SR\SRCNN",  "SRCNN"),
# }

# 6 levels (0 = clean). poisson_peak lower = noisier.  (verbatim from your file)
NOISE_LEVELS = {
    0: dict(poisson_peak=None, detector_sigma=0.000),
    1: dict(poisson_peak=2000, detector_sigma=0.005),
    2: dict(poisson_peak=1000, detector_sigma=0.010),
    3: dict(poisson_peak=500,  detector_sigma=0.020),
    4: dict(poisson_peak=160,  detector_sigma=0.1),
    5: dict(poisson_peak=80,   detector_sigma=0.2),
}
COL_LABELS  = {0: "HR clean", 1: "Noise 1", 2: "Noise 2",
               3: "Noise 3", 4: "Noise 4", 5: "Noise 5"}     # columns of the new panels
LEVEL_NAMES = {0: "clean", 1: "noise1", 2: "noise2", 3: "noise3", 4: "noise4", 5: "noise5"}
PANEL_ROWS  = ["HR Input", "LR Input", "Reconstructed"]      # rows of the new panels
# ========================================================================= #


# --------------------------- LPIPS (optional) ---------------------------- #
# Perceptual metric (Zhang et al., 2018). LOWER = better (opposite of SSIM/PSNR).
# Graceful: if the package/weights are unavailable the run continues and LPIPS
# columns are reported as NaN rather than crashing.
try:
    import lpips as _lpips_lib
    _LPIPS_NET = None
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False


def _get_lpips():
    global _LPIPS_NET
    if _LPIPS_NET is None:
        _LPIPS_NET = _lpips_lib.LPIPS(net="alex"); _LPIPS_NET.eval()
    return _LPIPS_NET


def lpips_score(sr, hr):
    """LPIPS between two [0,1] grayscale patches. Maps to [-1,1], replicates to
    3 channels (LPIPS was trained on natural RGB -> treat as a rough proxy on
    micro-CT). Returns NaN if LPIPS is unavailable."""
    if not LPIPS_OK:
        return float("nan")
    import torch
    sr, hr = np.clip(sr, 0, 1), np.clip(hr, 0, 1)
    h, w = min(sr.shape[0], hr.shape[0]), min(sr.shape[1], hr.shape[1])
    sr, hr = sr[:h, :w], hr[:h, :w]

    def to_t(a):
        a = a.astype(np.float32) * 2.0 - 1.0                # [0,1] -> [-1,1]
        return torch.from_numpy(a)[None, None].repeat(1, 3, 1, 1)
    with torch.no_grad():
        return float(_get_lpips()(to_t(sr), to_t(hr)).item())


# ----------------------------- helpers ----------------------------------- #
# (all verbatim from ils_noise_test.py unless a comment says otherwise)
def _open_2d_gray(path):
    from PIL import Image
    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        vol = tifffile.imread(str(path))
        if vol.ndim == 3:
            vol = vol[vol.shape[0] // 2]
        return vol.astype(np.float32)
    return np.asarray(Image.open(path).convert("L"), np.float32)


def _norm_minmax(p):
    """EXACT training normalization (per patch, to [0,1])."""
    return cv2.normalize(p.astype(np.float64), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)


def heldout_slice_files(folder, patch, train_patches, exts):
    """Return the sorted slice files that contributed ZERO patches to the training
    H5, i.e. the leakage-free HELD-OUT pool.

    The H5 builder (Load_*_H5*.py) walks slices in sorted() order and tiles each
    into (H//patch)*(W//patch) non-overlapping patches until it reaches
    train_patches (20 000). Therefore the first `consumed` sorted slices supplied
    every train/val/test patch; slices at index >= `consumed` were never touched.
    Sampling only from those is provably free of train/test overlap.
    """
    files = sorted([f for f in Path(folder).iterdir() if f.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    im0 = _open_2d_gray(files[0])
    pps = max(1, (im0.shape[0] // patch) * (im0.shape[1] // patch))   # tiled patches / slice
    consumed = int(np.ceil(train_patches / pps))
    heldout = files[consumed:]
    print(f"    slices: {len(files)} total, ~{pps} patches/slice, "
          f"{consumed} consumed by training H5, {len(heldout)} held-out")
    if len(heldout) < 1:
        raise RuntimeError(
            f"No held-out slices in {folder} (needs > {consumed}). "
            f"Volume too small for a leakage-free split at train_patches={train_patches}.")
    return heldout


def load_heldout_patches(folder, n_images, n_patches, patch, train_patches, seed):
    """Draw n_patches random crops from each of n_images HELD-OUT slices.
    Same per-patch NORM_MINMAX and same std>1e-3 texture filter as training."""
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    heldout = heldout_slice_files(folder, patch, train_patches, exts)
    rng = np.random.default_rng(seed)
    take = min(n_images, len(heldout))
    if take < n_images:
        print(f"    [warn] only {take} held-out slices available (< {n_images} requested)")
    sel = rng.choice(len(heldout), size=take, replace=False)
    out = []
    for file_idx in sel:
        im = _open_2d_gray(heldout[int(file_idx)])
        if min(im.shape) < patch:
            continue
        got = 0
        for _ in range(n_patches * 50):
            if got >= n_patches:
                break
            y = rng.integers(0, im.shape[0] - patch + 1)
            x = rng.integers(0, im.shape[1] - patch + 1)
            p = im[y:y + patch, x:x + patch]
            if p.std() < 1e-3:
                continue
            out.append(_norm_minmax(p)); got += 1
    if not out:
        raise RuntimeError("No patches collected from held-out slices. Check folder / PATCH.")
    return np.stack(out)


def add_noise(img, level, rng):
    q = NOISE_LEVELS[level]; out = np.clip(img, 0, 1).astype(np.float64)
    if q["poisson_peak"]:
        out = rng.poisson(out * q["poisson_peak"]) / q["poisson_peak"]
    if q["detector_sigma"]:
        out = out + rng.normal(0, q["detector_sigma"], out.shape)
    return np.clip(out, 0, 1).astype(np.float32)


def downscale(img, scale=SCALE, psf=None):
    """LR = two successive cv2.pyrDown (128->64->32). Matches training exactly."""
    tmp = cv2.pyrDown(img.astype(np.float64))
    lr = cv2.pyrDown(tmp)
    return np.clip(lr, 0.0, 1.0).astype(np.float32)


def metrics(sr, hr):
    """Reconstruction vs CLEAN HR. Returns (SSIM, PSNR)."""
    sr, hr = np.clip(sr, 0, 1), np.clip(hr, 0, 1)
    h, w = min(sr.shape[0], hr.shape[0]), min(sr.shape[1], hr.shape[1])
    sr, hr = sr[:h, :w], hr[:h, :w]
    return float(sk_ssim(hr, sr, data_range=1.0)), float(sk_psnr(hr, sr, data_range=1.0))


def save_tiff(path, arr):
    arr = arr if arr.dtype == np.uint8 else (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    tifffile.imwrite(str(path), arr)


def otsu(v, bins=256):
    h, e = np.histogram(v.ravel(), bins=bins, range=(0, 1)); p = h / h.sum()
    om = np.cumsum(p); mu = np.cumsum(p * (e[:-1] + np.diff(e) / 2)); mt = mu[-1]
    d = om * (1 - om); d[d == 0] = np.nan
    return float(e[:-1][np.nanargmax((mt * om - mu) ** 2 / d)] + np.diff(e)[0] / 2)


def _up(img, target):
    z = max(1, target // img.shape[0])
    return np.kron(img, np.ones((z, z))) if z > 1 else img


def load_model(folder, cls_name, weights):
    import torch
    spec = importlib.util.spec_from_file_location("m", os.path.join(folder, "models.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    cls = getattr(m, cls_name)
    try:
        net = cls(scale_factor=SCALE)
    except TypeError:
        net = cls()
    st = torch.load(weights, map_location="cpu")
    st = st.get("model_state_dict", st) if isinstance(st, dict) else st
    net.load_state_dict(st); net.eval()
    return net


def reconstruct(net, lr_patch, name=MODEL_NAME):
    """Run one LR patch through the net. SRCNN gets a bicubic pre-upsample; the
    FSRCNN family takes the raw 32x32 LR. (SRCNN branch only fires if baselines
    are re-enabled.)"""
    import torch
    lr = torch.from_numpy(lr_patch[None, None].astype(np.float32))
    x = (F.interpolate(lr, scale_factor=SCALE, mode="bicubic", align_corners=True)
         if name == "SRCNN" else lr)
    with torch.no_grad():
        return net(x).clamp(0, 1)[0, 0].numpy()


# ------------------------- H5 sanity (per dataset) ----------------------- #
def _read_h5_array(path):
    with h5py.File(path, "r") as hf:
        keys = list(hf.keys())
        for pref in ("3D_image", "HR", "LR", "data", "images"):
            if pref in keys:
                arr = np.array(hf[pref], dtype=np.float32); break
        else:
            arr = np.array(hf[keys[0]], dtype=np.float32)
    if arr.ndim == 2:
        side = int(round(arr.shape[1] ** 0.5))
        if side * side == arr.shape[1]:
            arr = arr.reshape(-1, side, side)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def run_sanity(cfg, n=3000):
    """MODE B (per dataset): reproduce the in-domain SSIM on the STORED test pairs.
    A1 = model(stored LR)   -> pipeline / weights / metric correct.
    A2 = model(LR regenerated from stored HR via downscale()) -> operator matches.
    A1 == A2 (as you already saw for SS: 0.9495) proves the degradation operator
    is identical to training."""
    import torch
    if not (Path(cfg["h5_hr"]).exists() and Path(cfg["h5_lr"]).exists()):
        print(f"[skip sanity] {cfg['label']}: H5 not found "
              f"({cfg['h5_hr']} / {cfg['h5_lr']})"); return []
    if not Path(cfg["weights"]).exists():
        print(f"[skip sanity] {cfg['label']}: weights not found {cfg['weights']}"); return []
    HR = _read_h5_array(cfg["h5_hr"]); LR = _read_h5_array(cfg["h5_lr"])
    m = min(len(HR), len(LR), n); HR, LR = HR[:m], LR[:m]
    print(f"  {cfg['label']} H5: HR{HR.shape} LR{LR.shape}")
    up = cv2.resize(LR[0], HR[0].shape[::-1], interpolation=cv2.INTER_CUBIC)
    print(f"    bicubic(stored LR) vs HR: SSIM={metrics(up, HR[0])[0]:.3f}")
    net = load_model(MODEL_DIR, MODEL_CLASS, cfg["weights"])
    psum = float(sum(p.abs().sum() for p in net.parameters()))
    print(f"    {MODEL_NAME}: param-sum {psum:.1f} (nonzero => weights loaded)")
    A2LR = np.stack([downscale(h) for h in HR])
    rows = []
    for variant, src in (("A1_storedLR", LR), ("A2_regenLR", A2LR)):
        ss, ps = [], []
        for i in range(len(HR)):
            sr = reconstruct(net, src[i]); s, p = metrics(sr, HR[i]); ss.append(s); ps.append(p)
        rows.append(dict(dataset=cfg["label"], variant=variant, n=len(HR),
                         ssim=float(np.mean(ss)), psnr=float(np.mean(ps))))
        print(f"    {variant}: SSIM={np.mean(ss):.4f}  PSNR={np.mean(ps):.2f} dB")
    return rows


# ------------------------- per-dataset evaluation ------------------------ #
def run_dataset(key, cfg, out_root):
    """Full robustness run for ONE dataset. Held-out patches -> 6 noise HR ->
    pyrDown LR -> FSRCNN-RB reconstruction -> metrics vs clean HR. Returns the
    metrics rows and the single-patch example needed for the combined figure."""
    print(f"\n=== {cfg['label']} ({key}) ===")
    out = out_root / key
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    levels = sorted(NOISE_LEVELS)

    HR = load_heldout_patches(cfg["raw_folder"], N_IMAGES, N_PATCHES_PER_IMAGE,
                              PATCH, TRAIN_H5_PATCHES[key], SEED)
    print(f"  loaded {len(HR)} held-out HR patches {HR.shape[1:]}")

    # noisy HR per level, and LR generated FROM that noisy HR
    HRn = {lvl: (HR.copy() if lvl == 0 else np.stack([add_noise(h, lvl, rng) for h in HR]))
           for lvl in levels}
    LR = {lvl: np.stack([downscale(h) for h in HRn[lvl]]) for lvl in levels}

    for lvl in levels:
        for kind, data in (("HR_input", HRn[lvl]), ("LR_input", LR[lvl])):
            d = out / kind / LEVEL_NAMES[lvl]; d.mkdir(parents=True, exist_ok=True)
            for i in range(len(HR)):
                save_tiff(d / f"{i:03d}.tif", data[i])

    export_geodict(key, cfg, out)     # per-dataset porosity export (held-out)

    if not Path(cfg["weights"]).exists():
        print(f"  [skip recon] {cfg['label']}: FSRCNN-RB weights not found at "
              f"{cfg['weights']}  (train the model first).")
        g = GRID_EXAMPLE
        example = dict(HR={l: HRn[l][g] for l in levels},
                       LR={l: LR[l][g] for l in levels}, SR=None)
        return [], example

    net = load_model(MODEL_DIR, MODEL_CLASS, cfg["weights"])
    results, SR = [], {}
    for lvl in levels:
        ss, ps, lp, srs = [], [], [], []
        for i in range(len(HR)):
            sr = reconstruct(net, LR[lvl][i])
            s, p = metrics(sr, HR[i])                 # vs CLEAN HR
            ss.append(s); ps.append(p); lp.append(lpips_score(sr, HR[i])); srs.append(sr)
        SR[lvl] = np.stack(srs)
        results.append(dict(
            dataset=cfg["label"], level=lvl, name=LEVEL_NAMES[lvl],
            ssim=float(np.mean(ss)), ssim_std=float(np.std(ss)),
            psnr=float(np.mean(ps)), psnr_std=float(np.std(ps)),
            lpips=float(np.nanmean(lp)), lpips_std=float(np.nanstd(lp))))
        print(f"  {cfg['label']} {LEVEL_NAMES[lvl]}: "
              f"SSIM={np.mean(ss):.4f}  PSNR={np.mean(ps):.2f}  LPIPS={np.nanmean(lp):.4f}")
        d = out / f"SR_{MODEL_NAME}" / LEVEL_NAMES[lvl]; d.mkdir(parents=True, exist_ok=True)
        for i in range(len(HR)):
            save_tiff(d / f"{i:03d}.tif", SR[lvl][i])

    g = GRID_EXAMPLE
    example = dict(HR={l: HRn[l][g] for l in levels},
                   LR={l: LR[l][g] for l in levels},
                   SR={l: SR[l][g] for l in levels})
    return results, example


def export_geodict(key, cfg, out):
    """Porosity export (Otsu) from held-out patches — one set per dataset."""
    g = out / "geodict"; (g / "gray").mkdir(parents=True, exist_ok=True); (g / "binary").mkdir(exist_ok=True)
    HR = load_heldout_patches(cfg["raw_folder"], N_GEODICT, 1, PATCH, TRAIN_H5_PATCHES[key], SEED + 1)
    thr = otsu(HR); rows = []
    for i in range(len(HR)):
        save_tiff(g / "gray" / f"{i:03d}.tif", HR[i])
        save_tiff(g / "binary" / f"{i:03d}.tif", (HR[i] < thr).astype(np.uint8) * 255)
        rows.append(dict(idx=i, porosity=round(float((HR[i] < thr).mean()), 4)))
    with open(g / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "porosity"]); w.writeheader(); w.writerows(rows)
    print(f"  GeoDict: {len(rows)} held-out images -> {g}")


# ----------------------------- figures ----------------------------------- #
def _panel_into(fig, gs_rows, example, levels, panel_tag, disp):
    """Draw one 3x6 panel (rows=HR/LR/Recon, cols=clean..noise5) into pre-made axes."""
    rows_data = [
        [example["HR"][l] for l in levels],
        [example["LR"][l] for l in levels],
        ([example["SR"][l] for l in levels] if example["SR"] is not None
         else [np.zeros((PATCH, PATCH), np.float32) for _ in levels]),
    ]
    for r in range(3):
        for c, lvl in enumerate(levels):
            ax = gs_rows[r][c]
            ax.imshow(_up(rows_data[r][c], disp), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(COL_LABELS[lvl], fontsize=10, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(PANEL_ROWS[r], fontsize=10, rotation=90, labelpad=8)
    if example["SR"] is None:
        gs_rows[2][0].text(0.02, 0.5, "(model not trained yet)", color="red",
                           fontsize=9, transform=gs_rows[2][0].transAxes)


def combined_figure(examples, levels, out_png):
    """ONE figure, TWO stacked panels: (a) SS, (b) ILS. 6 rows x 6 cols total."""
    order = [k for k in ("SS", "ILS") if k in examples]
    n_panels = len(order)
    disp = PATCH
    fig, axes = plt.subplots(3 * n_panels, len(levels),
                             figsize=(2.4 * len(levels), 2.4 * 3 * n_panels))
    axes = np.atleast_2d(axes)
    tags = {"SS": "(a) SS Berea", "ILS": "(b) ILS Carbonate"}
    for pi, key in enumerate(order):
        rows = [axes[pi * 3 + r] for r in range(3)]
        _panel_into(fig, rows, examples[key], levels, key, disp)
        # panel label on the left of the top row of each panel
        rows[0][0].annotate(tags[key], xy=(0, 1.35), xycoords="axes fraction",
                            fontsize=13, fontweight="bold", ha="left", va="bottom")
    fig.suptitle(f"{MODEL_NAME}: reconstruction under increasing input noise",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=300, facecolor="white"); plt.close(fig)
    print(f"combined reconstruction figure -> {out_png}")


def comparison_plot(all_metrics, field, ylabel, title, out_png, lower_better=False):
    """One metric (SSIM / PSNR / LPIPS) vs noise level, SS vs ILS overlaid.
    Keeps the existing plotting style (o- markers, xticks=levels, legend)."""
    levels = sorted(NOISE_LEVELS)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    any_data = False
    for label in sorted({r["dataset"] for r in all_metrics}):
        rs = sorted([r for r in all_metrics if r["dataset"] == label], key=lambda r: r["level"])
        if not rs or all(np.isnan(r[field]) for r in rs):
            continue
        ax.plot([r["level"] for r in rs], [r[field] for r in rs], "o-", label=label)
        any_data = True
    ax.set_xlabel("Noise level (0 = clean .. 5 = severe)"); ax.set_ylabel(ylabel)
    ax.set_xticks(levels); ax.set_title(title)
    if lower_better:
        ax.invert_yaxis()
        ax.text(0.98, 0.02, "lower = better", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, style="italic", color="gray")
    if any_data:
        ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=300, facecolor="white"); plt.close(fig)
    print(f"{field} comparison -> {out_png}")


# ------------------------------- run ------------------------------------- #
def run_eval():
    out_root = Path(OUT_DIR); (out_root / "figures").mkdir(parents=True, exist_ok=True)
    levels = sorted(NOISE_LEVELS)

    try:
        import torch  # noqa
    except ImportError:
        print("torch not found -> cannot run reconstruction. Install torch."); return

    all_metrics, examples = [], {}
    for key, cfg in DATASETS.items():
        try:
            rows, example = run_dataset(key, cfg, out_root)
        except RuntimeError as e:
            print(f"  [skip dataset] {cfg['label']}: {e}")
            continue
        all_metrics.extend(rows)
        if example is not None:
            examples[key] = example

    # metrics CSV (mean +/- std; correlated crops -> treat N as ~N_IMAGES, not N patches)
    if all_metrics:
        cols = ["dataset", "level", "name", "ssim", "ssim_std",
                "psnr", "psnr_std", "lpips", "lpips_std"]
        with open(out_root / "metrics_by_noise.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(all_metrics)
        print(f"\nmetrics -> {out_root / 'metrics_by_noise.csv'}")

        fdir = out_root / "figures"
        comparison_plot(all_metrics, "ssim",  "SSIM (vs clean HR)",
                        "SSIM vs input noise", fdir / "04_ssim_vs_noise.png")
        comparison_plot(all_metrics, "psnr",  "PSNR (dB)",
                        "PSNR vs input noise", fdir / "05_psnr_vs_noise.png")
        comparison_plot(all_metrics, "lpips", "LPIPS (perceptual)",
                        "LPIPS vs input noise", fdir / "06_lpips_vs_noise.png",
                        lower_better=True)

    # ONE combined reconstruction figure (a) SS  (b) ILS
    if examples:
        combined_figure(examples, levels, out_root / "figures" / "02_reconstruction_panels.png")

    if not LPIPS_OK:
        print("[note] lpips not installed -> LPIPS reported as NaN. "
              "`pip install lpips` (downloads AlexNet weights on first use).")


def run_all_sanity():
    print("===== SANITY (per dataset: A1 stored-LR vs A2 regen-LR) =====")
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, cfg in DATASETS.items():
        rows.extend(run_sanity(cfg, n=3000))
    if rows:
        with open(out / "sanity_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "variant", "n", "ssim", "psnr"])
            w.writeheader(); w.writerows(rows)
        print(f"sanity metrics -> {out / 'sanity_metrics.csv'}")


if __name__ == "__main__":
    print(f"===== MODE: {MODE} =====")
    run_all_sanity() if MODE == "sanity" else run_eval()