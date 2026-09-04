"""
prepare_carbonate.py  —  Task 3, Path A (headline carbonate benchmark)
======================================================================
ONE runnable script. Turns the grayscale middle-resolution (~2-3 um) carbonate
volume into leakage-free train/val/test HDF5 pairs, using the SAME calibrated
realistic degradation as your Berea pipeline, with train-only augmentation so a
single carbonate sample yields enough data.

Output (matches FSRCNN / FSRCNN-RB / SRCNN training scripts):
    train.h5 / val.h5 / test.h5   each with 'HR' (N,128,128) and 'LR' (N,32,32)

USAGE
-----
1. Point FOLDER at the grayscale middle-res carbonate slice stack
   (the RECONSTRUCTED images, NOT the segmentation).
2. Re-calibrate BLUR_SIGMA for the carbonate scanner (see calibrate_* helpers),
   or keep the Berea q-factor and re-derive sigma for the carbonate voxel size.
3. Run.  Then train your three models on these H5s unchanged.

Notes
-----
- Grayscale in, grayscale out: SR learns an intensity mapping. Use the
  segmentation ONLY afterwards, for porosity/permeability.
- val/test are NOT augmented, so reported metrics stay honest.
"""

from __future__ import annotations
import time
import numpy as np
from scipy.ndimage import gaussian_filter, shift

# heavy / platform deps guarded so logic stays importable & testable
try:
    import h5py
except ImportError:
    h5py = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    from PIL import Image
    import glob
except ImportError:
    Image = None

# --------------------------------------------------------------------------- #
# CONFIG                                                                      #
# --------------------------------------------------------------------------- #
FOLDER  = "D:/Muturi D/SR/Carbonate_ILS_middle_2-3um/"   # grayscale middle-res stack
OUT_DIR = "D:/Muturi D/SR/Carbonate_ILS_middle_2-3um/"
EXT     = "*.tif"            # or "*.png"
PATCH, SCALE = 128, 4
RATIOS  = (0.70, 0.15, 0.15)   # train / val / test, BY SLICE
GUARD   = 2                    # slices skipped between blocks (z-decorrelation)
SEED    = 123
# Carbonate degradation params — RE-CALIBRATE these (see helpers below).
# Default = Berea values; replace once you measure the carbonate PSF/noise.
BLUR_SIGMA     = (2.84, 3.72)
DETECTOR_SIGMA = 0.02
POISSON_PEAK   = None
MAX_SHIFT      = 0.0            # 0 for synthetic pairs (exact correspondence)
ANISOTROPIC    = False
AUGMENT_TRAIN  = True


# --------------------------------------------------------------------------- #
# 1. Calibration helpers — derive degradation params FROM the carbonate data  #
# --------------------------------------------------------------------------- #
def estimate_noise_sigma(flat_patch: np.ndarray) -> float:
    """Detector-noise std from a single-phase (flat) region of the HR scan."""
    return float(np.std(flat_patch))


def estimate_psf_sigma(edge_profile: np.ndarray, dx: float = 1.0) -> float:
    """PSF sigma (px) from an edge-spread profile across a sharp grain boundary."""
    esf = np.asarray(edge_profile, float)
    lsf = np.abs(np.gradient(esf, dx))
    s = lsf.sum()
    if s <= 0:
        raise ValueError("Flat edge profile: cannot estimate PSF.")
    x = np.arange(len(lsf)) * dx
    mu = (x * lsf).sum() / s
    return float(np.sqrt(((x - mu) ** 2 * lsf).sum() / s))


def blur_for_scale(hr_psf_px: float, scale: int = SCALE, q=(0.8, 1.0)) -> tuple[float, float]:
    """
    Convert a measured HR PSF (in HR pixels) into the extra Gaussian sigma needed
    so the downsampled LR has sharpness q (in LR voxels). Same q-factor logic you
    used for Berea. Returns (sigma_low, sigma_high) in HR pixels.
        target_HR_sigma = q * scale  (an LR-voxel blur expressed in HR pixels)
        extra_sigma = sqrt(max(target^2 - hr_psf^2, eps))   (Gaussian convolution)
    """
    out = []
    for qi in q:
        target = qi * scale
        out.append(float(np.sqrt(max(target ** 2 - hr_psf_px ** 2, 1e-6))))
    return tuple(sorted(out))


# --------------------------------------------------------------------------- #
# 2. Calibrated degradation (identical to your Load_HR_LR.Degradation_Model)   #
# --------------------------------------------------------------------------- #
def degradation_model(hr, scale=SCALE, blur_sigma=BLUR_SIGMA, poisson_peak=POISSON_PEAK,
                      detector_sigma=DETECTOR_SIGMA, max_shift=MAX_SHIFT,
                      anisotropic=ANISOTROPIC, rng=None):
    rng = rng or np.random.default_rng()
    hr = np.clip(hr.astype(np.float64), 0, 1)
    sy = rng.uniform(*blur_sigma)
    sx = rng.uniform(*blur_sigma) if anisotropic else sy
    img = gaussian_filter(hr, sigma=(sy, sx), mode="reflect")
    if max_shift > 0:
        img = shift(img, (rng.uniform(-max_shift, max_shift),
                          rng.uniform(-max_shift, max_shift)), order=1, mode="reflect")
    H, W = img.shape
    H2, W2 = H // scale * scale, W // scale * scale
    img = img[:H2, :W2].reshape(H2 // scale, scale, W2 // scale, scale).mean(axis=(1, 3))
    if poisson_peak is not None:
        img = rng.poisson(np.clip(img, 0, 1) * poisson_peak) / poisson_peak
    if detector_sigma > 0:
        img = img + rng.normal(0, detector_sigma, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. IO, slice partition, patch extraction, augmentation                      #
# --------------------------------------------------------------------------- #
def load_stack(folder, ext=EXT):
    """Load a grayscale 2D slice stack into (n_slices, H, W) float32."""
    if Image is None:
        raise RuntimeError("PIL not available in this environment.")
    files = sorted(glob.glob1(folder, ext))
    if not files:
        raise FileNotFoundError(f"No {ext} files in {folder}")
    h, w = Image.open(folder + files[0]).size
    stack = np.zeros((len(files), w, h), np.float32)
    for i, f in enumerate(files):
        stack[i] = np.array(Image.open(folder + f), dtype=np.float32)
    print(f"loaded {len(files)} slices, {w}x{h}")
    return stack


def partition_slices(n, ratios=RATIOS, guard=GUARD):
    """Contiguous, guarded, disjoint slice blocks (prevents patch leakage)."""
    n_tr, n_va = int(ratios[0] * n), int(ratios[1] * n)
    return {
        "train": np.arange(0, n_tr - guard),
        "val":   np.arange(n_tr, n_tr + n_va - guard),
        "test":  np.arange(n_tr + n_va, n),
    }


def normalize(p):
    if cv2 is not None:
        return cv2.normalize(p, None, 0.0, 1.0, cv2.NORM_MINMAX)
    lo, hi = float(p.min()), float(p.max())
    return (p - lo) / (hi - lo + 1e-12)


def extract_pairs(stack, slice_idx, stride, rng):
    HR, LR = [], []
    for s in slice_idx:
        img = stack[s]; h, w = img.shape
        for y in range(0, h - PATCH + 1, stride):
            for x in range(0, w - PATCH + 1, stride):
                hr = normalize(img[y:y+PATCH, x:x+PATCH]).astype(np.float32)
                HR.append(hr)
                LR.append(degradation_model(hr, rng=rng))
    return np.asarray(HR, np.float32), np.asarray(LR, np.float32)


def augment_train(HR, LR):
    """8x dihedral augmentation (90/180/270 rot + vertical flip), HR & LR in lockstep."""
    aH, aL = [HR], [LR]
    for k in (1, 2, 3):
        aH.append(np.rot90(HR, k, axes=(1, 2)))
        aL.append(np.rot90(LR, k, axes=(1, 2)))
    H, L = np.concatenate(aH), np.concatenate(aL)
    return (np.concatenate([H, np.flip(H, 1)]),
            np.concatenate([L, np.flip(L, 1)]))


def write_h5(path, HR, LR):
    if h5py is None:
        print(f"[dry-run] would write {path}: HR{HR.shape} LR{LR.shape}")
        return
    with h5py.File(path, "w") as hf:
        hf.create_dataset("HR", data=HR, compression="gzip", compression_opts=4)
        hf.create_dataset("LR", data=LR, compression="gzip", compression_opts=4)


def main():
    t = time.time()
    stack = load_stack(FOLDER)
    rng = np.random.default_rng(SEED)
    parts = partition_slices(stack.shape[0])
    for name, idx in parts.items():
        stride = 64 if name == "train" else PATCH        # augment train via overlap + dihedral
        HR, LR = extract_pairs(stack, idx, stride, rng)
        if name == "train" and AUGMENT_TRAIN:
            HR, LR = augment_train(HR, LR)
        print(f"{name:5s}: slices={len(idx):4d}  pairs={HR.shape[0]:6d}")
        write_h5(f"{OUT_DIR}/{name}.h5", HR, LR)
    print("done in", round(time.time() - t, 1), "s")


# --------------------------------------------------------------------------- #
# Self-test (synthetic stack; no PIL/cv2/h5py needed)                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv or h5py is None:
        rng = np.random.default_rng(0)
        stack = np.stack([gaussian_filter(rng.random((400, 400)), 3) for _ in range(40)]).astype(np.float32)

        # calibration demo
        print("PSF->sigma for hr_psf=0.90px:", blur_for_scale(0.90))
        parts = partition_slices(stack.shape[0])
        allidx = np.concatenate(list(parts.values()))
        assert len(allidx) == len(set(allidx.tolist())), "slice leakage!"

        HR, LR = extract_pairs(stack, parts["train"], 64, rng)
        aHR, aLR = augment_train(HR, LR)
        print(f"train before aug: {HR.shape[0]}  after 8x aug: {aHR.shape[0]} "
              f"(={aHR.shape[0] // HR.shape[0]}x)")
        vHR, vLR = extract_pairs(stack, parts["val"], PATCH, rng)
        print(f"val (un-augmented): {vHR.shape[0]} pairs, HR{vHR.shape[1:]} LR{vLR.shape[1:]}")
        # alignment check: augmented HR/LR counts match
        assert aHR.shape[0] == aLR.shape[0]
        print("partition disjoint + HR/LR aligned + train-only aug: OK")
    else:
        main()
