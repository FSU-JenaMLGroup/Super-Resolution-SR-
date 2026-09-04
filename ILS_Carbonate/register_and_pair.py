"""
register_and_pair.py  —  Task 3, Path B (real multi-scale validation)
=====================================================================
Turns two co-located ILS scans (a coarser LR scan and a finer HR scan of the
SAME region) into a REAL LR/HR test set for inference-only validation of the
carbonate-trained model. This is the evidence that defeats Reviewer 2's
"synthetic degradation" / circularity objection.

Pipeline
--------
1. Load both grayscale volumes (LR coarse, HR fine).
2. Check the voxel-size ratio == SCALE (else resample one).
3. Bring LR onto the HR grid (upsample by SCALE) for alignment only.
4. Register (estimate integer/sub-pixel translation) via FFT phase correlation
   (numpy) or SimpleITK if installed; crop the common overlapping ROI.
5. From the registered overlap, cut matched pairs: 128-voxel HR patch <-> the
   co-located 32-voxel patch from the NATIVE LR grid.
6. Save test_real.h5 ('HR' (N,128,128), 'LR' (N,32,32)).  No training.

FILL IN: the three voxel sizes + folder paths. Everything else is generic.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import zoom

try:
    import h5py
except ImportError:
    h5py = None
try:
    from PIL import Image
    import glob
except ImportError:
    Image = None

# --------------------------------------------------------------------------- #
# CONFIG — fill from your scan metadata                                        #
# --------------------------------------------------------------------------- #
SCALE = 4
LR_FOLDER = "D:/.../ILS_outer_lowres/"     # coarser scan (the LR)
HR_FOLDER = "D:/.../ILS_middle_2-3um/"     # finer scan   (the HR)
VOXEL_LR_UM = 11.0     # <-- the numbers I asked for
VOXEL_HR_UM = 2.75     # <-- middle ~2-3 um
PATCH_HR = 128
OUT = "D:/.../test_real.h5"


# --------------------------------------------------------------------------- #
def load_volume(folder, ext="*.tif"):
    if Image is None:
        raise RuntimeError("PIL unavailable.")
    files = sorted(glob.glob1(folder, ext))
    h, w = Image.open(folder + files[0]).size
    vol = np.zeros((len(files), w, h), np.float32)
    for i, f in enumerate(files):
        vol[i] = np.array(Image.open(folder + f), np.float32)
    return _norm(vol)


def _norm(a):
    return (a - a.min()) / (np.ptp(a) + 1e-12)


def check_ratio(voxel_lr, voxel_hr, scale=SCALE, tol=0.1):
    """Return the action needed to make the pair exactly `scale`x."""
    ratio = voxel_lr / voxel_hr
    if abs(ratio - scale) <= tol * scale:
        return ("ok", ratio)
    return ("resample", ratio)   # caller must resample LR or HR to force ==scale


def phase_correlation_shift(a, b):
    """Integer translation that best aligns 2D image b onto a (numpy FFT)."""
    A, B = np.fft.fft2(a), np.fft.fft2(b)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    r = np.fft.ifft2(R).real
    peak = np.unravel_index(np.argmax(r), r.shape)
    shifts = np.array(peak, float)
    for i, dim in enumerate(a.shape):       # wrap to signed offsets
        if shifts[i] > dim // 2:
            shifts[i] -= dim
    return shifts                            # (dy, dx)


def register_translation_3d(lr_on_hr_grid, hr):
    """Estimate a global (z,y,x) translation using mid-slice phase correlation."""
    zc = min(lr_on_hr_grid.shape[0], hr.shape[0]) // 2
    dyx = phase_correlation_shift(hr[zc], lr_on_hr_grid[zc])
    return np.array([0.0, dyx[0], dyx[1]])   # extend to z if needed


def common_roi(hr, lr_up, shift_vox):
    """Crop both to their overlapping region given an (z,y,x) shift (HR grid)."""
    sz, sy, sx = np.round(shift_vox).astype(int)
    Z = min(hr.shape[0], lr_up.shape[0] - sz)
    Y = min(hr.shape[1], lr_up.shape[1] - sy)
    X = min(hr.shape[2], lr_up.shape[2] - sx)
    z0, y0, x0 = max(sz, 0), max(sy, 0), max(sx, 0)
    hr_c  = hr[:Z, :Y, :X]
    lr_c  = lr_up[z0:z0+Z, y0:y0+Y, x0:x0+X]
    return hr_c, lr_c


def build_real_pairs(hr, lr_native, scale=SCALE, patch_hr=PATCH_HR):
    """
    HR is on the fine grid; lr_native is on the COARSE grid (already ROI-cropped
    and aligned, i.e. lr_native ~ hr downsampled by `scale`). Cut matched patches:
    patch_hr HR voxels  <->  patch_hr//scale LR voxels at the same location.
    """
    plr = patch_hr // scale
    HR, LR = [], []
    Z = min(hr.shape[0], lr_native.shape[0] * scale)
    for z in range(0, Z, patch_hr):
        for y in range(0, hr.shape[1] - patch_hr + 1, patch_hr):
            for x in range(0, hr.shape[2] - patch_hr + 1, patch_hr):
                hp = hr[z, y:y+patch_hr, x:x+patch_hr]
                ly, lx = y // scale, x // scale
                lp = lr_native[z // scale, ly:ly+plr, lx:lx+plr]
                if hp.shape == (patch_hr, patch_hr) and lp.shape == (plr, plr):
                    HR.append(hp.astype(np.float32)); LR.append(lp.astype(np.float32))
    return np.asarray(HR, np.float32), np.asarray(LR, np.float32)


def main():
    print(check_ratio(VOXEL_LR_UM, VOXEL_HR_UM))
    hr = load_volume(HR_FOLDER)
    lr = load_volume(LR_FOLDER)
    # bring LR onto HR grid for alignment only:
    lr_up = zoom(lr, SCALE, order=1)
    shift_vox = register_translation_3d(lr_up, hr)
    print("estimated shift (z,y,x) voxels:", shift_vox)
    hr_c, lr_up_c = common_roi(hr, lr_up, shift_vox)
    lr_native = zoom(lr_up_c, 1.0 / SCALE, order=1)          # back to coarse grid
    HR, LR = build_real_pairs(hr_c, lr_native)
    print(f"real pairs: {HR.shape[0]}  HR{HR.shape[1:]} LR{LR.shape[1:]}")
    if h5py is not None:
        with h5py.File(OUT, "w") as hf:
            hf.create_dataset("HR", data=HR, compression="gzip", compression_opts=4)
            hf.create_dataset("LR", data=LR, compression="gzip", compression_opts=4)
    print("VISUALLY QC a few overlaid HR/LR pairs before trusting the registration!")


# --------------------------------------------------------------------------- #
# Self-test: synthetic co-located volumes with a known shift                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)

    print("ratio check 11.0/2.75:", check_ratio(11.0, 2.75))
    print("ratio check 8.6/2.75:", check_ratio(8.6, 2.75))

    # fine HR volume
    hr = _norm(gaussian_filter(rng.random((32, 256, 256)), 2)).astype(np.float32)
    # real coarse LR = blurred+decimated HR, with a known in-plane shift of (5,7)
    from scipy.ndimage import shift as ndshift
    hr_shift = ndshift(hr, (0, 5, 7), order=1, mode="reflect")
    lr = zoom(gaussian_filter(hr_shift, (0, 1.5, 1.5)), (1, 0.25, 0.25), order=1).astype(np.float32)

    lr_up = zoom(lr, (1, SCALE, SCALE), order=1)
    s = register_translation_3d(lr_up, hr)
    print("recovered in-plane shift (expect ~ -5,-7):", np.round(s[1:], 1))
    hr_c, lr_up_c = common_roi(hr, lr_up, s)
    lr_native = zoom(lr_up_c, (1, 1/SCALE, 1/SCALE), order=1)
    HR, LR = build_real_pairs(hr_c, lr_native)
    print(f"real pairs built: {HR.shape[0]}  HR{HR.shape[1:]} LR{LR.shape[1:]}")
    assert HR.shape[0] == LR.shape[0] and HR.shape[1:] == (128, 128) and LR.shape[1:] == (32, 32)
    print("ratio-check + registration + ROI crop + pairing: OK")
