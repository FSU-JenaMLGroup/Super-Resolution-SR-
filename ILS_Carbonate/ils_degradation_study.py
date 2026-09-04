"""
ils_degradation_study.py
========================
Carbonate (ILS) degradation + noise-level study for the FSRCNN-RB revision.

Working scan : zoom1_0.4 ... 10um (9.98 um/voxel, 1003x1013x1024)  -> HR / GT
Real low-res : ILS45 ... 20um   (20.46 um/voxel)                    -> match target
Real high-res: zoom1_4x ... 3.9um (3.95 um/voxel)                   -> optional HR

SCALE NOTE
----------
20.46 / 9.98 = 2.05x  ->  to MATCH the real low-res scan, degrade the 9.98um
HR by SCALE=2. Use SCALE=4 only as a pure robustness sweep (LR ~40um has no
real counterpart). One config line below.

What this module does
---------------------
1. degrade()                 graded Poisson-Gaussian + PSF degradation (5 levels)
2. calibrate_*               anchor noise/blur to the REAL 20um scan (not guessed)
3. evaluate_across_levels()  SSIM/PSNR of the model at noise levels 1..5
4. plot_metric_vs_noise()    the SSIM-vs-noise curve
5. make_panel()              one-row figure: Original | HR(SR) | LR | noise1..5
6. export_for_geodict()      100 disjoint sub-cubes (GT & SR) as TIFF for GeoDict

Degradation/noise model references (see module footer for full cites):
  Foi et al. 2008 (Poisson-Gaussian); Usui et al. 2021 (graded low-dose CT
  simulation via Poisson + MTF, evaluated by SSIM/PSNR); Zhang et al. 2021
  (BSRGAN); Wang et al. 2021 (Real-ESRGAN); Wang et al. 2004 (SSIM).
"""

from __future__ import annotations
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

try:
    import tifffile
    _HAS_TIFF = True
except Exception:
    _HAS_TIFF = False

# --------------------------------------------------------------------------- #
# CONFIG                                                                       #
# --------------------------------------------------------------------------- #
SCALE = 2                      # 2 -> matches real 20um low-res ; 4 -> robustness only
HR_VOXEL_UM = 9.98             # working (HR/GT) scan voxel size
PSF_SIGMA = (1.10, 1.10)       # in-plane PSF std (HR px); calibrate via estimate_psf_sigma

# Five graded noise levels = five "dose" levels. Following the low-dose CT
# simulation convention (Usui 2021), each step roughly HALVES the photon budget
# (peak) and raises read noise. Level 3 is the one to anchor to the real 20um scan.
NOISE_LEVELS = {
    1: dict(poisson_peak=2000, detector_sigma=0.005),   # mild  (high dose)
    2: dict(poisson_peak=1000, detector_sigma=0.010),
    3: dict(poisson_peak=500,  detector_sigma=0.020),   # ~ real 20um (anchor)
    4: dict(poisson_peak=250,  detector_sigma=0.040),
    5: dict(poisson_peak=125,  detector_sigma=0.080),   # severe (low dose)
}


# --------------------------------------------------------------------------- #
# 1. Degradation operator                                                     #
# --------------------------------------------------------------------------- #
def degrade(hr, level=3, scale=SCALE, blur_sigma=PSF_SIGMA, rng=None):
    """
    Degrade an HR slice (2D float in [0,1]) to a noisy LR slice at a given level.
    Pipeline:  PSF blur -> area downsample -> Poisson(shot) -> Gaussian(read).
    """
    rng = rng or np.random.default_rng()
    p = NOISE_LEVELS[level]
    hr = np.clip(np.asarray(hr, np.float64), 0, 1)

    # (a) PSF blur (instrument MTF) — fixed across levels (a property of optics)
    img = gaussian_filter(hr, sigma=blur_sigma, mode="reflect")

    # (b) Area-average downsampling by `scale` (detector integration)
    H, W = img.shape
    H2, W2 = H // scale * scale, W // scale * scale
    img = img[:H2, :W2].reshape(H2 // scale, scale, W2 // scale, scale).mean(axis=(1, 3))

    # (c) Poisson shot noise — signal-dependent (photon counting), Foi 2008
    if p["poisson_peak"]:
        img = rng.poisson(np.clip(img, 0, 1) * p["poisson_peak"]) / p["poisson_peak"]

    # (d) Gaussian read noise — signal-independent (electronics), Foi 2008
    if p["detector_sigma"]:
        img = img + rng.normal(0, p["detector_sigma"], img.shape)

    return np.clip(img, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Calibration to the REAL low-res scan (so levels aren't guessed)          #
# --------------------------------------------------------------------------- #
def estimate_noise_sigma(flat_patch):
    """Read-noise std from a single-phase (flat) region of the REAL 20um scan."""
    return float(np.std(flat_patch))


def estimate_psf_sigma(edge_profile, dx=1.0):
    """PSF std from an edge-spread profile across a sharp grain boundary."""
    lsf = np.abs(np.gradient(np.asarray(edge_profile, float), dx))
    s = lsf.sum()
    if s <= 0:
        raise ValueError("flat edge profile")
    x = np.arange(len(lsf)) * dx
    mu = (x * lsf).sum() / s
    return float(np.sqrt(((x - mu) ** 2 * lsf).sum() / s))


def anchor_level_to_real(real_flat_patch, level=3):
    """Set NOISE_LEVELS[level]['detector_sigma'] from the measured real-scan noise."""
    sig = estimate_noise_sigma(real_flat_patch)
    NOISE_LEVELS[level]["detector_sigma"] = round(sig, 4)
    return sig


# --------------------------------------------------------------------------- #
# 3. Metrics + evaluation across noise levels                                 #
# --------------------------------------------------------------------------- #
def metrics(sr, hr):
    sr = np.clip(sr, 0, 1); hr = np.clip(hr, 0, 1)
    return (float(sk_ssim(hr, sr, data_range=1.0)),
            float(sk_psnr(hr, sr, data_range=1.0)))


def evaluate_across_levels(hr_stack, model_infer, scale=SCALE, seed=0):
    """
    For each noise level, degrade every HR slice, run the SR model, and average
    SSIM/PSNR(SR, HR). `model_infer(lr2d)->sr2d` wraps your trained network.
    Returns {level: {'ssim':.., 'psnr':..}}.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for lvl in sorted(NOISE_LEVELS):
        ss, ps = [], []
        for hr in hr_stack:
            lr = degrade(hr, level=lvl, scale=scale, rng=rng)
            sr = model_infer(lr)
            s, p = metrics(sr, hr)
            ss.append(s); ps.append(p)
        out[lvl] = dict(ssim=float(np.mean(ss)), psnr=float(np.mean(ps)))
        print(f"level {lvl}: SSIM={out[lvl]['ssim']:.4f}  PSNR={out[lvl]['psnr']:.2f} dB")
    return out


def plot_metric_vs_noise(results, out_png="ssim_psnr_vs_noise.png"):
    """SSIM and PSNR vs noise level (twin y-axes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = sorted(results)
    ssim = [results[l]["ssim"] for l in levels]
    psnr = [results[l]["psnr"] for l in levels]
    fig, ax1 = plt.subplots(figsize=(5, 3.4))
    ax1.plot(levels, ssim, "o-", color="tab:blue", label="SSIM")
    ax1.set_xlabel("Noise level (1 = mild  →  5 = severe)")
    ax1.set_ylabel("SSIM", color="tab:blue"); ax1.set_xticks(levels)
    ax2 = ax1.twinx()
    ax2.plot(levels, psnr, "s--", color="tab:red", label="PSNR")
    ax2.set_ylabel("PSNR (dB)", color="tab:red")
    fig.suptitle("FSRCNN-RB robustness vs degradation severity")
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# 4. Comparison panel:  Original | HR(SR) | LR | noise1..5  (with scale bars)  #
# --------------------------------------------------------------------------- #
def make_panel(named_images, out_png="ils_panel.png", bar_um=200.0):
    """
    named_images: list of (title, image2d, voxel_um). Images shown in one row;
    each gets a scale bar of length `bar_um`. LR/noise images are nearest-upscaled
    for display only (labelled with their true voxel size).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(named_images)
    disp = max(img.shape[0] for _, img, _ in named_images)   # common display size
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, (title, img, vox) in zip(axes, named_images):
        z = disp // img.shape[0]
        shown = np.kron(img, np.ones((max(z, 1), max(z, 1)))) if z > 1 else img
        ax.imshow(shown, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{title}\n{vox:.2f} µm/vox", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        bar_px = bar_um / vox * (shown.shape[1] / img.shape[1])   # scale bar in display px
        y0 = shown.shape[0] * 0.92
        ax.add_patch(Rectangle((shown.shape[1] * 0.05, y0), bar_px, shown.shape[0] * 0.02,
                               color="white"))
        ax.text(shown.shape[1] * 0.05, y0 - shown.shape[0] * 0.03,
                f"{bar_um:.0f} µm", color="white", fontsize=6)
    fig.tight_layout(); fig.savefig(out_png, dpi=300, facecolor="black"); plt.close(fig)
    return out_png


def build_noise_row(hr, scale=SCALE, sr_image=None, rng=None):
    """Convenience: assemble the (Original, [HR/SR], LR, noise1..5) list for make_panel."""
    rng = rng or np.random.default_rng(0)
    row = [("Original (GT)", hr, HR_VOXEL_UM)]
    if sr_image is not None:
        row.append(("HR (SR)", sr_image, HR_VOXEL_UM))
    lr_clean = degrade(hr, level=1, scale=scale, rng=rng)
    row.append(("LR input", lr_clean, HR_VOXEL_UM * scale))
    for lvl in sorted(NOISE_LEVELS):
        row.append((f"noise {lvl}", degrade(hr, level=lvl, scale=scale, rng=rng),
                    HR_VOXEL_UM * scale))
    return row


# --------------------------------------------------------------------------- #
# 5. GeoDict export: 100 disjoint sub-cubes (GT & SR) for porosity/perm        #
# --------------------------------------------------------------------------- #
def export_for_geodict(volumes, out_dir, n=100, cube=200, segment=False,
                       thr=None, seed=0):
    """
    volumes : dict name -> 3D grayscale volume in [0,1], e.g. {'GT':.., 'SR':..}
    Writes up to `n` disjoint cube^3 sub-volumes per named volume as TIFF stacks
    GeoDict can import (Process Geometry > Import). Returns a manifest list.

    segment=True writes a binary (0/255) volume using threshold `thr` (Otsu on GT
    if None) so GeoDict gets a ready segmented structure; otherwise grayscale uint8.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    ref = next(iter(volumes.values()))
    Z, Y, X = ref.shape
    # enumerate disjoint cube origins, shuffle, take n
    origins = [(z, y, x)
               for z in range(0, Z - cube + 1, cube)
               for y in range(0, Y - cube + 1, cube)
               for x in range(0, X - cube + 1, cube)]
    rng.shuffle(origins)
    origins = origins[:n]

    if segment and thr is None:
        thr = _otsu(ref)

    manifest = []
    for name, vol in volumes.items():
        sub = os.path.join(out_dir, name); os.makedirs(sub, exist_ok=True)
        for i, (z, y, x) in enumerate(origins):
            c = vol[z:z+cube, y:y+cube, x:x+cube]
            if segment:
                arr = ((c < thr).astype(np.uint8)) * 255      # pores=255
                phi = float((c < thr).mean())
            else:
                arr = (np.clip(c, 0, 1) * 255).astype(np.uint8)
                phi = float((c < (thr if thr else _otsu(ref))).mean())
            path = os.path.join(sub, f"cube_{i:04d}.tif")
            if _HAS_TIFF:
                tifffile.imwrite(path, arr)
            else:                                             # PIL multipage fallback
                from PIL import Image
                imgs = [Image.fromarray(s) for s in arr]
                imgs[0].save(path, save_all=True, append_images=imgs[1:])
            manifest.append(dict(name=name, idx=i, origin=(z, y, x),
                                 porosity=round(phi, 4), path=path))
    # write manifest CSV
    import csv
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "idx", "origin", "porosity", "path"])
        w.writeheader(); w.writerows(manifest)
    print(f"exported {len(origins)} cubes x {len(volumes)} volumes -> {out_dir}")
    return manifest


def _otsu(vol, bins=256):
    hist, edges = np.histogram(vol.ravel(), bins=bins, range=(0, 1))
    p = hist / hist.sum(); omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + np.diff(edges) / 2)); mu_t = mu[-1]
    denom = omega * (1 - omega); denom[denom == 0] = np.nan
    k = np.nanargmax((mu_t * omega - mu) ** 2 / denom)
    return float(edges[:-1][k] + np.diff(edges)[0] / 2)


# --------------------------------------------------------------------------- #
# Self-test (synthetic carbonate-like data)                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    # synthetic grayscale "carbonate" slice: grains + pores
    hr = gaussian_filter(rng.random((256, 256)), 2.5)
    hr = (hr - hr.min()) / np.ptp(hr)

    print("== degradation levels (SSIM of noisy LR upsampled vs HR) ==")
    for lvl in sorted(NOISE_LEVELS):
        lr = degrade(hr, level=lvl, scale=SCALE, rng=rng)
        up = np.kron(lr, np.ones((SCALE, SCALE)))[:hr.shape[0], :hr.shape[1]]
        print(f"  level {lvl}: LR{lr.shape}  SSIM(up,HR)={metrics(up, hr)[0]:.3f}")

    # dummy "model" = bicubic-ish upsample, just to exercise the curve
    def dummy_model(lr):
        up = np.kron(lr, np.ones((SCALE, SCALE)))
        return gaussian_filter(up, 0.5)[:hr.shape[0], :hr.shape[1]]

    res = evaluate_across_levels(np.stack([hr] * 4), dummy_model, scale=SCALE)
    print("curve png:", plot_metric_vs_noise(res, "/home/claude/ssim_vs_noise.png"))

    panel = make_panel(build_noise_row(hr, scale=SCALE, sr_image=dummy_model(degrade(hr,1)),
                                       rng=np.random.default_rng(2)),
                       "/home/claude/ils_panel.png")
    print("panel png:", panel)

    # GeoDict export test on tiny synthetic volumes
    gt = gaussian_filter(rng.random((200, 200, 200)), 2); gt = (gt - gt.min()) / np.ptp(gt)
    sr = np.clip(gt + rng.normal(0, 0.02, gt.shape), 0, 1)
    man = export_for_geodict({"GT": gt, "SR": sr}, "/home/claude/geodict_out",
                             n=4, cube=100, segment=True)
    print("manifest rows:", len(man), "| example:", man[0])
