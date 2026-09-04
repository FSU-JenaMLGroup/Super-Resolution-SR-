'''
Preprocessing: build realistic LR/HR patch databases for SR training.
 
Pipeline
    Load TIFF stack -> split slices -> normalize each slice (once) ->
    extract 128x128 HR patches -> realistic degradation -> 32x32 LR patch ->
    stream both to per-split HDF5 (train/val/test).
 
Output format (NEW):
    train.h5 / val.h5 / test.h5, each with TWO datasets: 'HR' (N,128,128)
    and 'LR' (N,32,32). The downstream FSRCNN / FSRCNN-RB / SRCNN training
'''
from PIL import Image
import glob
import numpy as np
import time
import cv2
import h5py
import os
from datetime import timedelta
from scipy.ndimage import gaussian_filter, shift

# =====================================================================
# Check sample image
# =====================================================================
Image_Folder_Path = "D:/Muturi D/SR/Dataset/"
SAMPLE_TIF   = Image_Folder_Path + "Q01_5mag_resample_NLMF_croppedZ00001_1200x1200x1200-His22-27k-NLMFZ00001.tif"

y = cv2.imread(SAMPLE_TIF, cv2.IMREAD_UNCHANGED)

print('dtype / shape of sample image:', y.dtype, y.shape)

# =====================================================================
# Load 2D Image stack
# =====================================================================

def Load_2D_Images (Image_Folder_Path):
    # Finding Number of PNG Images
    g = sorted(glob.glob1(Image_Folder_Path, "*.tif"))
    Number_of__Images = len(g)
    print(f'{Number_of__Images} 2D TIF images were Found')

    # Find w & h of a sample image at index of 0
    im = Image.open(Image_Folder_Path + g[0])
    #Image.open(os.path.join(Image_Folder_Path, g[0]))
    h, w = im.size
    print(f'2D image size: {h}x{w}')
    PNGarray = np.zeros((Number_of__Images, w, h)).astype(np.float32)
    for i, img in enumerate(g):
        PNGarray[i, :, :] = np.array(Image.open(Image_Folder_Path + img))

    return PNGarray

t = time.time()
print('*** Loading the main image ***')
# Loading 2D PNG Images
PNGarray = Load_2D_Images(Image_Folder_Path)
print('dtype Load Images: ', PNGarray.dtype)


# =====================================================================
# Degradation Model
# =====================================================================

def Degradation_Model(hr, scale=4, blur_sigma=(2.84, 3.72), poisson_peak=None, detector_sigma=0.02, max_shift=0.0, anisotropic=False, rng=None):
    """
    Degrade one HR patch (2D, float in [0,1]) to a realistic LR patch.
    blur_sigma=(2.84,3.72) corresponds to an LR sharpness q in [0.8,1.0] LR-voxels
    for scale=4 (derived from the measured HR PSF of 0.90 HR-px). max_shift is 0
    for SYNTHETIC pairs (perfect LR/HR correspondence is required for supervision).

    Parameters
    ----------
    hr            : HR slice, shape (H, W), values in [0, 1].
    scale         : integer downsampling factor (HR/LR), e.g. 4.
    blur_sigma    : (low, high) range for the PSF std (pixels). A value is
                    sampled uniformly in this range per call; set both equal
                    to use a fixed, calibrated sigma.
    poisson_peak  : photon budget for shot noise. Higher = less noise.
                    Set None to disable Poisson noise. calibrate per scanner.
    detector_sigma: additive Gaussian (read) noise std; calibrate from a flat
                    region via estimate_noise_sigma().
    max_shift  : max sub-pixel registration error (pixels) applied before
                    decimation. Set 0.0 to disable.
    anisotropic   : if True, sample independent sigma for rows/cols (mimics a
                    directional PSF); if False, use one isotropic sigma. 
                    the in-plane micro-CT PSF is effectively isotropic
    rng           : np.random.Generator for reproducibility.

    Returns
    """
    rng = rng if rng is not None else np.random.default_rng()
    hr = np.clip(hr.astype(np.float64), 0, 1)  # Ensure HR patch is in [0, 1]

    # 1. Gaussian PSF Blur simulates scanner optics
    sy = rng.uniform(*blur_sigma)
    sx = rng.uniform(*blur_sigma) if anisotropic else sy
    img = gaussian_filter(hr, sigma=(sy, sx), mode='reflect')

    # 2. Sub-pixel shift simulates registration errors  since misregistration (mimics imperfect LR/HR alignment)
    if max_shift > 0:
        dy = rng.uniform(-max_shift, max_shift)
        dx = rng.uniform(-max_shift, max_shift)
        img = shift(img, shift=(dy, dx), order=1, mode='reflect')

    # 3. Area Downsampling (×4) - area averaging simulates detector integration
    H, W = img.shape
    H2, W2 = H // scale * scale, W // scale * scale
    img = img[:H2, :W2].reshape(H2 // scale, scale, W2 // scale, scale).mean(axis=(1, 3)) # Crop to be divisible by scale

    # 4. Poisson Noise simulates photon counting statistics
    if poisson_peak is not None:
        img = rng.poisson(np.clip(img, 0, 1) * poisson_peak) / poisson_peak

    # 5. Gaussian Detector Noise -Gaussian noise simulates detector read noise
    if detector_sigma > 0:
        img += rng.normal(0, detector_sigma, img.shape)

    return np.clip(img, 0, 1).astype(np.float32)

# =====================================================================
# Per-slice normalization  (consistent intensity scale -> preserves petrophysics)
# =====================================================================
def normalize_slice(img):
    """Min-max normalize a FULL slice to [0,1] using the slice's own intensity
    range. (Do NOT normalize per patch: homogeneous patches would have their
    tiny range stretched to full contrast, fabricating structure.)"""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-12)

# =====================================================================
# Patch extraction -> incremental HDF5 (flat memory)
# =====================================================================
def generate_database(img_set,h5_path,rng, patch_size=128, stride=128, scale=4, max_shift=0.0): #shift OFF for synthetic pairs
    lr = patch_size // scale
    with h5py.File(h5_path, 'w') as hf:
        HR = hf.create_dataset('HR', shape=(0, patch_size, patch_size), maxshape=(None, patch_size, patch_size), chunks=(64, patch_size, patch_size), dtype='float32', compression='gzip', compression_opts=4)
        LR = hf.create_dataset('LR', shape=(0, lr, lr), maxshape=(None, lr, lr), chunks=(64, lr, lr), dtype='float32', compression='gzip', compression_opts=4)

        n = 0
        for img in img_set:
            img = normalize_slice(img) # Normalize the entire slice before patch extraction
            h, w = img.shape
            hb, lb = [], []
            
            for yy in range(0,h - patch_size + 1, stride):
                for xx in range(0,w - patch_size + 1, stride):
                    hp = img[yy:yy+patch_size, xx:xx+patch_size]
                    lp = Degradation_Model(hp, scale=scale, blur_sigma=(2.84, 3.72), poisson_peak=None, detector_sigma=0.02, anisotropic=False, max_shift=max_shift, rng=rng)
                    hb.append(hp); lb.append(lp)

            k = len(hb)
            HR.resize(n + k, axis=0); LR.resize(n + k, axis=0)
            HR[n:n+k] = np.array(hb, np.float32)
            LR[n:n+k] = np.array(lb, np.float32)
            n += k
            print(f'  {os.path.basename(h5_path)}: wrote {n} patch pairs')
            
        return n
#LR patches must be generated from the same HR patches so that every LR patch 
#corresponds exactly to an HR patch. HR - extracting patches from 1200×1200 images. 
#The LR - extract exact same patch then downsample each patch from 128×128 → 64×64.

# =====================================================================
# Split with a buffer GAP to decorrelate adjacent slices, then build DBs
# =====================================================================
print('*** Generating databases ***')
GAP = 20  #decorrelate split boundaries splits are disjoint with guard gaps (train 0–25, val 28–31, test 34–39), so no patch can leak across splits.
train_imgs = PNGarray[0:920]
val_imgs = PNGarray[920 + GAP: 1040]
test_imgs = PNGarray[1040 + GAP: 1200]

rng = np.random.default_rng(123)  # For reproducibility 123 is the seed value. 
#By setting a specific seed, you can ensure that the same sequence of random numbers is generated each time you run the code. 
#This is useful for debugging and comparing results, as it allows you to reproduce the same conditions and outcomes across different runs of the program.
out_dir =  Image_Folder_Path 
generate_database(train_imgs, out_dir + "train.h5", rng, stride=64) #overlap ok for train
generate_database(val_imgs, out_dir + "val.h5", rng, stride=128) #no overlap for val/test to avoid data leakage
generate_database(test_imgs, out_dir + "test.h5", rng, stride=128) # independent test patches

# =====================================================================
# Verify (read shapes back from the files we just wrote)
# =====================================================================
for split in ("train", "val", "test"):
    with h5py.File(out_dir + f"{split}.h5", 'r') as hf:
        hr_shape, lr_shape = hf['HR'].shape, hf['LR'].shape
        print(f"{split}: HR {hr_shape}  LR {lr_shape}")
        assert hr_shape[0] == lr_shape[0], f"{split}: HR/LR count mismatch"
        assert hr_shape[1:] == (128, 128) and lr_shape[1:] == (32, 32)
 
# =====================================================================
# Done
# =====================================================================
elapsed = time.time() - t
print('\nelapsed time =', str(timedelta(seconds=elapsed)))



