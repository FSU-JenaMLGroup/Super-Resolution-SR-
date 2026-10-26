# =============================================================================
# Load_LR_H5_ILS.py  —  ILS Carbonate LR dataset builder
# -----------------------------------------------------------------------------
# EXACT structural mirror of Load_LR_H5.py (the SS/Berea LR builder).
# Same tiling, same per-patch cv2.NORM_MINMAX, same 4x downscale via TWO
# successive cv2.pyrDown (128 -> 64 -> 32), same '3D_image' key. Only the
# SOURCE folder and OUTPUT .h5 filename differ.
#
# This operator MUST match Load_LR_H5.py exactly: it is the LR the model is
# trained on, and the same operator the noise-test regenerates. Your paper-mode
# check (A1_storedLR == A2_regenLR to 4 dp) confirmed the operators agree.
# =============================================================================
from PIL import Image
import glob
import numpy as np
import time
import cv2
import h5py
from datetime import timedelta

Image_Folder_Path = r'D:\Muturi D\SR\ILS45\ILS_Dataset\\'   # <-- ILS source
_first = sorted(glob.glob1(Image_Folder_Path, "*.tiff"))[0]
y = cv2.imread(Image_Folder_Path + _first)
print('dtype Original Images: ', y.dtype)

PATCH_SIZE = 128
LR_SIZE = 32   # FSRCNN scale=4: LRsample_shape 32x32 -> HRsample_shape 128x128
NUM_PATCHES = 13224             # must match Load_HR_H5_ILS.py exactly (same slice
                                 # order -> same held-out reserve of 150 slices)


def extract_patches(folder, patch_size, num_patches):
    g = sorted(glob.glob1(folder, "*.tiff"))
    print(f'{len(g)} 2D TIF images were Found')
    im = Image.open(folder + g[0])
    w, h = im.size
    print(f'2D image size: {w}x{h}')
    patches = []
    for img_name in g:
        img = np.array(Image.open(folder + img_name))
        rows = img.shape[0] // patch_size
        cols = img.shape[1] // patch_size
        for r in range(rows):
            for c in range(cols):
                patches.append(img[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size])
                if len(patches) >= num_patches:
                    return np.array(patches, dtype=np.float64)
    return np.array(patches, dtype=np.float64)


t = time.time()
print('*** Loading and patching images ***')
PNGarray = extract_patches(Image_Folder_Path, PATCH_SIZE, NUM_PATCHES)
print('dtype Load Images: ', PNGarray.dtype)

PNGarray_New = np.zeros((len(PNGarray), PATCH_SIZE, PATCH_SIZE), dtype=np.float64)
for i in range(len(PNGarray)):
    PNGarray_New[i] = cv2.normalize(PNGarray[i], None, 0.0, 1.0, cv2.NORM_MINMAX)
print('dtype Images After Normalization: ', PNGarray_New.dtype)
print('Shape Images After Normalization: ', PNGarray_New.shape)

PNGarray_New_Downsample = np.zeros((len(PNGarray), LR_SIZE, LR_SIZE), dtype=np.float64)
print(f'***** DownSampling (4x: {PATCH_SIZE}x{PATCH_SIZE} -> {LR_SIZE}x{LR_SIZE}) *****')
for j in range(len(PNGarray)):
    tmp = cv2.pyrDown(PNGarray_New[j])          # 128 -> 64
    PNGarray_New_Downsample[j] = cv2.pyrDown(tmp)  # 64 -> 32

print('dtype Images After DownSampling: ', PNGarray_New_Downsample.dtype)
print('Shape Images After DownSampling: ', PNGarray_New_Downsample.shape)

print(f'{PNGarray_New_Downsample.shape[0]} subimages with a size of {(LR_SIZE, LR_SIZE)} were generated.')

print('*** Saving h5 database ***')
h5py_name = r"D:\Muturi D\SR\ILS45\Data\LR_ILS.h5"      # <-- ILS output (was LR_Q01.h5)
hf = h5py.File(h5py_name, 'w')
hf.create_dataset('3D_image', data=PNGarray_New_Downsample)
hf.close()

elapsed = time.time() - t
print('elapsed time=', str(timedelta(seconds=elapsed)), ' sec')
