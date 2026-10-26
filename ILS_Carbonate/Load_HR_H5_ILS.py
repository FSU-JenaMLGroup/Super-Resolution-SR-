# =============================================================================
# Load_HR_H5_ILS.py  —  ILS Carbonate HR dataset builder
# -----------------------------------------------------------------------------
# EXACT structural mirror of Load_HR_H5.py (the SS/Berea HR builder).
# Only the SOURCE folder and the OUTPUT .h5 filename differ. Everything else —
# sequential 128x128 tiling, per-patch cv2.NORM_MINMAX to [0,1], the '3D_image'
# key, NUM_PATCHES — is byte-for-byte identical so the ILS pipeline behaves
# exactly like the SS pipeline.
# =============================================================================
from PIL import Image
import glob
import numpy as np
import time
import cv2
import h5py
from datetime import timedelta

# --- representative image just for the dtype print (mirror of SS script) ------
Image_Folder_Path = r'D:\Muturi D\SR\ILS45\ILS_Dataset\\'   # <-- ILS source (was SR\Dataset)
_first = sorted(glob.glob1(Image_Folder_Path, "*.tiff"))[0]
y = cv2.imread(Image_Folder_Path + _first)
print('dtype Original Images: ', y.dtype)

PATCH_SIZE = 128
NUM_PATCHES = 13224             # 551 of 701 slices (~24 patches/slice); reserves 150
                                 # slices as a leakage-free held-out pool for
                                 # SS_ILS45_noise_test.py (see check.py held-out warning)


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
print(np.min(PNGarray_New[0]), " , ", np.max(PNGarray_New[0]))
print('dtype Images After Normalization: ', PNGarray_New.dtype)
print('Shape Images After Normalization: ', PNGarray_New.shape)

print(f'{PNGarray_New.shape[0]} subimages with a size of {(PATCH_SIZE, PATCH_SIZE)} were generated.')

print('*** Saving h5 database ***')
h5py_name = r"D:\Muturi D\SR\ILS45\Data\HR_ILS.h5"      # <-- ILS output (was HR_Q01.h5)
hf = h5py.File(h5py_name, 'w')
hf.create_dataset('3D_image', data=PNGarray_New)
hf.close()

elapsed = time.time() - t
print('elapsed time=', str(timedelta(seconds=elapsed)), ' sec')