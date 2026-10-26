from PIL import Image
import glob
import numpy as np
import time
import cv2
import h5py
from datetime import timedelta

y = cv2.imread(r'D:\Muturi D\SR\Dataset\Q01_5mag_resample_NLMF_croppedZ00001_1200x1200x1200-His22-27k-NLMFZ00001.tif')
print('dtype Original Images: ', y.dtype)

Image_Folder_Path = r'D:\Muturi D\SR\Dataset\\'
PATCH_SIZE = 128
NUM_PATCHES = 20000

def extract_patches(folder, patch_size, num_patches):
    g = sorted(glob.glob1(folder, "*.tif"))
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
h5py_name = r"D:\Muturi D\SR\FSRCNN\HR_Q01.h5"
hf = h5py.File(h5py_name, 'w')
hf.create_dataset('3D_image', data=PNGarray_New)
hf.close()

elapsed = time.time() - t
print('elapsed time=', str(timedelta(seconds=elapsed)), ' sec')
