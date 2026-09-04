#===================================================================================================================================
"""
Created on Sun Feb  6 19:04:14 2022

@Author: Pouya Sadeghi
"""
#===================================================================================================================================
# Compare HR and LR Images Together
"""
Dataset/
│
├── HR.h5 roughly (346800, 128, 128)
└── LR.h5 roughly (346800, 32, 32)

FSRCNN
    ↓
FSRCNN-RB
    ↓
SRCNN

same train set, validation set, test set, fair comparison

Expected dataset size 
1200 slices
1200×1200 pixels
patch = 128
stride = 128 & 64 

Patches per dimension - (1200 - 128) / 64 + 1 ≈ 17 patches per dimension

Patches per slice - 17 × 17 = 289

Total = 289 × 1200 ≈ 346,800 paired patches

train: HR (265880, 128, 128)  LR (265880, 32, 32)
val: HR (8100, 128, 128)  LR (8100, 32, 32)
test: HR (11340, 128, 128)  LR (11340, 32, 32)

elapsed time = 0:33:11.205422

In this code, we want to compare HR images and LR images 
and check whether the produced HR images and LR images 
are similar to each other Pairwise or not?
"""
#===================================================================================================================================

import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.max_open_warning': 0})

#train
DB_path = r"D:\Muturi D\SR\Dataset\train.h5"
# #validation 
# DB_path = f"D:\Muturi D\SR\Dataset\val.h5"
# #test
# DB_path = f"D:\Muturi D\SR\Dataset\test.h5"

#===================================================================================================================================

print('*** Loading dataset ***')
with h5py.File(DB_path, "r") as hf:    
    HRdb = np.array(hf['HR'])    
    LRdb = np.array(hf['LR'])

print("HR shape:", HRdb.shape)
print("LR shape:", LRdb.shape)  

with h5py.File(DB_path, "r") as hf:
    print(list(hf.keys()))

#Verify pairwise correspondence
idx = np.random.randint(0, HRdb.shape[0])

hr = HRdb[idx]
lr = LRdb[idx]

print("Patch index:", idx)
print("HR:", hr.shape)
print("LR:", lr.shape)

#===================================================================================================================================

assert HRdb.shape[0] == LRdb.shape[0]
db_size = HRdb.shape[0]

#===================================================================================================================================
"""
Showing HR images and LR images Pairwise
"""
#===================================================================================================================================

"""
The Samples from 0 image to 49 image (50 Image)
"""
for i in range(50):
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(HRdb[i])
    ax1.set_title('The image {} of HR Images'.format(i), fontsize=18)
    ax2.imshow(LRdb[i])
    ax2.set_title('The Image {} of LR Images'.format(i), fontsize=18)
    plt.savefig(f'Sample{i}.png')
    plt.close(fig1)   # IMPORTANT: frees memory
    # plt.show()

"""
The Samples from 500 image to 550 image (50 Image)
"""

for i in range(500, 550):
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(HRdb[i])
    # ax1.set_title('The image {} of HR Images'.format(i), fontname="Brush Script MT", fontsize=18)
    ax1.set_title('The image {} of HR Images'.format(i), fontsize=18)
    ax2.imshow(LRdb[i])
    ax2.set_title('The Image {} of LR Images'.format(i), fontsize=18)
    plt.savefig(f'Sample{i}.png')
    plt.close(fig1)   # IMPORTANT: frees memory
    # plt.show()
"""
The Samples from 1000 image to 1010 image (10 Image)
"""

for i in range(1000, 1010):
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(HRdb[i])
    ax1.set_title('The image {} of HR Images'.format(i), fontsize=18)
    ax2.imshow(LRdb[i])
    ax2.set_title('The Image {} of LR Images'.format(i), fontsize=18)
    plt.savefig(f'Sample{i}.png')
    plt.close(fig1)   # IMPORTANT: frees memory
    # plt.show()
"""
The Samples from 5000 image to 5010 image (10 Image)
"""

for i in range(5000, 5010):
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(HRdb[i])
    ax1.set_title('The image {} of HR Images'.format(i), fontsize=18)
    ax2.imshow(LRdb[i])
    ax2.set_title('The Image {} of LR Images'.format(i), fontsize=18)
    plt.savefig(f'Sample{i}.png')
    plt.close(fig1)   # IMPORTANT: frees memory
    # plt.show()
"""
The Samples from 10000 image to 10010 image (10 Image)
"""

for i in range(10000, 10010):
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(HRdb[i])
    ax1.set_title('The image {} of HR Images'.format(i), fontsize=18)
    ax2.imshow(LRdb[i])
    ax2.set_title('The Image {} of LR Images'.format(i), fontsize=18)
    plt.savefig(f'Sample{i}.png')
    plt.close(fig1)   # IMPORTANT: frees memory
    # plt.show()
"""
The Samples from 19950 image to 20000 image (50 Image)
"""

for i in range(15980, 16000):
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(HRdb[i])
    ax1.set_title('The image {} of HR Images'.format(i), fontsize=18)
    ax2.imshow(LRdb[i])
    ax2.set_title('The Image {} of LR Images'.format(i), fontsize=18)
    plt.savefig(f'Sample{i}.png')
    plt.close(fig1)   # IMPORTANT: frees memory
    # plt.show()

#for i in range(19950, 20000): #crashes if dataset size is less than 20000 or more than 20000
for i in range(max(0, db_size-50), db_size): #prevents index error if dataset size is less than 20000    
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(HRdb[i])
    ax1.set_title('The image {} of HR Images'.format(i), fontsize=18)
    ax2.imshow(LRdb[i])
    ax2.set_title('The Image {} of LR Images'.format(i), fontsize=18)
    plt.savefig(f'Sample{i}.png')
    plt.close(fig1)   # IMPORTANT: frees memory
    # plt.show()
#===================================================================================================================================

#Above code opewns 200figure and never closes them.
# print('*** Loading dataset ***')
# with h5py.File(HRdb_path, "r") as hf:
#     HRdb = hf['3D_image']      # same as np.array(...)
# with h5py.File(LRdb_path, "r") as hf:
#     LRdb = hf['3D_image']
# #===================================================================================================================================
# assert HRdb.shape[0] == LRdb.shape[0]
# db_size = HRdb.shape[0]
# print("Dataset Size:", db_size)
# print("HR Shape:", HRdb.shape)
# print("LR Shape:", LRdb.shape)
# #===================================================================================================================================
# """
# Showing HR images and LR images Pairwise
# """
# #===================================================================================================================================
# # Prevent index errors if dataset size changes
# sample_ranges = [
#     range(0, min(50, db_size)),
#     range(500, min(550, db_size)),
#     range(1000, min(1010, db_size)),
#     range(5000, min(5010, db_size)),
#     range(10000, min(10010, db_size)),
#     range(max(0, db_size-50), db_size)
# ]
# for sample_range in sample_ranges:
#     for i in sample_range:
#         fig1, (ax1, ax2) = plt.subplots(1, 2)
#         ax1.imshow(HRdb[i], cmap='gray')
#         ax1.set_title(f'HR Image {i}', fontsize=18)
#         ax2.imshow(LRdb[i], cmap='gray')
#         ax2.set_title(f'LR Image {i}', fontsize=18)
#         plt.savefig(f'Sample{i}.png')
#         plt.close(fig1)   # IMPORTANT: frees memory
#===================================================================================================================================
print("Verification complete.")
#===================================================================================================================================

