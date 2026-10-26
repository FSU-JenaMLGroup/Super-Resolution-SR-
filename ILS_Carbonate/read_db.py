#===================================================================================================================================
"""
Created on Sun Feb  6 19:04:14 2022

@Author: Pouya Sadeghi
"""
#===================================================================================================================================
# Compare HR and LR Images Together

"""
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

HRdb_path = r"D:\Muturi D\SR\FSRCNN\HR_Q01.h5"
LRdb_path = r"D:\Muturi D\SR\FSRCNN\LR_Q01.h5"

import os
_out_dir = r"D:\Muturi D\SR\FSRCNN\output\db_samples"
os.makedirs(_out_dir, exist_ok=True)

#===================================================================================================================================

print('*** Loading dataset ***')
with h5py.File(HRdb_path, "r") as hf:    
    HRdb = np.array(hf['3D_image'])
with h5py.File(LRdb_path, "r") as hf:    
    LRdb = np.array(hf['3D_image'])

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
    plt.savefig(os.path.join(_out_dir, f'Sample{i}.png'))
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
    plt.savefig(os.path.join(_out_dir, f'Sample{i}.png'))
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
    plt.savefig(os.path.join(_out_dir, f'Sample{i}.png'))
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
    plt.savefig(os.path.join(_out_dir, f'Sample{i}.png'))
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
    plt.savefig(os.path.join(_out_dir, f'Sample{i}.png'))
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
    plt.savefig(os.path.join(_out_dir, f'Sample{i}.png'))
    # plt.show()

for i in range(19950, 20000):
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(HRdb[i])
    ax1.set_title('The image {} of HR Images'.format(i), fontsize=18)
    ax2.imshow(LRdb[i])
    ax2.set_title('The Image {} of LR Images'.format(i), fontsize=18)
    plt.savefig(os.path.join(_out_dir, f'Sample{i}.png'))
    # plt.show()
#===================================================================================================================================