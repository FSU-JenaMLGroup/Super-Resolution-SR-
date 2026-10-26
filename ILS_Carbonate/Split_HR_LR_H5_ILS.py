# =============================================================================
# Split_HR_LR_H5_ILS.py  —  builds train/val/test H5 files for FSRCNN-RB
# -----------------------------------------------------------------------------
# Reads the aligned patch arrays from HR_ILS.h5 and LR_ILS.h5 (each keyed
# '3D_image', produced by Load_HR_H5_ILS.py / Load_LR_H5_ILS.py) and writes
# three new files — train_ILS.h5, val_ILS.h5, test_ILS.h5 — each containing
# both an 'HR' and an 'LR' dataset, split by the same shuffled patch indices
# so HR/LR pairs stay aligned. This matches what FSRCNN-RB.py expects.
# =============================================================================
import h5py
import numpy as np

HR_PATH = r"D:\Muturi D\SR\ILS45\Data\HR_ILS.h5"
LR_PATH = r"D:\Muturi D\SR\ILS45\Data\LR_ILS.h5"

TRAIN_PATH = r"D:\Muturi D\SR\ILS45\Data\train_ILS.h5"
VAL_PATH = r"D:\Muturi D\SR\ILS45\Data\val_ILS.h5"
TEST_PATH = r"D:\Muturi D\SR\ILS45\Data\test_ILS.h5"

VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SEED = 123

with h5py.File(HR_PATH, 'r') as hf:
    HR_all = np.array(hf['3D_image'])

with h5py.File(LR_PATH, 'r') as hf:
    LR_all = np.array(hf['3D_image'])

assert HR_all.shape[0] == LR_all.shape[0], "HR/LR patch counts do not match"

num_samples = HR_all.shape[0]
rng = np.random.RandomState(SEED)
indices = rng.permutation(num_samples)

num_val = int(num_samples * VAL_SPLIT)
num_test = int(num_samples * TEST_SPLIT)

val_idx = indices[:num_val]
test_idx = indices[num_val:num_val + num_test]
train_idx = indices[num_val + num_test:]

splits = {
    TRAIN_PATH: train_idx,
    VAL_PATH: val_idx,
    TEST_PATH: test_idx,
}

for path, idx in splits.items():
    print(f'*** Saving {path} ({len(idx)} patches) ***')
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('HR', data=HR_all[idx])
        hf.create_dataset('LR', data=LR_all[idx])

print('Done.')
