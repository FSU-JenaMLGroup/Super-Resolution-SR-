import torch
import numpy as np
from torch.utils.data import Dataset

#=================================================================================================================================================================================
"""
SECTION: Datasets
"""
#=================================================================================================================================================================================
class Image_Dataset(Dataset):
    def __init__(self, img_x, img_y, img_x_shape, img_y_shape):
        # Convert to float32 shared-memory tensors so DataLoader workers can
        # access data without pickling the full arrays across process pipes.
        self.img_x = torch.from_numpy(np.asarray(img_x, dtype=np.float32)).share_memory_()
        self.img_y = torch.from_numpy(np.asarray(img_y, dtype=np.float32)).share_memory_()
        self.img_x_shape = img_x_shape
        self.img_y_shape = img_y_shape

    def __len__(self):
        return len(self.img_x)

    def __getitem__(self, i):
        # img_x = (img_x-0.5)/0.5 # change to [-1,1] ***************
        # img_y = (img_y-0.5)/0.5 # change to [-1,1] ***************
        return self.img_x[i].reshape(self.img_x_shape), self.img_y[i].reshape(self.img_y_shape)
#=================================================================================================================================================================================
