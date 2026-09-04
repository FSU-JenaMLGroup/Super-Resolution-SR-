import torch.nn as nn

#==================================================================================================================================================================================
"""
SECTION 4: Build Model
Fast Super Resolution Convolutional Neural Network: 
The proposed FSRCNN is different from SRCNN mainly in three aspects.
First, FSRCNN adopts the original low-resolution image as input without 
bicubic interpolation. A deconvolution layer is introduced at the end of the
network to perform upsampling. Second, the non-linear mapping step in 
SRCNN is replaced by three steps in FSRCNN, namely the shrinking, 
mapping, and expanding step. Third, FSRCNN adopts smaller filter sizes 
and a deeper network structure. These improvements provide FSRCNN 
with better performance but lower computational cost than SRCNN.

Three Sensitive Variable in FSRCNN:
-- The LR feature dimension : < d >
-- The Number of Shrinking Filters: < s >
-- The Mapping Depth: < m >

First Part        :  Conv(5, d, 1)       ,    represents the Feature Extraction.
Second Part   :  Conv(1, s, d)        ,    represents the Shrinking.
Third Part      :  Conv(3, s, s) * m  ,    represents the Mapping.
Fourth Part    :  Conv(1, d, s)        ,    represents the Expanding.
Fifth Part       :  Deconv(9, 1, d)    ,    represents the Deconvolution.
"""
# ****************************************************** Function: This Class Defines Structure of FSRCNN Model. *******************************************************

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56,  s=20, m=4):
        super(FSRCNN, self).__init__()

        # *****************************************************************  Feature Extraction Layer ******************************************************************
        self.first_part = nn.Sequential(
                                                     nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
                                                     nn.PReLU(d)
                                                    )

        # *******************************************************************   Shrinking Layer **********************************************************************
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]

        # ********************************************************************   Mapping Layer **********************************************************************
        for i in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])

        # *******************************************************************  Expanding Layer **********************************************************************
        self.mid_part.extend ([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential (*self.mid_part)

        # *****************************************************************  Deconvolution Layer *********************************************************************
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2, output_padding=scale_factor-1)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
#==================================================================================================================================================================================