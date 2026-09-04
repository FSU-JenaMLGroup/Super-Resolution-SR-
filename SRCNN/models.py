import torch.nn as nn

#==================================================================================================================================================================================
"""
SECTION : Build Model
Three Sensitive Variable in SRCNN:
-- The LR feature dimension : < d >
-- The Number of Shrinking Filters: < s >
-- The Mapping Depth: < m >
First Part        :  Conv(9, d, 1)       ,    represents the Feature Extraction.
Second Part   :  Conv(5, s, d)        ,    represents the Shrinking.
Third Part      :  Conv(5, 1, s) * m  ,    represents the Mapping.
"""
# ****************************************************** Function: This Class Defines Structure of SRCNN Model. ********************************************************
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4);
        self.relu1 = nn.ReLU();
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2);
        self.relu2 = nn.ReLU();
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2);

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out
#==================================================================================================================================================================================