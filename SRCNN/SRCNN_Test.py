"""
IN THE NAME OF GOD

Subject: Converts LR images to HR images

Created on Sun Jan 23 09:12:28 2022

@Author: Pouya Sadeghi
"""

#====================================================================================================================================================================================
"""
SECTION 1 : Import Necessary Packages (Libraries)
"""
from datasets import Image_Dataset
from utils import calc_psnr
from models import SRCNN
import numpy as np
import h5py
import copy
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import time
import torch; torch.manual_seed(0)
import torch.distributions
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('grayscale')
# mpl.style.use('classic')
# from skimage.metrics import structural_similarity
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.max_open_warning': 0})
#==========================================================================================================================================
if __name__ == '__main__':
    inp = {
        'test_h5': "D:/Muturi D/SR/Dataset/test.h5",
        'LRsample_shape': (1, 32, 32),  # (Channel, Height, Width)
        'HRsample_shape': (1, 128, 128),  # (Channel, Height, Width)
        'outputs-dir': 'P:/SR/SRCNN/output/',
        'scale': 4,
        'num_epochs': 100,
        'num_workers': 8,
        'seed': 123,
        'optimizer': 'SGD',
    }

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device:  {device}')

    t = time.time()

    model = SRCNN(scale_factor=inp['scale']).to(device)  # GPU
    # model = FSRCNN().to(device)  # GPU
    summary(model, inp['LRsample_shape'], device="cuda")
    print(model)
    inp['model'] = str(model)

    from colorama import Fore

    print(Fore.YELLOW + """" **************************************** Loading dataset **************************************** """)
    # pouya = 2675
    with h5py.File(inp['test_h5'], 'r') as hf:
        Test_y = np.array(hf['HR'])
        Test_x = np.array(hf['LR'])

    Test_Dataset = Image_Dataset(Test_x, Test_y, inp['LRsample_shape'], inp['HRsample_shape'])
    print('TestDB:', Test_Dataset.img_x.shape[0])
    Test_dataloader = DataLoader(dataset=Test_Dataset,
                                 batch_size=1)
    #============================================================================

    print(Fore.GREEN + """" **************************************** Start of Testing Network **************************************** """)

#===================================================================================================================================================================================
"""
SECTION 5 : Test Model
"""
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pathlib
chk_Path = str(pathlib.Path(__file__).parent / 'Image Output')
pathlib.Path(chk_Path).mkdir(parents=True, exist_ok=True)
from skimage.metrics import structural_similarity as ssim
import lpips
import pandas as pd

#============================================================
def plot_imgs_test(inm, inp, outp, gt, epoch):
    # input image, reconstructed image, ground truth
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.suptitle(f'Sample = {epoch}')
    ax1.imshow(inm.cpu().detach().numpy()[0, 0, :, :])
    ax1.set_title('Input')
    ax1.set_xlim([0, 64])
    ax1.set_ylim([0, 64])
    ax2.imshow(inp.cpu().detach().numpy()[0, 0, :, :])
    ax2.set_title('Bicubic')
    ax2.set_xlim([0, 128])
    ax2.set_ylim([0, 128])
    ax3.imshow(outp.cpu().detach().numpy()[0, 0, :, :])
    ax3.set_title('Reconstructed')
    ax3.set_xlim([0, 128])
    ax3.set_ylim([0, 128])
    ax4.imshow(gt.cpu().detach().numpy()[0, 0, :, :])
    ax4.set_title('Ground truth')
    ax4.set_xlim([0, 128])
    ax4.set_ylim([0, 128])
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{chk_Path}/{epoch}.png', dpi=300)
    time.sleep(0.1)
#============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args(['--weights-file', 'P:/SR/SRCNN/output/best.pth',
                                            '--image-file', 'P:/SR/TestImages/'])

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    i = 0

    psnr_values = []
    SSIM_values = []
    lpips_values = []
    loss_fn = lpips.LPIPS(net='alex').to(device)
    model.eval()
    for x, y in Test_dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            input = torch.nn.functional.interpolate(x, scale_factor=inp['scale'], align_corners=True, mode='bicubic')
            sr = model(input).clamp(0.0, 1.0)

        # ======================================
        # PSNR
        # ======================================

            PSNR = calc_psnr(y, sr)
            print(PSNR.item())
            psnr_values.append(PSNR.item())

        # ======================================
        # SSIM
        # ======================================

            sr_np = sr.cpu().numpy()[0][0]
            y_np = y.cpu().numpy()[0][0]
            SSIM = ssim(sr_np, y_np, data_range=1.0)
            print(SSIM)
            SSIM_values.append(SSIM)

        # ======================================
        # LPIPS
        # ======================================

            sr_lp = sr.repeat(1, 3, 1, 1)  # Convert to 3 channels
            y_lp = y.repeat(1, 3, 1, 1)
            sr_lp = sr_lp * 2 - 1  # expects images in [-1 , +1] but network outputs [0 , 1]
            y_lp = y_lp * 2 - 1
            lpips_value = loss_fn(sr_lp, y_lp)
            lpips_values.append(lpips_value.item())

            i+=1

print("\n==============================")
print("SRCNN RESULTS")
print("==============================")

print("Average PSNR:", np.mean(psnr_values))
print("Std PSNR:", np.std(psnr_values))
print("Max PSNR:", np.max(psnr_values))
print("Min PSNR:", np.min(psnr_values))

print("Average SSIM:", np.mean(SSIM_values))

print("Average LPIPS:", np.mean(lpips_values))
#=================================================================================================================
"""
END
"""
#=================================================================================================================