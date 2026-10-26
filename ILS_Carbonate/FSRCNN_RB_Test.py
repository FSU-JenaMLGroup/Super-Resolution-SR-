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
from models import FSRCNN_RB
import numpy as np
import h5py
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
#===================================================================================================================================================================================
if __name__ == '__main__':
    inp = {
        'test_h5': "D:/Muturi D/SR/ILS45/Data/test_ILS.h5",
        'LRsample_shape': (1, 32, 32),  # (Channel, Height, Width)
        'HRsample_shape': (1, 128, 128),  # (Channel, Height, Width)
        'outputs-dir': r'D:/Muturi D/SR/ILS45/FSRCNN-RB/output',
        'scale': 4,
        'learning_rate': 1e-3,
        'batch_size': 5,
        'num_epochs': 100,
        'num_workers': 8,
        'seed': 123,
        'optimizer': 'Adam',
    }

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device:  {device}')

    t = time.time()

    model = FSRCNN_RB(scale_factor=inp['scale']).to(device)  # GPU
    # model = FSRCNN_RB().to(device)  # GPU
    summary(model, inp['LRsample_shape'], device="cuda")
    print(model)
    inp['model'] = str(model)

    from colorama import Fore

    print(Fore.YELLOW + """" **************************************** Loading dataset **************************************** """)

    with h5py.File(inp['test_h5'], 'r') as hf:
        Test_y = np.array(hf['HR'])
        Test_x = np.array(hf['LR'])

    Test_Dataset = Image_Dataset(Test_x, Test_y, inp['LRsample_shape'], inp['HRsample_shape'], device)
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
chk_Path = f'./Image Output/'
from skimage.metrics import structural_similarity as ssim_fn
import lpips
import pandas as pd

#============================================================
def plot_imgs_test(inp, outp, gt, epoch):
    # input image, reconstructed image, ground truth
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(f'Sample = {epoch}')
    ax1.imshow(inp.cpu().detach().numpy()[0, 0, :, :])
    ax1.set_title('Input')
    ax2.imshow(outp.cpu().detach().numpy()[0, 0, :, :])
    ax2.set_title('Reconstructed')
    ax3.imshow(gt.cpu().detach().numpy()[0, 0, :, :])
    ax3.set_title('Ground truth')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{chk_Path}/{epoch}.png', dpi=300)
    time.sleep(0.1)
    plt.close(fig)
#============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args(['--weights-file', r'D:/Muturi D/SR/ILS45/FSRCNN-RB/output\best.pth',
                                            '--image-file', r'D:/Muturi D/SR/ILS45/Data\\'])

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN_RB(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    psnr_list, ssim_list, lpips_list = [], [], []
    i = 0

    model.eval()
    for x, y in Test_dataloader:
        with torch.no_grad():
            sr = model(x).clamp(0.0, 1.0)

            psnr_list.append(calc_psnr(y, sr).item())

            sr_np = sr[0, 0].cpu().numpy()
            y_np  = y[0, 0].cpu().numpy()
            ssim_list.append(ssim_fn(sr_np, y_np, data_range=1.0))

            sr_3ch = sr.repeat(1, 3, 1, 1) * 2 - 1
            y_3ch  = y.repeat(1, 3, 1, 1)  * 2 - 1
            lpips_list.append(loss_fn_lpips(sr_3ch, y_3ch).item())

            i += 1

    print(f'\n--- FSRCNN-RB Test Results ({i} samples) ---')
    print(f'PSNR  : {np.mean(psnr_list):.4f} +/- {np.std(psnr_list):.4f} dB')
    print(f'SSIM  : {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}')
    print(f'LPIPS : {np.mean(lpips_list):.4f} +/- {np.std(lpips_list):.4f}')
#===================================================================================================================================================================================
"""
END
"""
#===================================================================================================================================================================================