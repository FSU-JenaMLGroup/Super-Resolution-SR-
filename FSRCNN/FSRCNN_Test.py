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
from models import FSRCNN
from skimage.metrics import structural_similarity as ssim
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
        'outputs-dir': 'D:/Muturi D/SR/FSRCNN/output/',
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

    model = FSRCNN(scale_factor=inp['scale']).to(device)  # GPU
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
# from skimage.metrics import structural_similarity as ssim
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

def plot_imgs_fsrcnn(inp, outp, gt, epoch):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    fig.suptitle(f'Sample = {epoch}')
    ax1.imshow(inp.cpu().detach().numpy()[0, 0, :, :], cmap='gray')
    ax1.set_title('Input (LR)')
    ax2.imshow(outp.cpu().detach().numpy()[0, 0, :, :], cmap='gray')
    ax2.set_title('Reconstructed (SR)')
    ax3.imshow(gt.cpu().detach().numpy()[0, 0, :, :], cmap='gray')
    ax3.set_title('Ground Truth (HR)')
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent / f'Image Output/{epoch}_FSRCNN.png'), dpi=300)
    plt.close(fig)
    time.sleep(0.1)


def plot_imgs_lr(inp, epoch):
    plt.imshow(inp[0][0])
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent / f'{epoch}_LR.png'), dpi=300)
    time.sleep(0.1)

def plot_imgs_hr(inp, epoch):
    plt.imshow(inp[0][0])
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent / f'{epoch}_HR.png'), dpi=300)
    time.sleep(0.1)
#============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args(['--weights-file', 'D:/Muturi D/SR/FSRCNN/output/best.pth',
                                            '--image-file', 'D:/Muturi D/SR/FSRCNN/output/TestImages/'])

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=args.scale).to(device)

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
            sr = model(x).clamp(0.0, 1.0)

        # ======================================
        # PSNR
        # ======================================

            PSNR = calc_psnr(y, sr)
            print(PSNR.item())
            psnr_values.append(PSNR.item()) #.item() to convert from tensor from list to scalar
            
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
            sr_lp = sr_lp * 2 - 1  # expects images in [-1 , +1] but network outpus [0 , 1] 
            y_lp = y_lp * 2 - 1 # so we need to scale them to [-1 , +1]
            lpips_value = loss_fn(sr_lp, y_lp)
            lpips_values.append(lpips_value.item())
        

            i+=1

            if i % 100 == 0:
                plot_imgs_fsrcnn(x, sr, y, i)   # FSRCNN_Test.py
df = pd.DataFrame({
    "Sample": list(range(len(psnr_values))),
    "PSNR": psnr_values,
    "SSIM": SSIM_values,
    "LPIPS": lpips_values,
})
df.to_csv(f'{chk_Path}/results.csv', index=False)
print(f"Results saved to {chk_Path}/results.csv")
            
print("\n==============================")
print("FSRCNN RESULTS")
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