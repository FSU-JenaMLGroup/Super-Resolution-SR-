"""
IN THE NAME OF GOD

Subject: Converts LR images to HR images

@Author: Pouya Sadeghi
"""
#=================================================================================================================================================================================
"""
SECTION : Import Necessary Packages (Libraries)
"""
from datasets import Image_Dataset
from utils import calc_psnr, AverageMeter, plot_loss_animation, plot_imgs, save_multi_image
from models import FSRCNN
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import copy
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time
from datetime import timedelta, datetime
import pandas as pd
import torch; torch.manual_seed(0)
import torch.distributions
from torchsummary import summary
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
# mpl.style.use('grayscale')
mpl.style.use('classic')
from tabulate import tabulate
from skimage.metrics import structural_similarity as ssim_metric
import lpips
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.max_open_warning': 0})
#==================================================================================================================================================================================
"""
SECTION: Train Model
"""
if __name__ == '__main__':
    inp = {
        'train_h5': "D:/Muturi D/SR/Dataset/train.h5",
        'val_h5':   "D:/Muturi D/SR/Dataset/val.h5",
        'test_h5':  "D:/Muturi D/SR/Dataset/test.h5",
        'LRsample_shape': (1, 32, 32),  # (Channel, Height, Width)
        'HRsample_shape': (1, 128, 128),  # (Channel, Height, Width)
        'outputs-dir': 'D:/Muturi D/SR/FSRCNN/output/',
        'scale': 4,
        'learning_rate': 1e-3,
        'batch_size': 512,
        'num_epochs': 100,
        'num_workers': 8,
        'seed': 123,
        'optimizer': 'Adam',
    }

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device:  {device}')

    torch.manual_seed(inp['seed'])

    # For Test/Continue_Train chkPath_load at below should be set too.
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------
    chkp_Path = str(pathlib.Path(__file__).parent / f'_results/FSRCNN_{inp["optimizer"]}_b{inp["batch_size"]}_lr{inp["learning_rate"]}')
    pathlib.Path(chkp_Path).mkdir(parents=True, exist_ok=True)

    t = time.time()

    model = FSRCNN(scale_factor=inp['scale']).to(device)  # GPU
    # model = FSRCNN().to(device)  # GPU
    summary(model, inp['LRsample_shape'], device="cuda")
    print(model)
    inp['model'] = str(model)
    criterion = nn.MSELoss().to(device)
    loss_lpips = lpips.LPIPS(net='alex').to(device)

    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'learning_rate': inp['learning_rate'] * 0.1}
    ], lr=inp['learning_rate'])
    # {'params': model.last_part.parameters()}])

    from colorama import Fore

    print(Fore.YELLOW + """" **************************************** Loading dataset **************************************** """)

    with h5py.File(inp['train_h5'], 'r') as hf:
        Train_y = np.array(hf['HR'])
        Train_x = np.array(hf['LR'])

    with h5py.File(inp['val_h5'], 'r') as hf:
        Eval_y = np.array(hf['HR'])
        Eval_x = np.array(hf['LR'])

    with h5py.File(inp['test_h5'], 'r') as hf:
        Test_y = np.array(hf['HR'])
        Test_x = np.array(hf['LR'])

    Train_Dataset = Image_Dataset(Train_x, Train_y, inp['LRsample_shape'], inp['HRsample_shape'])
    print('TrainDB:', Train_Dataset.img_x.shape[0])
    train_dataloader = DataLoader(dataset=Train_Dataset,
                                  batch_size=inp['batch_size'],
                                  shuffle=True,
                                  num_workers=inp['num_workers'],
                                  pin_memory=True,
                                  persistent_workers=True,
                                  prefetch_factor=4)

    Eval_Dataset = Image_Dataset(Eval_x, Eval_y, inp['LRsample_shape'], inp['HRsample_shape'])
    print('TrainDB:', Eval_Dataset.img_x.shape[0])
    Eval_dataloader = DataLoader(dataset=Eval_Dataset,
                                 batch_size=inp['batch_size'],
                                 num_workers=inp['num_workers'],
                                 pin_memory=True,
                                 persistent_workers=True,
                                 prefetch_factor=4)

    Test_Dataset = Image_Dataset(Test_x, Test_y, inp['LRsample_shape'], inp['HRsample_shape'])
    print('TestDB:', Test_Dataset.img_x.shape[0])
    Test_dataloader = DataLoader(dataset=Test_Dataset,
                                 batch_size=inp['batch_size'],
                                 num_workers=inp['num_workers'],
                                 pin_memory=True,
                                 persistent_workers=True,
                                 prefetch_factor=4)

    # ============================================================================
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    elapsed = time.time() - t
    print('Dataset loading time=', str(timedelta(seconds=elapsed)))

    start_epoch = 0

    print(Fore.GREEN + """" **************************************** Start of Training Network **************************************** """)
    iteration_list, loss_list = [], []
    fig_params = None
    Epoch = np.arange(0, 1000)
    Loss_Training = np.zeros(1000)
    PSNR = np.zeros(1000)
    SSIM = np.zeros(1000)
    LPIPS = np.zeros(1000)
    Epoch_elapsed_time = np.zeros(1000)
    t_total = time.time()

    for epoch in range(start_epoch, inp['num_epochs']):
        loss_avg = 0
        t_epoch = time.time()
        model.train()
        epoch_losses = AverageMeter()

        print(Fore.GREEN + '\nThe Information of epoch {} as Follows: '.format(epoch))
        with tqdm(total=(len(Train_Dataset) - len(Train_Dataset) % inp['batch_size']), position=0, leave=True) as t:
            t.set_description('epoch: {}/{}'.format(epoch, inp['num_epochs'] - 1))

            total_loss = torch.tensor(0.0, device=device)
            for x, y in train_dataloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                # attention_matrix = torch.randint(self._initialize_weights.size())
                # model.fc.weight = model.fc.weight * attention_matrix
                y_preds = model(x)
                loss = criterion(y_preds, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach()
                t.update(len(x))
            # scheduler.step()
            torch.save(model.state_dict(), os.path.join(inp['outputs-dir'], 'epoch_{}.pth'.format(epoch)))
            loss_avg = total_loss.item() / len(train_dataloader)
            loss_list.append(loss_avg)
            iteration_list.append(epoch)

            # Plot loss animaiton
            fig_params = plot_loss_animation(iteration_list, loss_list, fig_params, chkp_Path)

            # plot reconstructed
            if epoch % 10 == 0:
                plot_imgs(x, y_preds, y, epoch, chkp_Path)

            # Print Loss
            # print(f'Epoch: {epoch}, Loss_Training: {loss_avg:.8f}')

            # print(tabulate([['epoch', epoch], ['loss_avg', loss_avg]], headers=['Epoch', 'Loss_Training']))
            print(tabulate([["\nEpoch", "\nLoss_Training"], ["epoch", epoch], ["loss_avg", loss_avg]], headers = "firstrow"))
            Loss_Training[epoch] = loss_avg
            # print(f'min: {y_preds.min().item()} max: {y_preds.max().item()}')
            elapsed = time.time() - t_epoch
            print('Epoch elapsed time=', str(timedelta(seconds=elapsed)))
            Epoch_elapsed_time[epoch] = elapsed

            model.eval()
            epoch_psnr = AverageMeter()
            epoch_ssim_eval = AverageMeter()
            epoch_lpips_eval = AverageMeter()
            for x, y in Eval_dataloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.no_grad():
                    preds = model(x).clamp(0.0, 1.0)

                epoch_psnr.update(calc_psnr(preds, y), len(x))
                sr_np = preds.cpu().numpy()[0,0]
                gt_np = y.cpu().numpy()[0,0]
                ssim_value = ssim_metric(sr_np, gt_np, data_range=1.0)
                epoch_ssim_eval.update(ssim_value, len(x))

                pred_lp = preds.repeat(1, 3, 1, 1)
                gt_lp = y.repeat(1, 3, 1, 1)
                pred_lp = pred_lp * 2 - 1
                gt_lp = gt_lp * 2 - 1
                lpips_score = loss_lpips(pred_lp, gt_lp).mean()
                epoch_lpips_eval.update(lpips_score.item(), len(x))

            # print('Eval PSNR: {:.2f}'.format(epoch_psnr.avg))
            table = [["PSNR", epoch_psnr.avg], ["SSIM", epoch_ssim_eval.avg], ["LPIPS", epoch_lpips_eval.avg]]
            print(tabulate(table, headers=["\nMetrics for Validation", "\nValue of Metrics"]))
            # print(tabulate([["\nPSNR"], ["Eval PSNR", epoch_psnr.avg], ["loss_avg", loss_avg]], headers = "firstrow"))

            model.eval()
            epoch_psnr_test = AverageMeter()
            for x, y in Test_dataloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.no_grad():
                    preds_test = model(x).clamp(0.0, 1.0)

                epoch_psnr_test.update(calc_psnr(preds_test, y), len(x))

            table = [["PSNR", epoch_psnr_test.avg]]
            print(tabulate(table, headers=["\nPeak Signal-to-Noise ratio for Test", "\nValue of PSNR"]))

            print(Fore.BLUE + '==================================================================================================================================')
            PSNR[epoch] = epoch_psnr.avg
            SSIM[epoch] = epoch_ssim_eval.avg
            LPIPS[epoch] = epoch_lpips_eval.avg
            # print("Image similarity: {}".format(score))

            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

    print(Fore.MAGENTA + 'Best Epoch: {}'.format(best_epoch))
    print(Fore.YELLOW + '==========================================')
    print(Fore.RED + 'Best PSNR: {}'.format(best_psnr))
    print(Fore.YELLOW + '==========================================')
    torch.save(best_weights, os.path.join(inp['outputs-dir'], 'best.pth'))
    # Save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'{chkp_Path}/epoch{epoch}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pt')
    # save loss vs epoch image
    # plt.savefig(f'{chkp_Path}/loss.png')
    # np.savetxt("PSNR.csv", PSNR, delimiter=",", header='PSNR')
    # np.savetxt("Loss_Training.csv", Loss_Training, delimiter=",", header='Loss_Training')
    df = pd.DataFrame({"Epoch": Epoch, "Loss_Training": Loss_Training, "PSNR": PSNR, "SSIM": SSIM, "LPIPS": LPIPS, "Epoch_elapsed_time": Epoch_elapsed_time})
    df.to_csv(str(pathlib.Path(__file__).parent / "output.csv"), index=False)

    save_multi_image(f'{chkp_Path}/loss.pdf')
    time_training_network = time.time() - t_total
    print(f'The Time of Train Model: {time_training_network}')
#=================================================================================================================================================================================
"""
END
"""
#==================================================================================================================================================================================