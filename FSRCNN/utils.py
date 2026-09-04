import torch
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

#==================================================================================================================================================================================
"""
SECTION:  Utils (Definition  Necessary Functions)
"""
# *****************************************************  Function: This Function Calculate PSNR between two images. *****************************************************
# def calc_psnr(img1, img2):
#     return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_psnr (img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore, PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

# *********************************************************  Function: This Class is for Calculate AverageMeter. *********************************************************
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# *******************************************************  Function: This function is used for plotting Loss animation. ******************************************************
def plot_loss_animation(iteration_list, loss, fig=None, chkp_Path='.'):
    # plots an animation of loss during training/validation
    def create_fig():
        # enable interactive mode
        plt.ion()
        # fig = plt.figure(figsize=(10,5))
        fig, ax1 = plt.subplots(1)
        fig.set_figheight(3)
        fig.set_figwidth(3)
        fig.suptitle('Loss during Training')
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Loss")
        ax1.plot(iteration_list, loss, label="train")
        # ax1.legend()
        plt.tight_layout()
        return fig

    if len(iteration_list) == 1:
        fig = create_fig()
    else:
        # updating the value of x and y
        fig.axes[0].lines[0].set_xdata(iteration_list)
        fig.axes[0].lines[0].set_ydata(loss)
        # recompute the ax.dataLim
        fig.axes[0].relim()
        # update ax.viewLim using the new dataLim
        fig.axes[0].autoscale_view()
        # plt.legend()

    # re-drawing the figure
    fig.canvas.draw()
    # to flush the GUI events
    fig.canvas.flush_events()
    plt.tight_layout()
    time.sleep(0.1)

    fig.savefig(f'{chkp_Path}/loss.png', dpi=300)
    np.save(f'{chkp_Path}/loss.npy', loss)
    np.save(f'{chkp_Path}/iteration_list.npy', iteration_list)
    return fig

# ************************************  Function: This Function is used for Plotting Input Image and Reconstructed Image and Ground truth Image. ************************************
def plot_imgs(inp, outp, gt, epoch, chkp_Path='.'):
    # input image, reconstructed image, ground truth
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(f'Epoch = {epoch}')
    ax1.imshow(inp.cpu().detach().numpy()[0, 0, :, :])
    ax1.set_title('Input')
    ax2.imshow(outp.cpu().detach().numpy()[0, 0, :, :])
    ax2.set_title('Reconstructed')
    ax3.imshow(gt.cpu().detach().numpy()[0, 0, :, :])
    ax3.set_title('Ground truth')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{chkp_Path}/{epoch}.png', dpi=300)
    plt.close(fig)
    # time.sleep(0.1)

# **************************************************  Function: This Function for Creating a PDF  File with Several Pages. ****************************************************
def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
#=================================================================================================================================================================================