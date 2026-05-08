import os
import logging
import argparse
import numpy as np

import torch

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from net.DRNet_arch import DRNet, DRMLP
from utils.data_util import crop_HWC_img
from utils.metric_util import AverageMeter
from utils.util import mkdir, setup_logger
from torchvision.transforms import ToTensor
from utils.tensor_op import save_image_tensor
from metrics.psnr_ssim import compute_psnr_ssim

class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15
        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        self.clean_ids = []
        name_list = os.listdir(self.args.data_path)
        self.clean_ids += [self.args.data_path + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)

    def set_dataset(self, dataset):
        # self.task_idx = self.dataset_dict[dataset]
        self._init_clean_ids()

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch
    
    def _edgeComputation(self,x):
        x_diffx = np.abs(x[:,1:,:] - x[:,:-1,:])
        x_diffy = np.abs(x[1:,:,:] - x[:-1,:,:])

        y = np.zeros_like(x)
        y[:,1:,:] += x_diffx
        y[:,:-1,:] += x_diffx
        y[1:,:,:] += x_diffy
        y[:-1,:,:] += x_diffy
        y = np.sum(y,2)/3
        y /= 4
        return y[:,:,None].astype(np.float32)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_HWC_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=32)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)

        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1, 'deblur':2, 'enhance':3}
        self.toTensor = ToTensor()

        self.set_dataset(args.task)

    def _init_input_ids(self):
        self.ids = []
        name_list = os.listdir(self.args.data_path + 'input/')
        self.ids += [self.args.data_path + 'input/' + id_ for id_ in name_list]
        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        elif self.task_idx == 2:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 3:
            gt_name = degraded_name.replace("input", "target")
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def _add_gaussian_noise(self, clean_patch, sigma=25):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch

    def _edgeComputation(self,x):
        x_diffx = np.abs(x[:,1:,:] - x[:,:-1,:])
        x_diffy = np.abs(x[1:,:,:] - x[:-1,:,:])

        y = np.zeros_like(x)
        y[:,1:,:] += x_diffx
        y[:,:-1,:] += x_diffx
        y[1:,:,:] += x_diffy
        y[:-1,:,:] += x_diffy
        y = np.sum(y,2)/3
        y /= 4
        return y[:,:,None].astype(np.float32)

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = np.array(Image.open(degraded_path).convert('RGB'))
        if clean_path is not None:
            clean_img = np.array(Image.open(clean_path).convert('RGB'))
            clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        else:
            clean_img = []
            degraded_img = self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


def test_Denoise(net, dataset, task="CBSD68", sigma=15, save_img=False, prompt=None):
    logger = logging.getLogger('base')
    output_path = opt.output_path + 'denoise/' + str(sigma) + '/'
    mkdir(output_path)
    window_size = 64
    
    dataset.set_dataset(task)
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            prompt = prompt.cuda()
            _, _, h_old, w_old = clean_patch.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            degrad_patch = torch.cat([degrad_patch, torch.flip(degrad_patch, [2])], 2)[:, :, :h_old + h_pad, :]
            degrad_patch = torch.cat([degrad_patch, torch.flip(degrad_patch, [3])], 3)[:, :, :, :w_old + w_pad]
            restored = net(degrad_patch, prompt)
            restored = restored[..., :h_old, :w_old]

            if type(restored) == list:
                restored = restored[0]
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            if save_img:
                save_image_tensor(restored, output_path + clean_name[0] + '.png')

        logger.info("Deonise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain",save_img=False, prompt=None):
    logger = logging.getLogger('base')
    output_path = opt.output_path + task + '/'
    mkdir(output_path)
    window_size = 64

    psnr = AverageMeter()
    ssim = AverageMeter()

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            if not isinstance(clean_patch, list):
                degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            else:
                degrad_patch = degrad_patch.cuda()
            prompt = prompt.cuda()
            _, _, h_old, w_old = degrad_patch.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            degrad_patch = torch.cat([degrad_patch, torch.flip(degrad_patch, [2])], 2)[:, :, :h_old + h_pad, :]
            degrad_patch = torch.cat([degrad_patch, torch.flip(degrad_patch, [3])], 3)[:, :, :, :w_old + w_pad]
            restored = net(degrad_patch, prompt)
            restored = restored[..., :h_old, :w_old].clamp(0,1)
            if type(restored) == list:
                restored = restored[0]

            N = degrad_patch.shape[0]

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            if save_img:
                save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        logger.info("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--data_path', type=str, default="enhance/lol/", help='save path of test low-light images')
    parser.add_argument('--task', type=str, default="derain", help='["derain","denoise","dehaze","enhance","deblur"]')
    
    parser.add_argument('--output_path', type=str, default="results/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="", help='checkpoint save path')
    parser.add_argument('--log_path', type=str, default="results/log", help='checkpoint save path')
    opt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(opt.cuda)

    if opt.task == 'denoise':
        testset = DenoiseTestDataset(opt)
    else:
        testset = DerainDehazeDataset(opt)

    candidates = ['denoise', 'dehaze', 'derain', 'deblur', 'enhance']
    base_prompts = {}
    for idx, key in enumerate(candidates):
        labels = [i for i in range(len(candidates))]
        labels = torch.tensor(labels)
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=len(candidates))
        one_hot_labels = one_hot_labels.float()
        base_prompts[key] = one_hot_labels[idx:idx+1,:]

    setup_logger('base', opt.log_path, level=logging.INFO, phase='test', screen=True, tofile=False)
    logger = logging.getLogger('base')

    if opt.task == 'denoise':
        # denoise
        net_denoise = DRNet(num_experts=4, input_size=128, expert_dim=5, num_blocks=[4,6,6,8])
        net_denoise.load_state_dict(torch.load(opt.ckpt_path))
        for m in net_denoise.modules():
            if isinstance(m, DRMLP):
                m.switch_to_deploy(base_prompts[opt.task])
        net_denoise = net_denoise.cuda()
        net_denoise.eval()
        test_Denoise(net_denoise, testset, task="CBSD68", sigma=25, prompt=base_prompts[opt.task])
    else:
        # derain 
        net_derain = DRNet(num_experts=4, input_size=128, expert_dim=5, num_blocks=[4,6,6,8])
        net_derain.load_state_dict(torch.load(opt.ckpt_path))
        for m in net_derain.modules():
            if isinstance(m, DRMLP):
                m.switch_to_deploy(base_prompts[opt.task])
        net_derain = net_derain.cuda()
        net_derain.eval()
        test_Derain_Dehaze(net_derain, testset, task=opt.task, prompt=base_prompts[opt.task])