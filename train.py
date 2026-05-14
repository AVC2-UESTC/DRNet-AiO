import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from net.DRNet_arch import DRNet
from options import options as opt
from torch.utils.data import DataLoader
from utils.dataset_utils import PromptTrainDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.schedulers import LinearWarmupCosineAnnealingLR
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger

class DRNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = DRNet(num_experts=4, input_size=128, expert_dim=5, num_blocks=[4,6,6,8])
        self.loss_fn  = nn.L1Loss()
        

    def rgb2ycbcr_pt(self, img, y_only=False):
        """Convert RGB images to YCbCr images (PyTorch version).

        It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
        https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

        Args:
            img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
            y_only (bool): Whether to only return Y channel. Default: False.

        Returns:
            (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
        """
        if y_only:
            weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
            out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
        else:
            weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
            bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
            out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

        out_img = out_img / 255.
        return out_img

    def calculate_psnr_pt(self, img, img2, crop_border=0, test_y_channel=False, **kwargs):
        """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

        Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: PSNR result.
        """

        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

        if crop_border != 0:
            img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

        if test_y_channel:
            img = self.rgb2ycbcr_pt(img, y_only=True)
            img2 = self.rgb2ycbcr_pt(img2, y_only=True)

        img = img.to(torch.float64)
        img2 = img2.to(torch.float64)

        mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
        return 10. * torch.log10(1. / (mse + 1e-8))
    
    def forward(self,x, prompt):
        return self.net(x, prompt)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch, prompts) = batch
        # print(prompts)
        restored = self.net(degrad_patch, prompts)

        loss = self.loss_fn(restored, clean_patch)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-4)

        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=100)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }



def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir, every_n_train_steps=5000, save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=False, num_workers=opt.num_workers)
    
    model = DRNetModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy='ddp_find_unused_parameters_true',logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)

if __name__ == '__main__':
    main()



