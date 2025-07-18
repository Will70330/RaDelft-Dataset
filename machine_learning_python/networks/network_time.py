import time
import matplotlib.pyplot as plt
#from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

# add parent directory to path
import sys

import torch.utils.checkpoint

# apend the absolute path of the parent directory
sys.path.append(sys.path[0] + "/..")
import scipy.io
import re
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_preparation import data_preparation
import torch.nn as nn
from pytorch_lightning.callbacks import RichProgressBar
from loaders.rad_cube_loader import RADCUBE_DATASET_TIME
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.models as models
from utils.compute_metrics import compute_metrics_time, compute_pd_pfa
import wandb

run = None

OUT_CLASSES = 34  # 44 elevation angles
IN_CHANNELS = 64  # output of the ReduceDNet

# ToDO: Check if goes faster with this:
torch.set_float32_matmul_precision('medium')
# mp.set_start_method("spawn", force=True)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# this gets rid of the Doppler dimension to get a "2D image".
# We go from B*C*D*H*W to B*C*H*W, H and W are ranges and azimuths
class DopplerEncoder(nn.Module):
    def __init__(self, use_groupNorm=False):
        super(DopplerEncoder, self).__init__()

        # Parameters
        in_channels = 2  # Elevation and power
        out_channel_1 = 32  # this can be changed to any number
        out_channel_2 = IN_CHANNELS  # this can be changed to any number, will be the input of next model
        kernel_size1 = (5, 3, 3)
        stride1 = (4, 1, 1)  # (D, H, W), 1/4 of the original size
        padding1 = (2, 1, 1)
        kernel_size2 = (4, 3, 3)
        stride2 = (4, 1, 1)  # (D, H, W), 1/4 of the original size
        padding2 = (1, 1, 1)

        pool_kernel = (8, 1, 1)
        pool_stride = (8, 1, 1)

        # Step 1: Convolution parameters to reduce from 240 to 60
        self.conv1 = nn.Conv3d(in_channels, out_channel_1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.norm1 = nn.BatchNorm3d(32) if not use_groupNorm else nn.GroupNorm(num_groups=8, num_channels=32)
        self.relu1 = nn.ReLU()

        # Step 2: Convolution parameters to reduce from 60 to 15
        self.conv2 = nn.Conv3d(out_channel_1, out_channel_2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.norm2 = nn.BatchNorm3d(64) if not use_groupNorm else nn.GroupNorm(num_groups=8, num_channels=64)
        self.relu2 = nn.ReLU()

        # Step 3: Pooling parameters to reduce from 15 to 1
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        # Apply first convolution
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Apply second convolution
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        # Apply max pooling
        x = self.pool(x)

        return x.squeeze(2)  # Remove the D dimension


class NeuralNetworkRadarDetector(pl.LightningModule):

    def __init__(self, arch, encoder_name, params, in_channels, out_classes, lr=3e-4, warmup_epochs=10, use_groupNorm=False, alpha=1.0, **kwargs):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_hyperparameters()
        self.DopplerReducer = DopplerEncoder(use_groupNorm=use_groupNorm)
        self.alpha = alpha

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=None, 
            in_channels=in_channels, 
            classes=out_classes, 
            **kwargs
        )

        # # Convert every BN inside the ResNet to SyncBN so BN layers
        # # see the *world-size* batch, not the 1–2 samples on each GPU.
        # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Enable gradient checkpointing for memory efficiency (optional)
        # Only use if you're running out of memory
        # self._enable_gradient_checkpointing()


        # Temporal smoothing layers
        kernel_size = (3, 5, 7)

        self.conv1 = nn.Conv3d(3, 6, kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(6, 12, kernel_size=kernel_size, padding='same')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(12, 24, kernel_size=kernel_size, padding='same')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(24, 12, kernel_size=kernel_size, padding='same')
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv3d(12, 6, kernel_size=kernel_size, padding='same')
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv3d(6, 3, kernel_size=kernel_size, padding='same')

        self.dropout = nn.Dropout3d(p=0.3)  # Increased dropout for stronger regularization

        # Initialize temporal smoothing layers
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        self.counter = 0
        self.params = params

    def forward(self, image):
        # Get single frames
        [image1, image2, image3] = torch.chunk(image, 3, axis=1)

        image1 = image1.squeeze(1)
        image2 = image2.squeeze(1)
        image3 = image3.squeeze(1)

        # DopplerReduce Nets
        image1 = self.DopplerReducer(image1)
        image2 = self.DopplerReducer(image2)
        image3 = self.DopplerReducer(image3)

        image1 = image1.float()
        image2 = image2.float()
        image3 = image3.float()

        # Segmentation Model
        mask1 = self.model(image1)
        mask2 = self.model(image2)
        mask3 = self.model(image3)

        mask = torch.stack([mask1, mask2, mask3], 4)
        mask = torch.permute(mask, [0, 4, 1, 2, 3])

        # Temporal smoothing
        mask = self.conv1(mask)
        mask = self.relu1(mask)
        mask = self.dropout(mask)
        mask = self.conv2(mask)
        mask = self.relu2(mask)
        mask = self.dropout(mask)
        mask = self.conv3(mask)
        mask = self.relu3(mask)
        mask = self.dropout(mask)
        mask = self.conv4(mask)
        mask = self.relu4(mask)
        mask = self.dropout(mask)
        mask = self.conv5(mask)
        mask = self.relu5(mask)
        mask = self.dropout(mask)
        mask = self.conv6(mask)

        return mask

    def shared_step(self, batch, stage):
        # Load input and GT
        RAD_cube = batch[0]  # range azimuth doppler cube, the input to the network
        gt_lidar_cube = batch[1]  # TODO here we have to get the gt_cloud and convert it to a mask that fits our loss
        # item_params = batch[2]

        # Run the network
        RAE_Cube = self.forward(RAD_cube)  # output is a binary dense mask of the cube in RAE format: range, azimuth, elevation

        loss = data_preparation.radarcube_lidarcube_loss_time(RAE_Cube, gt_lidar_cube, self.params)

        # Add L1 regularization for additional control over overfitting
        l1_lambda = 1e-5
        l1_reg = sum(p.abs().sum() for p in self.parameters() if p.requires_grad)
        loss = loss + l1_lambda * l1_reg

        if stage == 'valid':
            radar_cube_out = RAE_Cube.sigmoid().squeeze().cpu().detach().numpy()
            radar_cube_out = radar_cube_out > 0.5
            if radar_cube_out.ndim == 5:
                radar_cube_out = radar_cube_out[:, :, :, :-12, 8:-8]
            else:
                radar_cube_out = radar_cube_out[:, :, :-12, 8:-8]
            pd, pfa = compute_pd_pfa(gt_lidar_cube.cpu().detach().numpy(), radar_cube_out)

            return loss, pd, pfa

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        actual_batch_size = batch[0].shape[0] # This should be 1 / 2
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=actual_batch_size)
        
        # Log to wandb less frequently to reduce overhead
        if batch_idx % 10 == 0:
            run.log({'train_loss': loss.item(), 'lr': self.optimizers().param_groups[0]['lr']})
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pd, pfa = self.shared_step(batch, "valid")
        
        actual_batch_size = batch[0].shape[0] # This should be 1 / 2
        self.log_dict({'val_loss': loss, 'val_pd': pd, 'val_pfa': pfa, },
                      on_step=False, on_epoch=True, prog_bar=True,
                      logger=True, batch_size=actual_batch_size)
        
        # Log to wandb less frequently to reduce overhead
        if batch_idx % 10 == 0:
            run.log({'val_loss': loss.item(), 'val_pd': pd.item(), 'val_pfa': pfa.item(), 'lr': self.hparams.lr})
        
        return loss

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_after_backward(self):
        # Log gradient norms
        total_norm = 0
        param_count = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** 0.5
        self.log('grad_norm', total_norm, on_step=True, on_epoch=False)
        
        # Log to wandb less frequently
        if self.global_step % 10 == 0:
            run.log({"grad_norm": total_norm})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=1e-4)  # Increased weight decay for stronger L2 regularization

        # Warmup scheduler - crucial for stable training with batch_size=1
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01,  # Start at 1% of target LR (more conservative than 0.005)
            total_iters=self.hparams.warmup_epochs
        )
        
        # Cosine annealing with warm restarts
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,      # First restart after 10 epochs
            T_mult=2,    # Double the period after each restart (10, 20, 40, ...)
            eta_min=1e-7 # Minimum learning rate
        )
        
        # Combine warmup and cosine schedules
        base_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_epochs]
        )
        
        # Add ReduceLROnPlateau on top for additional adaptation
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,      # Less aggressive reduction since we have cosine
            patience=8,      # Higher patience to let cosine schedule work
            verbose=True,
            min_lr=1e-8      # Lower than cosine min_lr
        )

        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

        # Return both schedulers - PyTorch Lightning will handle them correctly
        return [optimizer], [
            {
                'scheduler': base_scheduler
            }, 
            {
                'scheduler': plateau_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }      
        ]


# main function
def main(params, resume_checkpoint=None):
    # Start a new wandb run to track this script.
    global run
    checkpointt_directory = "checkpoints-resnet152"
    if resume_checkpoint:
        ckpt_dir = os.path.dirname(resume_checkpoint)
    else: 
        ckpt_dir = checkpointt_directory
        
    run_id_file = os.path.join(ckpt_dir, "wandb_run_id.txt")
    if resume_checkpoint and os.path.exists(run_id_file):
        # Load old run ID and re-attach
        with open(run_id_file, "r") as f:
            old_id = f.read().strip()
        run = wandb.init(
            entity="will_70330",
            project="RISS-Research-RaDelft",
            config={
                "architecture": "ResNet152-regularized-deep-temporal",
                "dataset": "RaDelft",
                "epochs": 80,
            },
            id=old_id,
            resume="allow"
        )
    else:
        # Brand new run
        run = wandb.init(
            entity="will_70330",
            project="RISS-Research-RaDelft",
            config={
                "architecture": "ResNet152-regularized-deep-temporal",
                "dataset": "RaDelft",
                "epochs": 80,
            }
        )
        # check for exisiting checkpoint folder and write run ID
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(run_id_file, 'w') as f:
            f.write(run.id)

    # Create training and validation datasets
    train_dataset = RADCUBE_DATASET_TIME(mode='train', params=params)
    val_dataset = RADCUBE_DATASET_TIME(mode='val', params=params)

    # Create training and validation data loaders
    batch_size = 2  # Limited by GPU memory
    num_workers = 5 # Limited by CPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, prefetch_factor=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    model = NeuralNetworkRadarDetector("FPN", "resnet152", params, in_channels=IN_CHANNELS, out_classes=OUT_CLASSES, lr=1e-4)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpointt_directory,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,   # keep best 3 models
        mode="min",     # because we're minimizing loss
        save_last=True, # always save the last checkpoint
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="auto",         # We need this for multi-GPU / High Accumulated Batches since we want to sync BN
        sync_batchnorm=True,
        devices=1,
        max_epochs=80,
        precision="16-mixed",
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback, RichProgressBar(leave=True, theme=RichProgressBarTheme(metrics_format='.4e'))],
        gradient_clip_val=0.2,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_checkpoint
    )

    run.finish()


if __name__ == "__main__":
    # Fixes potential bug that causes workers to conflict with shared memory access and crash
    # try: 
    #     mp.set_start_method("spawn", force=True)
    # except RuntimeError:
    #     pass # Already Set

    # Force CUDA initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    params = data_preparation.get_default_params()

    # Initialise parameters
    params["dataset_path"] = '/media/muckelroyiii/ExtremePro/RaDelft/'
    params["train_val_scenes"] = [1,3,4,5,7]
    params["test_scenes"] = [2,6]
    params["train_test_split_percent"] = 0.8
    params["cfar_folder"] = 'radar_ososos'
    params["quantile"] = False

    # This must be kept to false. If the network without elevation is needed, use network_noElevation.py instead
    params["bev"] = False

    checkpoint_path = '/home/muckelroyiii/Desktop/RISS_Research/checkpoints-resnet152/last-v1.ckpt'

    # This trains the NN
    main(params, resume_checkpoint=checkpoint_path)
    # main(params)