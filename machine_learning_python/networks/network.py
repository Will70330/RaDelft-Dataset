# add parent directory to path
import sys

import torch.utils.checkpoint

# apend the absolute path of the parent directory
sys.path.append(sys.path[0] + "/..")
import re
import os
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_preparation import data_preparation
import torch.nn as nn
from pytorch_lightning.callbacks import RichProgressBar
from loaders.rad_cube_loader import RADCUBE_DATASET
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.compute_metrics import compute_metrics, compute_pd_pfa
import wandb

run = None

OUT_CLASSES = 34  # 44 elevation angles
IN_CHANNELS = 64  # output of the ReduceDNet

# ToDO: Check if goes faster with this:
torch.set_float32_matmul_precision('medium')


# This gets rid of the Doppler dimension to get a "2D image".
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

    def __init__(self, arch, encoder_name, params, in_channels, out_classes, 
                 lr=3e-4, warmup_epochs=5, use_groupNorm=False, debug=False, **kwargs):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_hyperparameters()
        self.DopplerReducer = DopplerEncoder(use_groupNorm=use_groupNorm)
        self.debug = debug
        self.model = smp.create_model(
            arch, 
            encoder_name=encoder_name, 
            in_channels=in_channels,
            classes=out_classes, 
            **kwargs
        )
        self.counter = 0
        self.params = params
        self.warmup_epochs = warmup_epochs

    def forward(self, image):
        image = self.DopplerReducer(image)
        image = image.float()
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        # Load input and GT
        RAD_cube = batch[0]  # range azimuth doppler cube, the input to the network
        gt_lidar_cube = batch[1]

        # Run the network
        RAE_Cube = self.forward(RAD_cube)  # output is a binary dense mask of the cube in RAE format: range, azimuth, elevation

        loss = data_preparation.radarcube_lidarcube_loss(RAE_Cube, gt_lidar_cube, self.params)

        # Add L1 regularization for additional control over overfitting
        l1_lambda = 1e-5
        l1_reg = sum(p.abs().sum() for p in self.parameters() if p.requires_grad)
        loss = loss + l1_lambda * l1_reg

        if stage == 'valid':
            radar_cube_out = RAE_Cube.sigmoid().squeeze().cpu().detach().numpy()
            radar_cube_out = radar_cube_out > 0.5
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
            run.log({'val_loss': loss.item(), 'val_pd': pd.item(), 'val_pfa': pfa.item(), 'lr': self.hparams.lr,})

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

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=1)
    #     return {'optimizer': optimizer, 'lr_scheduler': scheduler,
    #             'monitor': 'train_loss_epoch'}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=1e-4)  # Increased weight decay for stronger L2 regularization

        # Warmup scheduler - crucial for stable training with batch_size=1
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01,  # Start at 1% of target LR (more conservative than 0.005)
            total_iters=self.warmup_epochs
        )
        
        # Cosine annealing with warm restarts
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,      # First restart after 10 epochs
            T_mult=2,    # Double the period after each restart (10, 20, 40, ...)
            eta_min=1e-8 # Minimum learning rate
        )
        
        # Combine warmup and cosine schedules
        base_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs]
        )
        
        # Add ReduceLROnPlateau on top for additional adaptation
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,      # Less aggressive reduction since we have cosine
            patience=8,      # Higher patience to let cosine schedule work
            # verbose=True,
            min_lr=5e-9      # Lower than cosine min_lr
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

def extract_model_name(checkpoint_path):
    # Look for pattern like "modelname-epoch..." and extract the model name part
    match = re.search(r'([^/]+)-epoch', checkpoint_path)
    if match:
        return match.group(1)
    else:
        return 'unknown_model'

def generate_point_clouds(params):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet18-t0-epoch=14-val_loss=0.0012.ckpt'
    # Grab the model name
    model_name = extract_model_name(path)
    eval_modes = ['train', 'test', 'val']
    # NOTE file is not always readable, permissions can be fucked
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = NeuralNetworkRadarDetector("FPN", "resnet18", params, in_channels=IN_CHANNELS, out_classes=OUT_CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for mode in eval_modes:
        # Create Loader
        dataset = RADCUBE_DATASET(mode=mode, params=params)
        base_network_path = f'network/{model_name}/{mode}/'

        # Create training and validation data loaders
        num_workers = 16
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=False)
        counter = 0

        for batch_idx, batch in enumerate(loader):
            counter = counter + 1
            radar_cube, lidar_cube, data_dict = batch

            with torch.no_grad():
                output = model(radar_cube)
                for i in range(lidar_cube.shape[0]):
                    # Construct Save Path
                    cfar_path = data_dict["cfar_path"][i]
                    save_path = re.sub(r"radar_.+/", rf'{base_network_path}', cfar_path)
                    print(save_path) if batch_idx == 1 else None
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    radar_pc = data_preparation.cube_to_pointcloud(output[i, :, :, :], params, radar_cube[i, :, :, :, :],
                                                                data_dict["elevation_path"][i], 'radar')
                    radar_pc[:, 2] = -radar_pc[:, 2]

                    np.save(save_path, radar_pc)

# main function
def main(params, resume_checkpoint=None, debug=False):
    # Start a new wandb run to track this script
    global run
    checkpoint_directory = "checkpoints-resnet152-t0"
    if resume_checkpoint:
        ckpt_dir = os.path.dirname(resume_checkpoint)
    else: 
        ckpt_dir = checkpoint_directory
        
    run_id_file = os.path.join(ckpt_dir, "wandb_run_id.txt")
    if resume_checkpoint and os.path.exists(run_id_file):
        # Load old run ID and re-attach
        with open(run_id_file, "r") as f:
            old_id = f.read().strip()
        run = wandb.init(
            entity="will_70330",
            project="RISS-Research-RaDelft",
            config={
                "architecture": "ResNet152-regularized-t0",
                "dataset": "RaDelft",
                "epochs": 50,
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
                "architecture": "ResNet152-regularized-t0",
                "dataset": "RaDelft",
                "epochs": 50,
            }
        )
        # check for exisiting checkpoint folder and write run ID
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(run_id_file, 'w') as f:
            f.write(run.id)

    # Create training and validation datasets
    train_dataset = RADCUBE_DATASET(mode='train', params=params)
    val_dataset = RADCUBE_DATASET(mode='val', params=params)

    # Create training and validation data loaders
    batch_size = 8
    num_workers = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, prefetch_factor=2)
    model = NeuralNetworkRadarDetector("FPN", "resnet152", params, in_channels=IN_CHANNELS, out_classes=OUT_CLASSES)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_directory,
        filename="resnet152-t0-{epoch:02d}-{val_loss:.4f}",
        save_top_k=10,   # keep best 5 models
        mode="min",     # because we're minimizing loss
        save_last=True, # always save the last checkpoint
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="auto",         # We need this for multi-GPU / High Accumulated Batches since we want to sync BN
        sync_batchnorm=True,
        devices=1,
        max_epochs=50,
        precision="16-mixed",
        # accumulate_grad_batches=2,
        callbacks=[checkpoint_callback, RichProgressBar(leave=True, theme=RichProgressBarTheme(metrics_format='.4e'))],
        gradient_clip_val=0.25,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_checkpoint
    )


if __name__ == "__main__":
    params = data_preparation.get_default_params()

    # Initialise parameters
    params["dataset_path"] = '/media/muckelroyiii/Mass-Storage/RaDelft'
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2,6]
    params["train_test_split_percent"] = 0.8
    params["cfar_folder"] = 'radar_ososos'
    params["quantile"] = False

    # This must be kept to false. If the network without elevation is needed, use network_noElevation.py instead
    params["bev"] = False

    # This train the NN
    # main(params)

    # PC Generation
    generate_point_clouds(params)
