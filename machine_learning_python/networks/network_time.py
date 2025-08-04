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
import torch.nn.init as init
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
from utils.compute_metrics import compute_metrics_time, compute_pd_pfa, compute_chamfer_distance
import wandb

run = None

OUT_CLASSES = 34  # 44 elevation angles
IN_CHANNELS = 64  # output of the ReduceDNet

# ToDO: Check if goes faster with this:
torch.set_float32_matmul_precision('medium')
# mp.set_start_method("spawn", force=True)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def GN(c: int) -> nn.GroupNorm:
    groups_map = {3: 3, 6: 3, 12: 6, 24: 8}
    g = groups_map.get(c, max([g for g in (8,6,4,3,2,1) if c % g == 0]))
    return nn.GroupNorm(num_groups=g, num_channels=c)

class ResidualConv3D(nn.Module):
    def __init__(self, c_in, c_mid, c_out, d1=(1,1,1), d2=(1,1,1), norm_last=True, p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in,  c_mid, kernel_size=3, dilation=d1, padding='same', bias=False)
        self.gn1   = GN(c_mid)
        self.relu  = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout3d(p_drop) if p_drop > 0 else nn.Identity()

        self.conv2 = nn.Conv3d(c_mid, c_out, kernel_size=3, dilation=d2, padding='same', bias=False)
        self.gn2   = GN(c_out) if norm_last else nn.Identity()
        self.drop2 = nn.Dropout3d(p_drop) if p_drop > 0 else nn.Identity()

        self.proj  = nn.Identity() if c_in == c_out else nn.Conv3d(c_in, c_out, kernel_size=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        # He (Kaiming) init for ReLU nets
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(self.proj, nn.Conv3d):
            init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

        # GN affine gamma=1, beta=0 is PyTorch default; leave as-is.

    def forward(self, x):
        skip = self.proj(x)
        y = self.conv1(x); y = self.gn1(y); y = self.relu(y); y = self.drop1(y)
        y = self.conv2(y); y = self.gn2(y); y = self.relu(y + skip); y = self.drop2(y)
        return y

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

    def __init__(
            self, arch, encoder_name, params, in_channels, out_classes, 
            lr=3e-4, warmup_epochs=10, use_groupNorm=False, p_drop=0.0,
            debug=False, **kwargs
    ):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_hyperparameters()
        self.DopplerReducer = DopplerEncoder(use_groupNorm=use_groupNorm)
        self.p_drop = p_drop
        self.debug = debug

        # Chamfer Distance Tracking
        self.val_chamfer_distances = []

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=None, 
            in_channels=in_channels, 
            classes=out_classes, 
            **kwargs
        )

        # Temporal smoothing layers (Dilated + Residuals)
        # self.block1 = ResidualConv3D(3, 6, 12, d1=(1,1,1), d2=(1,1,1), p_drop=self.p_drop)
        # self.block2 = ResidualConv3D(12, 24, 12, d1=(1,2,2), d2=(1,1,1), p_drop=self.p_drop)
        # self.block3 = ResidualConv3D(12, 6, 3, d1=(1,1,1), d2=(1,2,2), p_drop=self.p_drop)
        
        kernel_size = (3, 5, 7)

        # layer_size = 6 if encoder_name == 'resnet50' else 12

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

        # self.dropout = nn.Dropout3d(p=self.p_drop)  # Increased dropout for stronger regularization
        self.temporal_mix = nn.Parameter(torch.tensor(0.85))  # Start with 85% temporal

        # # Initialize temporal smoothing layers
        for m in [self.conv1, self.conv2, self.conv3, self.conv4]: # , self.conv5, self.conv6]:
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
        # original_mask = mask.clone()

        # Temporal smoothing
        # mask = self.block1(mask)
        # mask = self.block2(mask)
        # mask = self.block3(mask)

        mask = self.conv1(mask)
        mask = self.relu1(mask)
        # mask = self.dropout(mask) if self.training else mask
        mask = self.conv2(mask)
        mask = self.relu2(mask)
        # mask = self.dropout(mask) if self.training else mask
        mask = self.conv3(mask)
        mask = self.relu3(mask)
        # mask = self.dropout(mask) if self.training else mask
        mask = self.conv4(mask)
        mask = self.relu4(mask)
        # mask = self.dropout(mask) if self.training else mask
        mask = self.conv5(mask)
        mask = self.relu5(mask)
        # mask = self.dropout(mask) if self.training else mask
        mask = self.conv6(mask)

        # mask = (
        #     torch.sigmoid(self.temporal_mix) * mask + 
        #     (1 - torch.sigmoid(self.temporal_mix)) * original_mask
        # )

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
            radar_cube_out = RAE_Cube.sigmoid().detach()

            sigmoid_vals = radar_cube_out.flatten()
            if self.debug:
                print(f"Predictions > 0.3: {(sigmoid_vals > 0.3).sum()}")
                print(f"Predictions > 0.4: {(sigmoid_vals > 0.4).sum()}")
                print(f"Predictions > 0.5: {(sigmoid_vals > 0.5).sum()}")
                print(f"Predictions > 0.6: {(sigmoid_vals > 0.6).sum()}")

            if self.counter % 10 == 0 and self.debug:  # Print every 10th validation step
                print(f"\n=== Validation Debug (step {self.counter}) ===")
                print(f"RAE_Cube shape: {RAE_Cube.shape}")
                print(f"RAE_Cube min/max before sigmoid: {RAE_Cube.min().item():.4f} / {RAE_Cube.max().item():.4f}")
                print(f"After sigmoid min/max: {radar_cube_out.min().item():.4f} / {radar_cube_out.max().item():.4f}")
                print(f"GT lidar shape: {gt_lidar_cube.shape}")
                print(f"GT lidar unique values: {torch.unique(gt_lidar_cube)}")
                print(f"GT lidar sum: {gt_lidar_cube.sum().item()}")

            # Threshold and crop
            radar_cube_out = (radar_cube_out > 0.5).cpu().numpy()
            if radar_cube_out.ndim == 5:
                radar_cube_out = radar_cube_out[:, :, :, :-12, 8:-8]
            else:
                radar_cube_out = radar_cube_out[:, :, :-12, 8:-8]

            # More debug info
            if self.counter % 10 == 1 and self.debug:  # Right after the previous print
                print(f"After slicing shape: {radar_cube_out.shape}")
                print(f"Radar predictions sum: {radar_cube_out.sum()}")
                print(f"Radar predictions mean: {radar_cube_out.mean():.6f}")
                print("=====================================\n")
                
            pd, pfa = compute_pd_pfa(gt_lidar_cube.cpu().numpy(), radar_cube_out)

            # Compute chamfer distance
            cd = self.compute_batch_chamfer_distance(batch, RAE_Cube, radar_cube_out)

            self.counter += 1

            return loss, pd, pfa, cd

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        actual_batch_size = batch[0].shape[0] # This should be 1 / 2
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=actual_batch_size)
        
        # Log to wandb less frequently to reduce overhead
        if batch_idx % 10 == 0:
            run.log({'train_loss': loss.item(), 'lr': self.optimizers().param_groups[0]['lr']}) #'temporal_mix': torch.sigmoid(self.temporal_mix).item()})
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.last_rad_cube = batch[0]
        loss, pd, pfa, cd = self.shared_step(batch, "valid")
        self.val_chamfer_distances.append(cd)
        
        actual_batch_size = batch[0].shape[0] # This should be 1 / 2
        self.log_dict({'val_loss': loss, 'val_pd': pd, 'val_pfa': pfa, 'val_cd': cd},
                      on_step=False, on_epoch=True, prog_bar=True,
                      logger=True, batch_size=actual_batch_size)
        
        # Log to wandb less frequently to reduce overhead
        if batch_idx % 10 == 0:
            run.log({
                'val_loss': loss.item(),
                'val_pd': pd.item() if isinstance(pd, torch.Tensor) else pd,
                'val_pfa': pfa.item() if isinstance(pfa, torch.Tensor) else pfa,
                'val_cd': cd,
                'lr': self.hparams.lr,
                # 'temporal_mix': torch.sigmoid(self.temporal_mix).item(),
                }
            )
        
        return loss

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_validation_epoch_end(self):
        """Calculate and log epoch-average Chamfer Distance."""
        if self.val_chamfer_distances:
            avg_chamfer_distance = np.mean(self.val_chamfer_distances)
            
            # Log epoch average
            self.log('val_chamfer_distance_epoch_avg', avg_chamfer_distance, 
                    on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            # Log to wandb
            if run:
                run.log({
                    'val_chamfer_distance_epoch_avg': avg_chamfer_distance,
                    'epoch': self.current_epoch
                })
            
            if self.debug:
                print(f"\nEpoch {self.current_epoch} - Average Chamfer Distance: {avg_chamfer_distance:.4f}")
                print(f"Min CD: {np.min(self.val_chamfer_distances):.4f}")
                print(f"Max CD: {np.max(self.val_chamfer_distances):.4f}")
                print(f"Std CD: {np.std(self.val_chamfer_distances):.4f}\n")
        
        # Clear the list for next epoch
        self.val_chamfer_distances = []

    def on_train_epoch_start(self):
        """Ensure validation metrics are cleared when starting training."""
        self.val_chamfer_distances = []

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
            total_iters=10
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
            milestones=[10]
        )
        
        # Add ReduceLROnPlateau on top for additional adaptation
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,      # Less aggressive reduction since we have cosine
            patience=8,      # Higher patience to let cosine schedule work
            # verbose=True,  # removed from pytorch 2.7
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

    # Helper Functions
    def get_gt_pointcloud(self, batch):
        """
        Extract or load the ground truth point cloud from the batch.
        This assumes the GT point cloud path is provided in the batch.
        """
        # If your batch includes the GT point cloud path (modify based on your dataloader)
        # Option 1: If GT path is in batch
        if len(batch) > 2 and 'gt_path' in batch[2]:
            gt_path = batch[2]['gt_path']
            gt_pointcloud = np.load(gt_path)
            # Ensure y-axis is correctly oriented (from compute_metrics_time)
            gt_pointcloud[:, 1] = -gt_pointcloud[:, 1]
            return gt_pointcloud
        
        # Option 2: Convert GT cube back to point cloud (less accurate but works)
        else:
            gt_lidar_cube = batch[1].cpu().numpy()
            # Use a simplified conversion since we don't have the original radar cube
            gt_pointcloud = self.cube_to_pointcloud_simple(gt_lidar_cube)
            return gt_pointcloud

    def cube_to_pointcloud_simple(self, cube):
        """
        Simple conversion from binary cube to point cloud for GT data.
        """
        # Get indices where cube is 1
        if cube.ndim == 5:  # (batch, time, elevation, range, azimuth)
            # Process middle time frame for simplicity
            cube = cube[:, 1, :, :, :]
        
        points = []
        for b in range(cube.shape[0]):
            # Find non-zero indices
            elev_idx, range_idx, az_idx = np.where(cube[b] > 0)
            
            if len(elev_idx) > 0:
                # Convert to physical coordinates
                ranges = self.params['range_axis'][range_idx]
                azimuths = self.params['azimuth_axis'][az_idx]
                elevations = self.params['elevation_axis'][elev_idx]
                
                # Convert to Cartesian
                x = ranges * np.cos(elevations) * np.cos(azimuths)
                y = ranges * np.cos(elevations) * np.sin(azimuths)
                z = ranges * np.sin(elevations)
                
                points.append(np.stack([x, y, z], axis=1))
            else:
                points.append(np.zeros((0, 3)))
        
        return points

    def compute_batch_chamfer_distance(self, batch, RAE_Cube, radar_cube_out):
        chamfer_distances = []
        batch_size = RAE_Cube.shape[0]
        item_params_list = batch[2]
        
        # item_params_list has 3 elements (time frames)
        # Each element has batch_size paths
        for b in range(batch_size):
            # Get the GT path for batch item b from middle time frame
            gt_path = item_params_list[1]['gt_path'][b]
            # print(f"CD GT path: {gt_path}")
            
            gt_pointcloud = np.load(gt_path)
            gt_pointcloud[:, 1] = -gt_pointcloud[:, 1]
            
            # Convert prediction to point cloud
            pred_cube = torch.from_numpy(radar_cube_out[b, 1]).unsqueeze(0)
            pred_pointcloud = data_preparation.cube_to_pointcloud(
                pred_cube,
                self.params,
                batch[0][b, 1].cpu().numpy(),
                mode='radar'
            )
            
            if pred_pointcloud.shape[1] == 4:
                pred_pointcloud = pred_pointcloud[:, :3]
            
            # Compute CD
            if len(pred_pointcloud) > 0 and len(gt_pointcloud) > 0:
                cd = compute_chamfer_distance(gt_pointcloud, pred_pointcloud)
                chamfer_distances.append(cd)
        
        return np.mean(chamfer_distances) if chamfer_distances else 0.0


# main function
def main(params, resume_checkpoint=None, debug=False):
    # Start a new wandb run to track this script.
    global run
    model_name = 'resnet50'
    checkpoint_directory = f"checkpoints-{model_name}-t2"
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
                "architecture": f"{model_name}-t2",
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
                "architecture": f"{model_name}-t2",
                "dataset": "RaDelft",
                "epochs": 50,
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
    batch_size = 4  # Limited by GPU memory
    num_workers = 12 # Limited by CPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    model = NeuralNetworkRadarDetector("FPN", f"{model_name}", params, in_channels=IN_CHANNELS, out_classes=OUT_CLASSES, lr=1e-4, debug=False, use_groupNorm=False, p_drop=0.2)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_directory,
        filename="resnet50-t2-{epoch:02d}-{val_loss:.4f}",
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
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback, RichProgressBar(leave=True, theme=RichProgressBarTheme(metrics_format='.4e'))],
        gradient_clip_val=0.25,
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
    params["dataset_path"] = '/media/muckelroyiii/Mass-Storage/RaDelft'
    params["train_val_scenes"] = [1,3,4,5,7]
    params["test_scenes"] = [2,6]
    params["train_test_split_percent"] = 0.8
    params["cfar_folder"] = 'radar_ososos'
    params["quantile"] = False

    # This must be kept to false. If the network without elevation is needed, use network_noElevation.py instead
    params["bev"] = False

    checkpoint_path = '/home/muckelroyiii/Desktop/riss-research/checkpoints-resnet50-t2/last.ckpt'

    # This trains the NN
    main(params, resume_checkpoint=checkpoint_path)
    # main(params, debug=False)