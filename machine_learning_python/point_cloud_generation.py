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
from networks.network_time import NeuralNetworkRadarDetector

IN_CHANNELS=64
OUT_CLASSES=34

# this gets rid of the Doppler dimension to get a "2D image".
# We go from B*C*D*H*W to B*C*H*W, H and W are ranges and azimuths
class DopplerEncoder_Batch(nn.Module):
    def __init__(self):
        super(DopplerEncoder_Batch, self).__init__()

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
        # self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.norm1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()

        # Step 2: Convolution parameters to reduce from 60 to 15
        self.conv2 = nn.Conv3d(out_channel_1, out_channel_2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        # self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.norm2 = nn.BatchNorm3d(64)
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

class DopplerEncoder_Group(nn.Module):
    def __init__(self):
        super(DopplerEncoder_Group, self).__init__()

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
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        # self.norm1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()

        # Step 2: Convolution parameters to reduce from 60 to 15
        self.conv2 = nn.Conv3d(out_channel_1, out_channel_2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)
        # self.norm2 = nn.BatchNorm3d(64)
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

    def __init__(self, arch, encoder_name, params, in_channels, out_classes, DopplerReducer, lr=3e-4, warmup_epochs=10, **kwargs):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_hyperparameters()
        self.DopplerReducer = DopplerReducer

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

        # enabling gradient checkpointing for Resnet50
        if hasattr(self.model.encoder, 'model'):
            self.model.encoder.model.set_grad_checkpointing(True)

        kernel_size = (3, 5, 7)

        self.conv1 = nn.Conv3d(3, 6, kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(6, 12, kernel_size=kernel_size, padding='same')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(12, 6, kernel_size=kernel_size, padding='same')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(6, 3, kernel_size=kernel_size, padding='same')

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
        mask = self.conv2(mask)
        mask = self.relu2(mask)
        mask = self.conv3(mask)
        mask = self.relu3(mask)
        mask = self.conv4(mask)

        return mask

    def shared_step(self, batch, stage):
        # Load input and GT
        RAD_cube = batch[0]  # range azimuth doppler cube, the input to the network
        gt_lidar_cube = batch[1]  # TODO here we have to get the gt_cloud and convert it to a mask that fits our loss
        # item_params = batch[2]

        # Run the network
        RAE_Cube = self.forward(
            RAD_cube)  # output is a binary dense mask of the cube in RAE format: range, azimuth, elevation

        loss = data_preparation.radarcube_lidarcube_loss_time(RAE_Cube, gt_lidar_cube, self.params)

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
        actual_batch_size = batch[0].shape[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=actual_batch_size)
        run.log({'train_loss': loss.item(), 'lr': self.hparams.lr})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pd, pfa = self.shared_step(batch, "valid")
        
        self.log_dict({'val_loss': loss, 'val_pd': pd, 'val_pfa': pfa, }, on_step=False, on_epoch=True, prog_bar=True,
                      logger=True, batch_size=4)
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
        run.log({"grad_norm": total_norm})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=5e-5)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.005, total_iters=10)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - 10,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[10]
        )
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

        # manual warmup
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
                # 'monitor': 'train_loss_epoch'}

def extract_model_name(checkpoint_path):
    # Look for pattern like "checkpoints-resnet18"
    match = re.search(r'checkpoints-([^/]+)', checkpoint_path)
    if match:
        return match.group(1)
    else:
        return 'unknown_model'

def generate_point_clouds(params, checkpoints, print_path=False, overwrite_pc=False):
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    eval_modes = ['train', 'test', 'val']

    # Generate PCs based on each model checkpoint path
    for checkpoint in checkpoints:

        # Grab the model name
        model_name = extract_model_name(checkpoint)
        
        # Load Model
        try: 
            cp = torch.load(checkpoint, map_location=device) # Load checkpoint to device (GPU)
            if model_name == 'resnet18' or model_name == 'resnet50':
                model = NeuralNetworkRadarDetector('FPN', model_name, params, in_channels=64, out_classes=34, DopplerReducer=DopplerEncoder_Batch())
            else:
                model = NeuralNetworkRadarDetector('FPN', model_name, params, in_channels=64, out_classes=34, DopplerReducer=DopplerEncoder_Group())
            model.load_state_dict(cp['state_dict'])
        except Exception as e:
            print(f'Error loading model ({model_name}) from checkpoint {checkpoint}: {e}')
            continue
        model.to(device)
        model.eval()

        for mode in eval_modes:
            # Construct Data Loader
            dataset = RADCUBE_DATASET_TIME(mode=mode, params=params)
            loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=6, pin_memory=False, prefetch_factor=1)

            # Create base directory structure
            base_network_path = f'network/{model_name}/{mode}/'
            file_skip_counter = 0

            # Actual Generation of the point clouds
            for batch_idx, batch in tqdm(enumerate(loader), desc=f'generating point clouds for {model_name}: {mode}', unit='batch(s)'):
                radar_cube, lidar_cube, data_dict = batch

                # Move data to GPU
                radar_cube = radar_cube.to(device, non_blocking=True)
                lidar_cube = lidar_cube.to(device, non_blocking=True)

                with torch.no_grad():
                    output = model(radar_cube)

                    # Move output back to CPU for post-processing
                    output_cpu = output.cpu()
                    radar_cube_cpu = radar_cube.cpu()

                    for i in range(lidar_cube.shape[0]):
                        for t in range(lidar_cube.shape[1]):
                            output_t = output_cpu[i, t, :, :, :]
                            data_dict_t = data_dict[t]

                            # Construct save path:
                            cfar_path = data_dict_t['cfar_path'][i]
                            save_path = re.sub(r"radar_.+/", rf'{base_network_path}', cfar_path)
                            if os.path.exists(save_path) and not overwrite_pc:
                                file_skip_counter += 1
                                continue
                            print(save_path) if print_path and batch_idx == 1 else None
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)

                            radar_pc = data_preparation.cube_to_pointcloud(
                                output_t, params, radar_cube_cpu[i, t, :, :, :], 'radar'
                            )
                            radar_pc[:, 2] = -radar_pc[:, 2]

                            # save result
                            np.save(save_path, radar_pc)
            print(f'PC Overwriting set to {overwrite_pc}, skipped overwriting {file_skip_counter} files...')
        # Free up GPU Memory
        del model, cp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":

    # Force CUDA initialization
    if torch.cuda.is_available():
        # torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # torch.set_float32_matmul_precision('medium')

    params = data_preparation.get_default_params()

    # Initialise parameters
    params["dataset_path"] = '/media/muckelroyiii/ExtremePro/RaDelft'
    params["train_val_scenes"] = [1, 3]
    params["test_scenes"] = [2, 6]
    params["train_test_split_percent"] = 0.8
    params["cfar_folder"] = 'radar_ososos'
    params["quantile"] = False

    # This must be kept to false. If the network without elevation is needed, use network_noElevation.py instead
    params["bev"] = False

    checkpoint_paths = {
        '/home/muckelroyiii/Desktop/RISS_Research/checkpoints-resnet18/model-epoch=19-val_loss=0.0004.ckpt',
        '/home/muckelroyiii/Desktop/RISS_Research/checkpoints-resnet50/model-epoch=15-val_loss=0.0004.ckpt',
        '/home/muckelroyiii/Desktop/RISS_Research/checkpoints-resnet101/model-epoch=19-val_loss=0.0004.ckpt',
        '/home/muckelroyiii/Desktop/RISS_Research/checkpoints-resnet152/model-epoch=17-val_loss=0.0009.ckpt',
    }

    generate_point_clouds(params, checkpoint_paths, print_path=True, overwrite_pc=False)

