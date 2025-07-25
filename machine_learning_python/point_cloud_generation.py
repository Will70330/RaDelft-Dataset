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

def extract_model_name(checkpoint_path):
    # Look for pattern like "checkpoints-resnet18"
    match = re.search(r'checkpoints-([^/]+)', checkpoint_path)
    if match:
        return match.group(1)
    else:
        return 'unknown_model'

def generate_point_clouds(params, checkpoints, print_path=False, overwrite_pc=False):
    # Check for GPU availability
    device = torch.device('cpu') # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    eval_modes = ['train', 'test', 'val']

    # Generate PCs based on each model checkpoint path
    for checkpoint in checkpoints:

        # Grab the model name
        model_name = extract_model_name(checkpoint)
        
        # Load Model
        try: 
            cp = torch.load(checkpoint, map_location=device) # Load checkpoint to device (GPU)
            # Load the entire Lightning module
            if model_name == 'resnet18' or model_name == 'resnet50':
                model = NeuralNetworkRadarDetector('FPN', model_name, params, in_channels=64, out_classes=34, use_groupNorm=False)
            else:
                model = NeuralNetworkRadarDetector('FPN', model_name, params, in_channels=64, out_classes=34, use_groupNorm=True)
            model.load_state_dict(cp['state_dict'])
        except Exception as e:
            print(f'Error loading model ({model_name}) from checkpoint {checkpoint}: {e}')
            continue
        model.to(device)
        model.eval()

        for mode in eval_modes:
            # Construct Data Loader
            dataset = RADCUBE_DATASET_TIME(mode=mode, params=params)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=False)

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
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2, 6]
    params["train_test_split_percent"] = 0.8
    params["cfar_folder"] = 'radar_ososos'
    params["quantile"] = False

    # This must be kept to false. If the network without elevation is needed, use network_noElevation.py instead
    params["bev"] = False

    checkpoint_paths = {
        # '/home/muckelroyiii/Desktop/RISS_Research/checkpoints-resnet18/model-epoch=19-val_loss=0.0004.ckpt',
        # '/home/muckelroyiii/Desktop/RISS_Research/checkpoints-resnet50/model-epoch=15-val_loss=0.0004.ckpt',
        # '/home/muckelroyiii/Desktop/RISS_Research/checkpoints-resnet101/model-epoch=19-val_loss=0.0004.ckpt',
        # '/home/muckelroyiii/Desktop/RISS_Research/checkpoints-resnet152/model-epoch=17-val_loss=0.0009.ckpt',
        '/home/muckelroyiii/Desktop/RISS_Research/Results/checkpoints-resnet152/model-epoch=39-val_loss=0.0016.ckpt'
    }

    generate_point_clouds(params, checkpoint_paths, print_path=True, overwrite_pc=True)

