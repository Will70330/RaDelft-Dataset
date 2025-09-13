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
from loaders.rad_cube_loader import RADCUBE_DATASET
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.models as models 
from utils.compute_metrics import compute_metrics_time, compute_pd_pfa
import wandb
from networks.network_time import NeuralNetworkRadarDetector as nnDetector_time
from networks.network import NeuralNetworkRadarDetector as nnDetector

IN_CHANNELS=64
OUT_CLASSES=34

def extract_model_name(checkpoint_path):
    # Look for pattern like "checkpointname-epoch..." and extract the checkpoint name part
    match = re.search(r'([^/]+)-epoch', checkpoint_path)
    if match:
        checkpoint_name = match.group(1)
        # Extract the model name (first part before '-')
        model_name = checkpoint_name.split('-')[0]
        return checkpoint_name, model_name
    else:
        return 'unknown_checkpoint', 'unknown_model'

def compute_pc(radar_cube, lidar_cube, data_dict, output, i, batch_idx, overwrite_pc, base_network_path, file_skip_counter, print_path, use_temporal):
    # PC GENERATION FOR NONE TEMPORAL MODELS
    if not use_temporal:
        # Construct Save Path
        cfar_path = data_dict["cfar_path"][i]
        save_path = re.sub(r"radar_.+/", rf'{base_network_path}', cfar_path)
        if os.path.exists(save_path) and not overwrite_pc:
            file_skip_counter += 1
            return
        print(save_path) if print_path and batch_idx == 1 else None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # print(f"\nOutput shape: {output[i, :, :, :].shape}")
        # print(f"\nRadar cube shape: {radar_cube[i, :, :, :, :].shape}\n")
        radar_pc = data_preparation.cube_to_pointcloud(
            cube=output[i, :, :, :],
            params=params,
            radar_cube=radar_cube[i, :, :, :, :],
            mode='radar',
            # dop_fold_path=data_dict["elevation_path"][i]
        )

        radar_pc[:, 2] = -radar_pc[:, 2]

        # Save Result
        np.save(save_path, radar_pc)

    else:
        # PC GENERATION FOR TEMPORAL MODELS
        for t in range(lidar_cube.shape[1]):
            output_t = output[i, t, :, :, :]
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
                output_t,
                params, radar_cube[i, t, :, :, :],
                'radar'
            )
            radar_pc[:, 2] = -radar_pc[:, 2]

            # save result
            np.save(save_path, radar_pc)

def generate_point_clouds(params, checkpoints, print_path=False, overwrite_pc=False, use_temporal=False):
    path = '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet50-t0-epoch=27-val_loss=0.0016.ckpt'
    ckpt_name, model_name = extract_model_name(path)
    base_network_path = f'network/{ckpt_name}/test/'
    # NOTE file is not always readable, permissions can be fucked
    checkpoint = torch.load(path, weights_only=False)
    model = nnDetector("FPN", "resnet50", params, in_channels=IN_CHANNELS, out_classes=OUT_CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Create Loader
    val_dataset = RADCUBE_DATASET(mode='test', params=params)

    # Create training and validation data loaders
    num_workers = 16
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=False)
    counter = 0

    for batch in tqdm(val_loader):
        counter = counter + 1
        radar_cube, lidar_cube, data_dict = batch

        with torch.no_grad():
            output = model(radar_cube)
            for i in range(lidar_cube.shape[0]):
                radar_pc = data_preparation.cube_to_pointcloud(cube=output[i, :, :, :], params=params, radar_cube=radar_cube[i, :, :, :, :],
                                                               dop_fold_path=data_dict["elevation_path"][i], mode='radar')

                radar_pc[:, 2] = -radar_pc[:, 2]

                cfar_path = data_dict["cfar_path"][i]
                save_path = re.sub(r"radar_.+/", rf'{base_network_path}', cfar_path)
                # print(save_path)

                np.save(save_path, radar_pc)
    # # Check for GPU availability
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')
    # eval_modes = ['test']

    # # Generate PCs based on each model checkpoint path
    # for checkpoint in checkpoints:

    #     # Grab the model name
    #     ckpt_name, model_name = extract_model_name(checkpoint)
        
    #     # Load Model
    #     try: 
    #         cp = torch.load(checkpoint, map_location=device, weights_only=False) # Load checkpoint to device (GPU)
    #         use_groupNorm = False if model_name == 'resnet18' or model_name == 'resnet50' else True
    #         if not use_temporal:
    #             model = nnDetector(arch='FPN', encoder_name=model_name, params=params, in_channels=64, out_classes=34, use_groupNorm=False)
    #         else:
    #             model = nnDetector_time(arch='FPN', encoder_name=model_name, params=params, in_channels=64, out_classes=34, use_groupNorm=use_groupNorm)
    #         model.load_state_dict(cp['state_dict'])
    #     except Exception as e:
    #         print(f'Error loading model ({ckpt_name}) from checkpoint {checkpoint}: {e}')
    #         continue
    #     model.to(device)
    #     model.eval()

    #     for mode in eval_modes:
    #         # Construct Data Loader
    #         # dataset = RADCUBE_DATASET_TIME(mode=mode, params=params)
    #         dataset = RADCUBE_DATASET(mode=mode, params=params) if not use_temporal else RADCUBE_DATASET_TIME(mode=mode, params=params)
    #         loader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=16, pin_memory=False, prefetch_factor=2)

    #         # Create base directory structure
    #         base_network_path = f'network/{ckpt_name}/{mode}/'
    #         file_skip_counter = 0

    #         # Actual Generation of the point clouds
    #         for batch_idx, batch in tqdm(enumerate(loader), desc=f'generating point clouds for {ckpt_name}: {mode}', unit='batch(s)'):
    #             radar_cube, lidar_cube, data_dict = batch

    #             # Move data to GPU
    #             radar_cube = radar_cube.to(device, non_blocking=True)
    #             lidar_cube = lidar_cube.to(device, non_blocking=True)

    #             with torch.no_grad():
    #                 output = model(radar_cube)

    #                 # Move output back to CPU for post-processing
    #                 output_cpu = output.cpu()
    #                 radar_cube_cpu = radar_cube.cpu()

    #                 for i in range(lidar_cube.shape[0]):
    #                     compute_pc(
    #                         radar_cube=radar_cube_cpu, lidar_cube=lidar_cube, data_dict=data_dict,
    #                         output=output_cpu, i = i, batch_idx=batch_idx,
    #                         base_network_path=base_network_path, file_skip_counter=file_skip_counter,
    #                         print_path=print_path, use_temporal=use_temporal, overwrite_pc=overwrite_pc
    #                     )
                        

    #         print(f'PC Overwriting set to {overwrite_pc}, skipped overwriting {file_skip_counter} files...')
    #     # Free up GPU Memory
    #     del model, cp
    #     torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":

    # Force CUDA initialization
    if torch.cuda.is_available():
        # torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # torch.set_float32_matmul_precision('medium')

    params = data_preparation.get_default_params()

    # Initialise parameters
    params["dataset_path"] = '/media/muckelroyiii/Mass-Storage/RaDelft/'
    params["train_val_scenes"] = []
    params["test_scenes"] = [2, 6]
    params["train_test_split_percent"] = 0.8
    params["cfar_folder"] = 'radar_ososos'
    params["quantile"] = False

    # This must be kept to false. If the network without elevation is needed, use network_noElevation.py instead
    params["bev"] = False

    checkpoint_paths = {
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet18-t0-epoch=14-val_loss=0.0012.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet18-t4-epoch=19-val_loss=0.0004.ckpt',
        '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet50-t0-epoch=27-val_loss=0.0016.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet50-t0-epoch=29-val_loss=0.0011.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet50-t2-epoch=39-val_loss=0.0015.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet50-t4-epoch=17-val_loss=0.0004.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet50-t4-new-epoch=34-val_loss=0.0016.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet101-t0-epoch=19-val_loss=0.0012.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet101-t0-epoch=34-val_loss=0.0012.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet101-t2-epoch=29-val_loss=0.0033.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet101-t2-epoch=34-val_loss=0.0016.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet101-t4-epoch=19-val_loss=0.0004.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet101-t4-new-epoch=39-val_loss=0.0017.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet101-t6-epoch=39-val_loss=0.0016.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet152-t0-epoch=34-val_loss=0.0012.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/resnet152-t4-epoch=39-val_loss=0.0016.ckpt',
        # '/home/muckelroyiii/Desktop/riss-research/results_collection/xception-epoch=21-val_loss=0.0054.ckpt'
    }

    generate_point_clouds(params, checkpoint_paths, print_path=True, overwrite_pc=True, use_temporal=False)

