import argparse
import os
import sys

# Ensure project root is on sys.path so model imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib

import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from utils import prepare_device

# Custom colormap: 0=Dark Slate Gray (Background), 1=Vibrant Red (Non-Forest), 2=Forest Green (Forest)
FOREST_CMAP = ListedColormap(['#1E293B', '#E53E3E', '#38A169'])


def mask_to_rgb(mask):
    """Convert a label mask (H, W) with values {0, 1, 2} to an RGB uint8 image.

    0 = Background  -> Dark Slate Gray  #1E293B  (30, 41, 59)
    1 = Non-Forest   -> Vibrant Red      #E53E3E  (229, 62, 62)
    2 = Forest        -> Forest Green    #38A169  (56, 161, 105)
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 0] = [30, 41, 59]       # Background  -> Dark Slate Gray
    rgb[mask == 1] = [229, 62, 62]      # Non-Forest  -> Vibrant Red
    rgb[mask == 2] = [56, 161, 105]     # Forest      -> Forest Green
    return rgb


def main(config):
    logger = config.get_logger('qa_generate')

    test_data_loader = config.init_obj('train_data_loader', module_data, mode='test')

    model = config.init_obj('arch', module_arch)

    # get checkpoint path
    resume_path = config.resume if config.resume else config['trainer']['pretrained_model']
    logger.info(f'Loading checkpoint: {resume_path} ...')

    # Fix for pathlib loading issue between Windows/Linux
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        checkpoint = torch.load(resume_path, weights_only=False)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        pathlib.PosixPath = temp
        return
    pathlib.PosixPath = temp

    state_dict = checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=False)

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    model.eval()

    # create dedicated output path
    run_name = config.config.get('name', 'QA_Run')
    # parse timestamp from the resume path if possible
    parts = str(resume_path).replace('\\', '/').split('/')
    timestamp = parts[-2] if len(parts) >= 2 else "unknown_time"

    vis_dir = os.path.join('saved', 'Outputs', f'QA_{run_name}_{timestamp}')
    os.makedirs(vis_dir, exist_ok=True)

    logger.info(f"Saving minimalistic predicted masks and numpy arrays to: {vis_dir}")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            if isinstance(output, tuple):
                softmaxed = output[1]
            else:
                softmaxed = output

            pred = torch.argmax(softmaxed, dim=1)

            for i in range(data.size(0)):
                prediction = pred[i].cpu().numpy()   # values: 0=Non-Forest, 1=Forest
                truth = target[i].cpu().numpy()       # values: 0=Background, 1=Non-Forest, 2=Forest
                if len(truth.shape) == 3:
                    truth = truth.squeeze(0)

                # Remap prediction to match ground-truth label space:
                # model outputs 0=Non-Forest, 1=Forest -> shift to 1=Non-Forest, 2=Forest
                prediction_remapped = prediction + 1
                # Propagate background mask from ground truth to prediction
                # (predictions don't have background, so overlay it from GT)
                background_mask = (truth == 0)
                prediction_remapped[background_mask] = 0  # set background pixels to 0 (black)

                # Save the predicted mask as an RGB image
                pred_path_png = os.path.join(vis_dir, f'batch_{batch_idx}_sample_{i}_pred.png')
                plt.imsave(pred_path_png, mask_to_rgb(prediction_remapped))

                # Save the ground truth mask as an RGB image
                truth_path_png = os.path.join(vis_dir, f'batch_{batch_idx}_sample_{i}_truth.png')
                plt.imsave(truth_path_png, mask_to_rgb(truth))

                # Save numpy arrays for ensembling (raw prediction, not remapped)
                pred_path_npy = os.path.join(vis_dir, f'batch_{batch_idx}_sample_{i}_pred.npy')
                np.save(pred_path_npy, prediction.astype(np.uint8))

            if batch_idx % 10 == 0:
                logger.info(f'Processed batch {batch_idx}/{len(test_data_loader)}')

    logger.info(f'Finished QA generation. Output saved to {vis_dir}')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='QA generation Script')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable')
    args.add_argument('-n', '--name', default=None, type=str, help='custom name for the output directory')

    config = ConfigParser.from_args(args)
    # Also pass the parsed name arg into main if it was provided
    parsed_args = args.parse_args()
    if parsed_args.name:
        config.config['name'] = parsed_args.name

    main(config)
