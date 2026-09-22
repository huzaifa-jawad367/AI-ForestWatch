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
    logger = config.get_logger('test_visualize')

    # setup data_loader instances using the 'test' split from your pkl files
    test_data_loader = config.init_obj('train_data_loader', module_data, mode='test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    # get checkpoint path
    resume_path = config.resume if config.resume else config['trainer']['pretrained_model']
    logger.info(f'Loading checkpoint: {resume_path} ...')
    checkpoint = torch.load(resume_path, weights_only=False)

    # restore pathlib
    pathlib.PosixPath = temp

    # Check if the state dict was saved with DataParallel 'module.' prefix
    state_dict = checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=False)

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    model.eval()

    # create dedicated output path
    vis_dir = os.path.join(config.log_dir, 'test_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    logger.info(f"Saving 256x256 test visualizations to: {vis_dir}")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # UNetMFFSE returns a tuple (x_binary, softmaxed)
            if isinstance(output, tuple):
                softmaxed = output[1]
            else:
                softmaxed = output

            pred = torch.argmax(softmaxed, dim=1)

            # Generate figure for each sample in the batch
            for i in range(data.size(0)):
                img = data[i].cpu().numpy()
                truth = target[i].cpu().numpy()       # values: 0=Background, 1=Non-Forest, 2=Forest
                prediction = pred[i].cpu().numpy()    # values: 0=Non-Forest, 1=Forest

                # Ensure truth mask is 2D
                if len(truth.shape) == 3:
                    truth = truth.squeeze(0)

                # Remap prediction to match ground-truth label space:
                # model outputs 0=Non-Forest, 1=Forest -> shift to 1=Non-Forest, 2=Forest
                prediction_remapped = prediction + 1
                # Propagate background mask from ground truth to prediction
                background_mask = (truth == 0)
                prediction_remapped[background_mask] = 0  # set background pixels to 0 (black)

                # Extract first 3 bands for a pseudo-RGB visualization
                if img.shape[0] >= 3:
                    rgb = img[:3, :, :].transpose(1, 2, 0)
                else:
                    rgb = img[0, :, :]

                # Normalize RGB purely for visualization [0, 1]
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb)
                axes[0].set_title('Input Image (First 3 Bands)')
                axes[0].axis('off')

                # Use custom RGB rendering: Black=Background, Red=Non-Forest, Green=Forest
                axes[1].imshow(mask_to_rgb(truth))
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')

                axes[2].imshow(mask_to_rgb(prediction_remapped))
                axes[2].set_title('Model Prediction Mask')
                axes[2].axis('off')

                save_path = os.path.join(vis_dir, f'batch_{batch_idx}_sample_{i}.png')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

            if batch_idx % 5 == 0:
                logger.info(f'Processed batch {batch_idx}/{len(test_data_loader)}')

    logger.info('Finished generating all test visualizations.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Test Visualization Script')
    args.add_argument('-c', '--config', default="./config.json", type=str,
                      help='config file path (default: ./config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
