import argparse
import os
import sys

# Ensure project root is on sys.path so model imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import pathlib

import data_loader.data_loaders as module_data
from parse_config import ConfigParser

def main(config):
    logger = config.get_logger('generate_rgb')

    test_data_loader = config.init_obj('train_data_loader', module_data, mode='test')

    # create dedicated output path for RGB images
    rgb_dir = os.path.join('saved', 'Outputs', 'RGB_Images')
    os.makedirs(rgb_dir, exist_ok=True)

    logger.info(f"Saving RGB images to: {rgb_dir}")

    # We need to map Landsat-8 bands 4, 3, 2 to R, G, B
    # Let's check config bands
    bands = config['train_data_loader']['args']['bands']
    logger.info(f"Configured bands in dataloader: {bands}")

    # 1-indexed band numbers: B4=Red, B3=Green, B2=Blue
    try:
        r_idx = bands.index(4)
        g_idx = bands.index(3)
        b_idx = bands.index(2)
        logger.info(f"Mapped B4->channel {r_idx}, B3->channel {g_idx}, B2->channel {b_idx}")
    except ValueError:
        # Fallback to user specified channels: 4th band/channel is R (index 3), 3rd is G (index 2), 2nd is B (index 1)
        r_idx = 3 if len(bands) > 3 else min(2, len(bands)-1)
        g_idx = 2 if len(bands) > 2 else min(1, len(bands)-1)
        b_idx = 1 if len(bands) > 1 else 0
        logger.warning(f"Bands [4, 3, 2] not found in config. Using fallback: R->{r_idx}, G->{g_idx}, B->{b_idx}")

    for batch_idx, (data, _) in enumerate(test_data_loader):
        for i in range(data.size(0)):
            img = data[i].cpu().numpy()

            # Extract R, G, B channels
            r = img[r_idx, :, :]
            g = img[g_idx, :, :]
            b = img[b_idx, :, :]

            rgb = np.stack([r, g, b], axis=-1)

            # Normalize to [0, 1] for visualization
            rgb_min = rgb.min()
            rgb_max = rgb.max()
            if rgb_max > rgb_min:
                rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
            else:
                rgb = np.zeros_like(rgb)

            # Clip to [0, 1] to avoid matplotlib warning/errors
            rgb = np.clip(rgb, 0, 1)

            save_path = os.path.join(rgb_dir, f'batch_{batch_idx}_sample_{i}_rgb.png')
            plt.imsave(save_path, rgb)

        logger.info(f'Processed batch {batch_idx}/{len(test_data_loader)}')

    logger.info(f'Finished RGB visualization generation. Output saved to {rgb_dir}')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='RGB Generation Script')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str, help='dummy argument for compatibility')
    args.add_argument('-d', '--device', default=None, type=str, help='dummy argument for compatibility')

    config = ConfigParser.from_args(args)
    main(config)
