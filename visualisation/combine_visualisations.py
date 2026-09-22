import os
import sys

# Ensure project root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

def main():
    parser = argparse.ArgumentParser(description="Combine test set predictions from 5 directories into single comparison figures.")
    parser.add_argument('--rgb_dir', default='saved/Outputs/RGB_Images', type=str)
    parser.add_argument('--unet_mffse_dir', default='saved/Outputs/QA_Landsat8_UNetMFFSE_MS_0627_122418', type=str)
    parser.add_argument('--segformer_dir', default='saved/Outputs/QA_Landsat8_CustomSegformer_MS_0628_210105', type=str)
    parser.add_argument('--vanilla_unet_dir', default='saved/Outputs/QA_vanilla_unet_author_PretrainedModel', type=str)
    parser.add_argument('--out_dir', default='saved/Outputs/Combined_Visualisations', type=str)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Find all RGB files
    if not os.path.exists(args.rgb_dir):
        print(f"Error: RGB directory {args.rgb_dir} does not exist.")
        return

    rgb_files = sorted([f for f in os.listdir(args.rgb_dir) if f.endswith('_rgb.png')])
    print(f"Found {len(rgb_files)} test samples to combine.")

    # Legend patches for color coding
    legend_patches = [
        mpatches.Patch(color='#1E293B', label='Background / No-Data'),
        mpatches.Patch(color='#E53E3E', label='Non-Forest'),
        mpatches.Patch(color='#38A169', label='Forest'),
    ]

    for rgb_file in rgb_files:
        base_name = rgb_file.replace('_rgb.png', '')

        rgb_path = os.path.join(args.rgb_dir, rgb_file)
        unet_mffse_path = os.path.join(args.unet_mffse_dir, f'{base_name}_pred.png')
        segformer_path = os.path.join(args.segformer_dir, f'{base_name}_pred.png')
        vanilla_unet_path = os.path.join(args.vanilla_unet_dir, f'{base_name}_pred.png')

        # Check if all paths exist
        missing = []
        for path, name in [
            (unet_mffse_path, "UNetMFFSE"),
            (segformer_path, "Segformer"),
            (vanilla_unet_path, "VanillaUNet")
        ]:
            if not os.path.exists(path):
                missing.append(name)

        if missing:
            print(f"Skipping {base_name} due to missing prediction files for: {', '.join(missing)}")
            continue

        # Load images
        img_rgb = mpimg.imread(rgb_path)
        img_mffse = mpimg.imread(unet_mffse_path)
        img_segformer = mpimg.imread(segformer_path)
        img_vanilla = mpimg.imread(vanilla_unet_path)

        # Load Ground Truth mask (stored in segformer_dir as batch_{batch_idx}_sample_{sample_idx}_truth.png)
        truth_path = os.path.join(args.segformer_dir, f'{base_name}_truth.png')
        if not os.path.exists(truth_path):
            print(f"Skipping {base_name} due to missing ground truth file: {truth_path}")
            continue
        img_truth = mpimg.imread(truth_path)

        # Plot 5 things side-by-side (RGB, GT, VanillaUNet, UNetMFFSE, Segformer)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))

        axes[0].imshow(img_rgb)
        axes[0].set_title('RGB Image', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(img_truth)
        axes[1].set_title('Ground Truth', fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(img_vanilla)
        axes[2].set_title('Vanilla UNet (Author)', fontsize=10)
        axes[2].axis('off')

        axes[3].imshow(img_mffse)
        axes[3].set_title('UNetMFFSE_MS', fontsize=10)
        axes[3].axis('off')

        axes[4].imshow(img_segformer)
        axes[4].set_title('CustomSegformer_MS', fontsize=10)
        axes[4].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0)

        # Add color legend below the figure
        fig.legend(
            handles=legend_patches,
            loc='lower center',
            ncol=3,
            fontsize=8,
            frameon=True,
            bbox_to_anchor=(0.5, -0.02),
        )

        save_path = os.path.join(args.out_dir, f'{base_name}_combined.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"All combined figures saved to: {args.out_dir}")

if __name__ == '__main__':
    main()
