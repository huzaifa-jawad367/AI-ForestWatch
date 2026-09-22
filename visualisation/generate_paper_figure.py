"""
Generate a publication-quality comparison figure for the paper.

Layout:
    Columns: RGB Image | Ground Truth | Vanilla UNet | Proposed Model | Segformer
    Rows:    One per selected test sample (5 rows)

Color Legend (bottom):
    ■ Background / No-Data  (#1E293B)
    ■ Non-Forest            (#E53E3E)
    ■ Forest                (#38A169)
"""
import os
import sys

# Ensure project root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── Configuration ────────────────────────────────────────────────────
RGB_DIR = 'saved/Outputs/RGB_Images'
UNET_MFFSE_DIR = 'saved/Outputs/QA_Landsat8_UNetMFFSE_MS_0627_122418'
SEGFORMER_DIR = 'saved/Outputs/QA_Landsat8_CustomSegformer_MS_0628_210105'
VANILLA_UNET_DIR = 'saved/Outputs/QA_vanilla_unet_author_PretrainedModel'
OUT_DIR = 'saved/Outputs'

# Samples to include (row order)
SAMPLES = [
    'batch_0_sample_16',
    'batch_0_sample_30',
    'batch_0_sample_7',
    'batch_0_sample_20',
    'batch_0_sample_31',
]

# Column titles matching the paper
COL_TITLES = [
    'Landsat-8 RGB',
    'Ground Truth',
    'Vanilla UNet',
    'Proposed Model',
    'Segformer',
]

# Color legend
LEGEND_PATCHES = [
    mpatches.Patch(facecolor='#1E293B', edgecolor='black', linewidth=0.5,
                   label='Background / No-Data'),
    mpatches.Patch(facecolor='#E53E3E', edgecolor='black', linewidth=0.5,
                   label='Non-Forest'),
    mpatches.Patch(facecolor='#38A169', edgecolor='black', linewidth=0.5,
                   label='Forest'),
]


def load_sample_images(base_name):
    """Return (rgb, truth, vanilla, mffse, segformer) image arrays."""
    rgb = mpimg.imread(os.path.join(RGB_DIR, f'{base_name}_rgb.png'))
    truth = mpimg.imread(os.path.join(SEGFORMER_DIR, f'{base_name}_truth.png'))
    vanilla = mpimg.imread(os.path.join(VANILLA_UNET_DIR, f'{base_name}_pred.png'))
    mffse = mpimg.imread(os.path.join(UNET_MFFSE_DIR, f'{base_name}_pred.png'))
    segformer = mpimg.imread(os.path.join(SEGFORMER_DIR, f'{base_name}_pred.png'))
    return rgb, truth, vanilla, mffse, segformer


def main():
    n_rows = len(SAMPLES)
    n_cols = 5

    # ── Figure setup ─────────────────────────────────────────────────
    # Use gridspec for precise control over spacing
    fig = plt.figure(figsize=(10, 2.2 * n_rows + 0.6), dpi=300)

    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,                   # extra row for legend
        height_ratios=[1] * n_rows + [0.08],  # legend row is thin
        hspace=0.08, wspace=0.04,
        left=0.01, right=0.99, top=0.94, bottom=0.01,
    )

    for row_idx, sample_name in enumerate(SAMPLES):
        rgb, truth, vanilla, mffse, segformer = load_sample_images(sample_name)
        images = [rgb, truth, vanilla, mffse, segformer]

        for col_idx, img in enumerate(images):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            # Thin border around each subplot
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color('#999999')

            # Column titles on the first row only
            if row_idx == 0:
                ax.set_title(COL_TITLES[col_idx], fontsize=13,
                             fontweight='bold', pad=6,
                             fontfamily='serif')

            # Row labels (a), (b), ... on the first column
            if col_idx == 0:
                row_letter = chr(ord('a') + row_idx)
                ax.set_ylabel(f'({row_letter})', fontsize=13,
                              fontweight='bold', rotation=0,
                              labelpad=18, va='center',
                              fontfamily='serif')

    # ── Legend row ───────────────────────────────────────────────────
    ax_legend = fig.add_subplot(gs[n_rows, :])
    ax_legend.axis('off')
    ax_legend.legend(
        handles=LEGEND_PATCHES,
        loc='center',
        ncol=3,
        fontsize=11,
        frameon=True,
        edgecolor='#cccccc',
        fancybox=False,
        handlelength=1.4,
        handleheight=1.0,
        columnspacing=2.5,
        prop={'family': 'serif'},
    )

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'paper_figure_predictions.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    # Also save a PDF version for LaTeX
    out_pdf = os.path.join(OUT_DIR, 'paper_figure_predictions.pdf')
    fig2 = plt.figure(figsize=(10, 2.2 * n_rows + 0.6), dpi=300)
    gs2 = gridspec.GridSpec(
        n_rows + 1, n_cols,
        height_ratios=[1] * n_rows + [0.08],
        hspace=0.08, wspace=0.04,
        left=0.01, right=0.99, top=0.94, bottom=0.01,
    )

    for row_idx, sample_name in enumerate(SAMPLES):
        rgb, truth, vanilla, mffse, segformer = load_sample_images(sample_name)
        images = [rgb, truth, vanilla, mffse, segformer]
        for col_idx, img in enumerate(images):
            ax = fig2.add_subplot(gs2[row_idx, col_idx])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color('#999999')
            if row_idx == 0:
                ax.set_title(COL_TITLES[col_idx], fontsize=13,
                             fontweight='bold', pad=6,
                             fontfamily='serif')
            if col_idx == 0:
                row_letter = chr(ord('a') + row_idx)
                ax.set_ylabel(f'({row_letter})', fontsize=13,
                              fontweight='bold', rotation=0,
                              labelpad=18, va='center',
                              fontfamily='serif')

    ax_legend2 = fig2.add_subplot(gs2[n_rows, :])
    ax_legend2.axis('off')
    ax_legend2.legend(
        handles=LEGEND_PATCHES,
        loc='center',
        ncol=3,
        fontsize=11,
        frameon=True,
        edgecolor='#cccccc',
        fancybox=False,
        handlelength=1.4,
        handleheight=1.0,
        columnspacing=2.5,
        prop={'family': 'serif'},
    )

    fig2.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig2)

    print(f"Saved PNG: {out_path}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == '__main__':
    main()
