import numpy as np
import logging
import random
import glob
import argparse

from tqdm import tqdm
from pathlib import Path

from features.image_feature import Image
from utils.tables import load_computed_feature_from_folder
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def save_images_grid_pdf(images, titles, pdf_path, images_per_page=24, ncols=6, dpi=150):
    """
    images: list/array of HxWx{1|3} uint8
    titles: list of strings (same length), shown above each tile
    """
    assert len(images) == len(titles)
    nrows = int(np.ceil(images_per_page / ncols))
    with PdfPages(pdf_path) as pdf:
        for img_batch, title_batch in zip(_chunks(images, images_per_page),
                                          _chunks(titles, images_per_page)):
            rows = int(np.ceil(len(img_batch) / ncols))
            fig, axes = plt.subplots(rows, ncols, figsize=(ncols*2.2, rows*2.2), dpi=dpi)
            if rows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = np.array([axes])
            axes = axes.reshape(rows, ncols)

            for ax in axes.ravel():
                ax.axis('off')

            for ax, img, ttl in zip(axes.ravel(), img_batch, title_batch):
                ax.imshow(img)
                ax.set_title(Path(ttl).name, fontsize=6)
                ax.axis('off')

            plt.tight_layout(pad=0.5)
            pdf.savefig(fig)
            plt.close(fig)

def save_pairs_pdf(ref_images, ref_titles, gen_images, gen_titles, pdf_path, rows_per_page=8, dpi=150):
    """
    Side-by-side reference vs generated (2 columns). Uses the min length of the two lists.
    """
    L = min(len(ref_images), len(gen_images))
    ref_images, ref_titles = ref_images[:L], ref_titles[:L]
    gen_images, gen_titles = gen_images[:L], gen_titles[:L]

    with PdfPages(pdf_path) as pdf:
        per_page = rows_per_page
        for i in range(0, L, per_page):
            rimgs = ref_images[i:i+per_page]
            gimgs = gen_images[i:i+per_page]
            rtls  = ref_titles[i:i+per_page]
            gtls  = gen_titles[i:i+per_page]

            rows = len(rimgs)
            fig, axes = plt.subplots(rows, 2, figsize=(2*3.0, rows*2.0), dpi=dpi)  # 2 columns
            if rows == 1:
                axes = np.array([axes])
            for r in range(rows):
                # left = reference
                axes[r, 0].imshow(rimgs[r]); axes[r, 0].set_title(f"REF: {Path(rtls[r]).name}", fontsize=7); axes[r, 0].axis('off')
                # right = generated
                axes[r, 1].imshow(gimgs[r]); axes[r, 1].set_title(f"GEN: {Path(gtls[r]).name}", fontsize=7); axes[r, 1].axis('off')

            plt.tight_layout(pad=0.5)
            pdf.savefig(fig)
            plt.close(fig)

    return

def get_npz(sample_paths, mode: str):
    bev_semantic_maps = []
    kept_paths = []
    for sample_path in tqdm(sample_paths):

        if mode == "real":
            semantic_map_path = Path(sample_path) / "bev_semantic_map.gz"
        elif mode == "generated":
            semantic_map_path = Path(sample_path)
        else:
            raise ValueError

        semantic_map = load_computed_feature_from_folder(semantic_map_path, Image)
        bev_semantic_map = semantic_map.data

        # filters (same as before)
        all_white = np.all(bev_semantic_map == 255, axis=-1).all()
        any_black = np.all(bev_semantic_map == 0, axis=-1).any()
        if all_white or any_black:
            continue

        bev_semantic_maps.append(bev_semantic_map)
        kept_paths.append(str(semantic_map_path))

    bev_semantic_maps = np.stack(bev_semantic_maps)
    return bev_semantic_maps, kept_paths

def run_generation_metrics(args) -> None:

    assert (
        args.reference_cache_path is not None
    ), "args.reference_cache_path is not specified!"
    assert (
        args.generated_samples_cache_path is not None
    ), "args.generated_samples_cache_path is not specified!"

    num_samples = args.num_samples
    real_samples_cache_path = Path(args.reference_cache_path)
    reference_batch_path = (
        real_samples_cache_path / f"reference_batch_{num_samples}.txt"
    )
    assert (
        reference_batch_path.exists()
    ), f"Reference batch file not found: {reference_batch_path}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_sample_paths = []

    with open(reference_batch_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            sequence_name = parts[1]
            frame_id = parts[2]

            sample_path = (
                real_samples_cache_path
                / sequence_name
                / "train"
                / str(int(float(frame_id)))
            )
            reference_sample_paths.append(sample_path)

    logger.info(
        f"Loaded {len(reference_sample_paths)} reference paths from {reference_batch_path}"
    )

    generated_sample_paths = glob.glob(
        f"{args.generated_samples_cache_path}/bev_semantic_map_*.gz"
    )
    assert len(generated_sample_paths) == args.generated_samples_cache_size

    selected_generated_sample_paths = random.sample(
        generated_sample_paths, min(num_samples, len(generated_sample_paths))
    )

    assert (
        len(reference_sample_paths)
        == len(selected_generated_sample_paths)
        == num_samples
    )

    reference_maps, reference_kept_paths = get_npz(reference_sample_paths, mode="real")
    generated_maps, generated_kept_paths = get_npz(selected_generated_sample_paths, mode="generated")

    logger.info(
        f"Found {reference_maps.shape[0]} reference maps and {generated_maps.shape[0]} generated maps"
    )

    # PDFs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_ref = args.output_dir / f"ref_used_for_fps_{reference_maps.shape[0]}.pdf"
    pdf_gen = args.output_dir / f"gen_used_for_fps_{generated_maps.shape[0]}.pdf"

    # single-set PDFs
    save_images_grid_pdf(
        list(reference_maps), reference_kept_paths, pdf_ref, images_per_page=24, ncols=6
    )
    save_images_grid_pdf(
        list(generated_maps), generated_kept_paths, pdf_gen, images_per_page=24, ncols=6
    )

    logger.info(f"Saved PDF of reference samples actually used: {pdf_ref}")
    logger.info(f"Saved PDF of generated samples actually used: {pdf_gen}")

    reference_save_path = Path(args.output_dir) / f"ref_batch_fps_{num_samples}.npz"
    generated_save_path = Path(args.output_dir) / f"samples_batch_fps_{num_samples}.npz"

    np.savez_compressed(reference_save_path, arr_0=reference_maps)
    np.savez_compressed(generated_save_path, arr_0=generated_maps)

    logger.info(
        f"Saved {reference_maps.shape[0]} reference maps to {reference_save_path}."
    )
    logger.info(
        f"Saved {generated_maps.shape[0]} generated maps to {generated_save_path}."
    )

    return None

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Set seeds for reproducibility
    args.seed = 0
    random.seed(args.seed)
    np.random.seed(args.seed)

    stage = "s_1"
    args.num_samples = 5000
    args.generated_samples_cache_size = 50000
    args.reference_cache_path = Path(
        "/home/raniatze/Documents/skitti_workspace/cache/semantic_cache"
    )
    args.generated_samples_cache_path = Path(
         f"/media/raniatze/Elements/PhD/Research/pyramid-discrete-diffusion/generated/{stage}_50K_no_augmentation/Rendering"
    )
    args.output_dir = Path(
        f"/media/raniatze/Elements/PhD/Research/pyramid-discrete-diffusion/generated/{stage}_50K_no_augmentation/GenerationMetrics"
    )
    run_generation_metrics(args)


if __name__ == "__main__":
    main()
