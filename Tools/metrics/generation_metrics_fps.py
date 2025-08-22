import numpy as np
import logging
import random
import glob
import argparse

from tqdm import tqdm
from pathlib import Path

from features.image_feature import Image
from utils.tables import load_computed_feature_from_folder

logger = logging.getLogger(__name__)


def get_npz(sample_paths, mode: str):
    bev_semantic_maps = []
    for sample_path in tqdm(sample_paths):

        if mode == "real":
            semantic_map_path = Path(sample_path) / "bev_semantic_map.gz"
        elif mode == "generated":
            semantic_map_path = Path(sample_path)
        else:
            raise ValueError

        semantic_map = load_computed_feature_from_folder(semantic_map_path, Image)
        bev_semantic_map = semantic_map.data

        # Check if ALL pixels are white ([255, 255, 255])
        all_white = np.all(bev_semantic_map == 255, axis=-1).all()

        # Check if ANY pixel is black ([0, 0, 0])
        any_black = np.all(bev_semantic_map == 0, axis=-1).any()

        if all_white or any_black:
            continue

        bev_semantic_maps.append(bev_semantic_map)

    bev_semantic_maps = np.stack(bev_semantic_maps)
    return bev_semantic_maps


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

    reference_maps = get_npz(reference_sample_paths, mode="real")
    generated_maps = get_npz(selected_generated_sample_paths, mode="generated")

    logger.info(
        f"Found {reference_maps.shape[0]} reference maps and {generated_maps.shape[0]} generated maps"
    )

    reference_save_path = Path(args.output_dir) / f"ref_batch_fps_{num_samples}.npz"
    generated_save_path = Path(args.output_dir) / f"samples_batch_fps_{num_samples}.npz"

    np.savez_compressed(reference_save_path, arr_0=reference_maps)
    np.savez_compressed(generated_save_path, arr_0=generated_maps)

    logger.info(
        f"Saved {reference_maps.shape[0]} reference images to {reference_save_path}."
    )
    logger.info(
        f"Saved {generated_maps.shape[0]} generated images to {generated_save_path}."
    )

    return None


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Set seeds for reproducibility
    args.seed = 0
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.num_samples = 10000
    args.generated_samples_cache_size = 50000
    args.reference_cache_path = Path(
        "/home/raniatze/Documents/skitti_workspace/cache/semantic_cache"
    )
    args.generated_samples_cache_path = Path(
         "/home/raniatze/Documents/PhD/Research/pyramid-discrete-diffusion/generated/s_1_to_s_2_50K/Rendering"
    )
    args.output_dir = Path(
        "/home/raniatze/Documents/PhD/Research/pyramid-discrete-diffusion/generated/s_1_to_s_2_50K/GenerationMetrics"
    )
    run_generation_metrics(args)


if __name__ == "__main__":
    main()
