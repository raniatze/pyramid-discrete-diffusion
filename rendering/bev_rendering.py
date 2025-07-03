import os
import glob
import numpy as np
import open3d as o3d
import uuid
import gc
import logging

from omegaconf import DictConfig
from typing import Dict, List, Union
from pathlib import Path
from typing import Optional

from features.image_feature import Image
from features.voxel_feature import VoxelGrid
from utils.tables import store_computed_feature_to_folder, load_computed_feature_from_folder
from multithreading.worker_utils import worker_map
from multithreading.worker_pool import WorkerPool
from Tools.visualize.voxel_grids_visualization import make_voxel_grid

logger = logging.getLogger(__name__)

def open3d_camera_setup(renderer, cfg: DictConfig):

    front_extent = cfg.grid_shape[0] * cfg.voxel_size
    side_extent = (cfg.grid_shape[1] / 2) * cfg.voxel_size

    # Camera setup when no remapping
    # camera_target = np.array([32, 32, 0])
    # camera_position = camera_target - np.array([0, 0, 1]) * 50
    # camera_up = np.array([0, 1, 0])  # X axis is the "up" direction

    # Camera setup when remapping
    camera_target = np.array([32, 0, 0])
    camera_position = camera_target - np.array([0, 0, 1]) * 50
    camera_up = np.array([1, 0, 0])  # X axis is the "up" direction

    renderer.scene.camera.look_at(camera_target, camera_position, camera_up)

    # Define the orthographic projection parameters
    left = -side_extent
    right = side_extent
    bottom = -front_extent / 2
    top = front_extent / 2
    near = 1
    far = 100.0

    # Set the orthographic projection
    renderer.scene.camera.set_projection(
        projection_type=o3d.visualization.rendering.Camera.Projection.Ortho,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        near=near,
        far=far,
    )
    return

def bev_voxel_grid_rendering(
    renderer, voxel_grid: VoxelGrid, cfg: DictConfig
) -> Optional[Image]:
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    # Convert voxel grid to cube mesh
    voxel_mesh = make_voxel_grid(voxel_grid.data, voxel_size=1.0, remap=True)

    renderer.scene.add_geometry("voxel_grid", voxel_mesh, material)

    open3d_camera_setup(renderer, cfg)
    rendered_image = renderer.render_to_image()
    rendered_image = np.asarray(rendered_image)

    # Filter out blank/empty renders
    if np.all(rendered_image == 0) or np.all(rendered_image == 255):
        return None

    renderer.scene.clear_geometry()
    return Image(data=rendered_image)


def run_semantic_map_rendering(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Applies the diffusion model generate and cache scenarios.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """

    target_paths = glob.glob(os.path.join(cfg.target_path, "*.gz"))

    assert len(target_paths) == cfg.generated_samples_cache_size

    data_points = [
        {
            "target_path": target_path,
            "cfg": cfg,
        }
        for target_path in target_paths
    ]

    logger.info("Starting semantic BEV rendering of %s files...", str(len(data_points)))

    _ = worker_map(worker, semantic_map_rendering, data_points)
    logger.info("Completed semantic BEV rendering!")

    return None


def semantic_map_rendering(
    data_points: List[Dict[str, Union[Path, DictConfig]]]
) -> None:
    """
    Process and cache a single sample.
    :param sample: Dictionary containing frame and object information for a single frame.
    """

    def semantic_map_rendering_internal(
        data_points: List[Dict[str, Union[Path, DictConfig]]]
    ) -> None:
        thread_id = str(uuid.uuid4())

        target_paths: List[Path] = [d["target_path"] for d in data_points]
        cfg: DictConfig = data_points[0]["cfg"]

        # Prepare the renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(
            cfg.pixel_width, cfg.pixel_height
        )
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        renderer.scene.view.set_post_processing(False)

        logger.info(
            f"Extracted {len(target_paths)} scenarios for thread_id={thread_id}"
        )

        for idx, target_path in enumerate(target_paths):
            logger.info(
                f"Processing scenario {idx + 1} / {len(target_paths)} in thread_id={thread_id}"
            )

            save_path = Path(cfg.save_path)
            os.makedirs(save_path, exist_ok=True)
            semantic_file_path = save_path / f"bev_semantic_map_{idx}"
            if semantic_file_path.with_suffix(".gz").exists():
                logger.info(f"Semantic file path {semantic_file_path} already exists!")
                continue

            target_voxel_grid = load_computed_feature_from_folder(
                Path(target_path), VoxelGrid
            )

            bev_semantic_map: Image = bev_voxel_grid_rendering(
                renderer, target_voxel_grid, cfg
            )

            if bev_semantic_map is None:
                logger.info(f"Empty semantic map found for {idx}")
                break

            if (bev_semantic_map.data == 0).all(axis=-1).any():
                logger.info(
                    f"Semantic map for sample {idx} contains black pixels — skipping."
                )
                break

            semantic_file_path.parent.mkdir(parents=True, exist_ok=True)
            store_computed_feature_to_folder(semantic_file_path, bev_semantic_map)

            logger.info(
                f"Saved semantic map for sample {idx} at {semantic_file_path}."
            )

        logger.info(f"Finished processing scenarios for thread_id={thread_id}")
        return None

    result = semantic_map_rendering_internal(data_points)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result
