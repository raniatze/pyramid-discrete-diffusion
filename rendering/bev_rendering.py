import os
import glob
import numpy as np
import open3d as o3d
import uuid
import gc
import re
import logging
import matplotlib.pyplot as plt

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

# Color map for semantic labels
pritti_colors = {
    0: [0, 0, 0],
    1: [81, 0, 81],
    2: [152, 251, 152],
    3: [244, 35, 232],
    4: [250, 170, 160],
    5: [128, 64, 128],
    6: [107, 142, 35],
    7: [107, 142, 35],
    8: [0, 60, 100],
    9: [0, 0, 142],
    10: [119, 11, 32],
    11: [220, 20, 60],
    12: [70, 70, 70],
    13: [102, 102, 156],
    14: [153, 153, 153],
    15: [250, 170, 30],
    16: [0, 128, 192],
}

def open3d_camera_setup(renderer, grid_shape, voxel_size):

    front_extent = grid_shape[0] * voxel_size
    side_extent = (grid_shape[1] / 2) * voxel_size

    # Camera setup when no remapping
    camera_target = np.array([32, 0, 0])
    camera_position = camera_target - np.array([0, 0, 1]) * 50
    camera_up = np.array([1, 0, 0])  # X axis is the "up" direction

    # Camera setup when remapping
    # camera_target = np.array([32, 0, 0])
    # camera_position = camera_target - np.array([0, 0, 1]) * 50
    # camera_up = np.array([1, 0, 0])  # X axis is the "up" direction

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
    renderer, voxel_mesh, grid_shape, voxel_size) -> Optional[Image]:

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    renderer.scene.add_geometry("voxel_grid", voxel_mesh, material)

    open3d_camera_setup(renderer, grid_shape=grid_shape, voxel_size=voxel_size)
    rendered_image = renderer.render_to_image()
    rendered_image = np.asarray(rendered_image)

    # plt.imshow(rendered_image)
    # plt.show()

    # Filter out blank/empty renders
    if np.all(rendered_image == 0) or np.all(rendered_image == 255):
        return None

    renderer.scene.clear_geometry()
    return Image(data=rendered_image)

def make_voxel_grid_from_points(points, colors, cfg, remap=True):
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = cfg.voxel_size
    voxel_grid.origin = [0.0, 0.0, 0.0]

    for i in range(points.shape[0]):
        x, y, z = points[i]
        r, g, b = colors[i]
        color = np.array([r, g, b]) / 255.0

        # Step 1: Re-orient axes (X = forward, Y = right, Z = up)
        if remap:
            rotated_x = y
            rotated_y = x
            rotated_z = z

            # Step 2: Set ego vehicle at (max_x, center_y)
            max_x = cfg.grid_shape[1]  # was Y direction
            center_y = cfg.grid_shape[0] / 2  # was X direction

            # Step 3: Shift world so that this point becomes the origin
            world_x = -(rotated_x - max_x) * cfg.voxel_size  # move back to 0
            world_y = (rotated_y - center_y) * cfg.voxel_size  # move to center
            world_z = rotated_z * cfg.voxel_size  # Z stays the same

            # Step 4: Create voxel at new position
            grid_index = (
                int(round(world_x / cfg.voxel_size)),
                int(round(world_y / cfg.voxel_size)),
                int(round(world_z / cfg.voxel_size)),
            )
        else:
            grid_index = (
                int(round(x)),
                int(round(y)),
                int(round(z))
        )

        voxel = o3d.geometry.Voxel(grid_index=grid_index, color=color)
        voxel_grid.add_voxel(voxel)

    return voxel_grid


def run_semantic_map_rendering(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Applies the diffusion model generate and cache scenarios.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """

    target_paths = glob.glob(os.path.join(cfg.target_path, "*.txt"))

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

        logger.info(
            f"Extracted {len(target_paths)} scenarios for thread_id={thread_id}"
        )

        renderer = o3d.visualization.rendering.OffscreenRenderer(cfg.pixel_width, cfg.pixel_height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        renderer.scene.view.set_post_processing(False)

        for idx, target_path in enumerate(target_paths):
            logger.info(
                f"Processing scenario {idx + 1} / {len(target_paths)} in thread_id={thread_id}"
            )

            save_path = Path(cfg.save_path)
            os.makedirs(save_path, exist_ok=True)
            file = re.search(r'result_(\d+_\d+)\.txt$', os.path.basename(target_path)).group(1)
            semantic_file_path = save_path / f"bev_semantic_map_{file}"
            if semantic_file_path.with_suffix(".gz").exists():
                logger.info(f"Semantic file path {semantic_file_path} already exists!")
                continue

            points_colors = np.loadtxt(target_path, delimiter=' ')
            if points_colors.shape[1] != 4:
                print(f"Invalid format in file: {target_path}. Expected x y z label.")
                continue

            points = points_colors[:, -3:]
            labels = points_colors[:, 0]
            colors = np.array([pritti_colors[int(l)] for l in labels])

            voxel_grid = make_voxel_grid_from_points(points, colors, cfg)
            bev_semantic_map: Image = bev_voxel_grid_rendering(
                renderer, voxel_grid, grid_shape=cfg.grid_shape, voxel_size=cfg.voxel_size
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
