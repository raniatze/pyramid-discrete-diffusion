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


def voxel_grid_to_cubes(
    points, colors, voxel_z_offset=0.5
):
    cubes = []
    origin = np.array([0.125, -31.875, -3.5])
    nz = 16
    vz = 0.25
    z_min = voxel_z_offset - nz * vz
    z_max = voxel_z_offset
    vz_eff = (z_max - z_min) / (nz - 1)
    voxel_size = np.array([0.25, 0.25, vz_eff])

    for i in range(points.shape[0]):
        y, x, z = points[i]  # these are voxel indices and not world coordinates
        r, g, b = colors[i]
        color = np.array([r, g, b]) / 255.0

        center = origin + np.array([x, y, z]) * voxel_size

        # Add cube
        cube = o3d.geometry.TriangleMesh.create_box(
            width=voxel_size[0], height=voxel_size[1], depth=voxel_size[2]
        )

        cube.translate(center - 0.5 * voxel_size)
        cube.compute_vertex_normals()
        cube.paint_uniform_color(color)
        cubes.append(cube)

        # verts = np.asarray(cube.vertices)
        # print("Min corner:", verts.min(axis=0))
        # print("Max corner:", verts.max(axis=0))
        # print("Cube center from verts:", (verts.min(axis=0) + verts.max(axis=0)) / 2)

    combined_cubes = o3d.geometry.TriangleMesh()
    for cube in cubes:
        combined_cubes += cube

    return combined_cubes

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
            file = re.search(r'merged_(\d+)\.txt$', os.path.basename(target_path)).group(1)
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

            voxel_grid = voxel_grid_to_cubes(points, colors)
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
