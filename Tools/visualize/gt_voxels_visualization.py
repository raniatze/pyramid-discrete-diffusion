import random
import numpy as np
import open3d as o3d

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from features.voxel_feature import VoxelGrid
from utils.tables import load_computed_feature_from_folder

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

def infer_voxel_params(folder: str):
    if "32" in folder:
        voxel_size = 2.0
        voxel_dims = [32, 32, 4]
        origin = np.array([1.0, -31.0, -7.5])
    elif "64" in folder or "PrevSceneContextFusion" in folder:
        voxel_size = 1.0
        voxel_dims = [64, 64, 8]
        origin = np.array([0.5, -31.5, -7.5])
    elif "256" in folder:
        voxel_size = 0.25
        voxel_dims = [256, 256, 16]
        origin = np.array([0.125, -31.875, -3.5])
    else:
        raise ValueError(f"Unknown voxel size: {folder}")

    return voxel_size, voxel_dims, origin

def split_sample_paths(cache_path: Path, feature_names: List[str]) -> Dict[str, List[Path]]:
    sample_path_dict: Dict[str, List[Path]] = {split: [] for split in ["train", "val", "ignore"]}

    for sequence_path in cache_path.iterdir():
        if not sequence_path.is_dir():
            continue
        for split_path in sequence_path.iterdir():
            for sample_path in split_path.iterdir():
                has_features = [
                    (sample_path / f"{feature_name}.gz").is_file() for feature_name in feature_names
                ]

                if all(has_features):
                    sample_path_dict[split_path.name].append(sample_path)

    return sample_path_dict

def voxel_grid_to_cubes(voxel_grid_data, origin, voxel_size=0.25, voxel_z_offset=0.5):
    cubes = []
    dims = voxel_grid_data.shape

    nz = dims[2]
    z_min = voxel_z_offset - nz * voxel_size
    z_max = voxel_z_offset
    vz_eff = (z_max - z_min) / (nz - 1)
    voxel_size = np.array([voxel_size, voxel_size, vz_eff])

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if voxel_grid_data[i, j, k] > 0:  # occupied voxel
                    label = voxel_grid_data[i, j, k]
                    color = np.array(pritti_colors[int(label)]) / 255

                    center = origin + np.array([i, j, k]) * voxel_size

                    # Add cube
                    if isinstance(voxel_size, (tuple, list, np.ndarray)):
                        cube = o3d.geometry.TriangleMesh.create_box(
                            width=voxel_size[0],
                            height=voxel_size[1],
                            depth=voxel_size[2]
                        )
                    elif isinstance(voxel_size, (int, float)):
                        # same size in all dimensions
                        cube = o3d.geometry.TriangleMesh.create_box(
                            width=voxel_size,
                            height=voxel_size,
                            depth=voxel_size
                        )
                    else:
                        raise TypeError(f"voxel_size must be a float or tuple of 3, got {type(voxel_size)}")

                    cube.translate(center - 0.5 * voxel_size)
                    cube.compute_vertex_normals()
                    cube.paint_uniform_color(color)
                    cubes.append(cube)

    combined_cubes = o3d.geometry.TriangleMesh()
    for cube in cubes:
        combined_cubes += cube

    return combined_cubes


def visualize_voxel_grids(voxel_cache_path: Path):

    sample_path_dict = split_sample_paths(voxel_cache_path, feature_names=["voxel_grid"])
    voxel_size, _, origin = infer_voxel_params(str(voxel_cache_path))

    for split in ["train", "val"]:
        sample_paths = sample_path_dict.get(split)
        random.shuffle(sample_paths)
        for i, sample_path in enumerate(tqdm(sample_paths)):

            # Voxel grid visualization
            voxel_feature_path = sample_path / "voxel_grid.gz"
            voxel_grid = load_computed_feature_from_folder(
                voxel_feature_path, VoxelGrid
            )
            voxel_mesh = voxel_grid_to_cubes(voxel_grid.data, origin=origin, voxel_size=voxel_size)
            normal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
            o3d.visualization.draw_geometries(
                 [voxel_mesh, normal_frame], window_name="Target Voxel Grid"
            )

voxel_cache_path = Path("/home/raniatze/Documents/skitti_workspace/cache/pdd_cache/voxel_cache_256")
visualize_voxel_grids(voxel_cache_path)
