import os
import glob

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import argparse
from typing import List, Dict
from pathlib import Path
from features.voxel_feature import VoxelGrid
from Tools.visualize.visualizer import pritti_colors
from utils.tables import load_computed_feature_from_folder


def voxel_grid_to_cubes_with_wireframes(
    voxel_grid_data, voxel_size=0.25, voxel_z_offset=0.5
):
    cubes = []
    wireframes = []
    dims = voxel_grid_data.shape

    def get_cube_lines(wire_color, voxel_size):
        points = (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ]
            )
            * voxel_size
        )

        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([wire_color for _ in lines])
        return line_set

    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                label = int(voxel_grid_data[x, y, z])
                if label > 0:  # occupied voxel
                    color = np.array(pritti_colors[label]) / 255
                    wire_color = color * 0.8  # Slightly darker version of fill color
                    wire_color = np.clip(wire_color, 0, 1)
                    center = (
                        np.array([x, y - (dims[1] / 2), z - dims[2]]) * voxel_size
                    )  # Change voxel origin to match with that from primitives
                    center[2] += voxel_z_offset

                    # Add cube
                    cube = o3d.geometry.TriangleMesh.create_box(
                        width=voxel_size, height=voxel_size, depth=voxel_size
                    )

                    cube.translate(center)
                    cube.compute_vertex_normals()
                    cube.paint_uniform_color(color)
                    cubes.append(cube)

                    # Add wireframe
                    wire = get_cube_lines(wire_color, voxel_size=voxel_size)
                    wire.translate(center)
                    wireframes.append(wire)

    combined_cubes = o3d.geometry.TriangleMesh()
    for cube in cubes:
        combined_cubes += cube
    combined_lineset = o3d.geometry.LineSet()
    for wf in wireframes:
        combined_lineset += wf

    return combined_cubes, combined_lineset

def make_voxel_grid(voxel_data, voxel_size: float = 0.25, voxel_z_offset: float = 0.5, remap: bool = False):

    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = voxel_size

    shape_x, shape_y, shape_z = voxel_data.shape

    for x in range(shape_x):
        for y in range(shape_y):
            for z in range(shape_z):
                if voxel_data[x, y, z]:
                    label = voxel_data[x, y, z].item()
                    color = np.array(pritti_colors[label]) / 255

                    # Apply axis remapping
                    if remap:
                        new_x = x - (shape_x / 2)
                        new_y = y - (shape_y / 2)
                        new_z = (z - shape_z) + 0.5
                        # new_x = y
                        # new_y = (shape_x - 1 - x) - center_y  # right (flipped x)
                        # new_z = (z - shape_z) + voxel_z_offset
                        voxel_index = (new_x, new_y, new_z)
                    else:
                        voxel_index = (x, y, z)
                    vox = o3d.geometry.Voxel(
                        grid_index=voxel_index,
                        color=color,
                    )
                    voxel_grid.add_voxel(vox)

    return voxel_grid


def split_sample_paths(cache_path: Path, feature_names: List[str]) -> Dict[str, List[Path]]:
    sample_path_dict: Dict[str, List[Path]] = {split: [] for split in ["train", "val"]}
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


def run_visualization(args):

    target_paths = glob.glob(os.path.join(args.target_path, "*.gz"))
    # target_paths = split_sample_paths(args.target_path, feature_names=["voxel_grid"])["val"]

    for idx, target_path in enumerate(target_paths):

        target_voxel_grid = load_computed_feature_from_folder(
            Path(target_path), VoxelGrid
        )

        # combined_cubes, combined_wireframes = voxel_grid_to_cubes_with_wireframes(
            # target_voxel_grid.data
        # )
        voxel_grid_after = make_voxel_grid(target_voxel_grid.data, args.voxel_size, remap=False)
        # voxel_grid_before = make_voxel_grid(target_voxel_grid.data, args.voxel_size, remap=False)

        # Open3D visualization
        normal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        # o3d.visualization.draw_geometries([voxel_grid_before, normal_frame], window_name="Before")
        o3d.visualization.draw_geometries([voxel_grid_after, normal_frame], window_name="After")
        # o3d.visualization.draw_geometries(
        # [combined_cubes, combined_wireframes, normal_frame], window_name="Generated Voxel Grid"
        # )

    return


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.target_path = Path("/home/raniatze/Documents/PhD/Research/pyramid-discrete-diffusion/generated/s_1/Generated")
    args.voxel_size = 0.25
    run_visualization(args)


if __name__ == "__main__":
    main()
