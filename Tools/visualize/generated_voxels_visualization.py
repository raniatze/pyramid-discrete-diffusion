import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import os
import argparse

from typing import Optional
from features.image_feature import Image
from pathlib import Path
from utils.line_mesh import LineMesh

def infer_voxel_params(folder: str):
    if "s_1" in folder:
        stage = "s_1"
        voxel_size = 2.0
        voxel_dims = [32, 32, 4]
        origin = np.array([1.0, -31.0, -7.5])
    elif "s_2" in folder or "PrevSceneContextFusion" in folder:
        stage = "s_2"
        voxel_size = 1.0
        voxel_dims = [64, 64, 8]
        origin = np.array([0.5, -31.5, -7.5])
    elif "s_3" in folder:
        stage = "s_3"
        voxel_size = 0.25
        voxel_dims = [256, 256, 16]
        origin = np.array([0.125, -31.875, -3.5])
    else:
        raise ValueError(f"Unknown voxel size: {folder}")

    return voxel_size, voxel_dims, origin, stage


version = "v1"
stage = "s_3"
sub_folder = "GeneratedFusion"

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default=f'/home/raniatze/Documents/PhD/Research/pyramid-discrete-diffusion/generated/{version}/{stage}_20/{sub_folder}')
parser.add_argument('--save_path', default=f'/home/raniatze/Documents/PhD/Research/pyramid-discrete-diffusion/generated/{version}/{stage}_20/Visualizations')
parser.add_argument('--voxel_grid', action='store_false')

opt = parser.parse_args()
opt.voxel_size, opt.voxel_dims, opt.origin, opt.stage = infer_voxel_params(opt.folder)

save_path = Path(opt.save_path)
save_path.mkdir(parents=True, exist_ok=True)

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


def voxel_grid_to_cubes_with_wireframes(
    points, colors, voxel_dims, origin, voxel_size=0.25, voxel_z_offset=0.5
):
    cubes = []
    wireframes = []

    nz = voxel_dims[2]
    z_min = voxel_z_offset - nz * voxel_size
    z_max = voxel_z_offset
    vz_eff = (z_max - z_min) / (nz - 1)
    voxel_size = np.array([voxel_size, voxel_size, vz_eff])

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

    for i in range(points.shape[0]):
        y, x, z = points[i]  # these are voxel indices and not world coordinates
        r, g, b = colors[i]
        color = np.array([r, g, b]) / 255.0
        wire_color = color * 0.8  # Slightly darker version of fill color
        wire_color = np.clip(wire_color, 0, 1)

        center = origin + np.array([x, y, z]) * voxel_size

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

        # verts = np.asarray(cube.vertices)
        # print("Min corner:", verts.min(axis=0))
        # print("Max corner:", verts.max(axis=0))
        # print("Cube center from verts:", (verts.min(axis=0) + verts.max(axis=0)) / 2)

        # Add wireframe
        wire = get_cube_lines(wire_color, voxel_size=voxel_size)
        wire.translate(center - 0.5 * voxel_size)
        wireframes.append(wire)

    combined_cubes = o3d.geometry.TriangleMesh()
    for cube in cubes:
        combined_cubes += cube
    combined_lineset = o3d.geometry.LineSet()
    for wf in wireframes:
        combined_lineset += wf

    return combined_cubes, combined_lineset

def save_screenshot(primitive_meshes, lookat_custom, eye_custom, save_file, render_bev: bool = False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Screenshot Viewer", width=1024, height=1024, visible=True)

    # Generate geometry
    for primitive in primitive_meshes:
        if isinstance(primitive, LineMesh):
            for cylinder in primitive.cylinder_segments:
                vis.add_geometry(cylinder)
        else:
            vis.add_geometry(primitive)

    vis.poll_events()
    vis.update_renderer()

    # Get view control
    view_ctl = vis.get_view_control()

    # render_options = vis.get_render_option()
    # render_options.line_width = 3

    # Compute front direction
    front = eye_custom - lookat_custom
    front /= np.linalg.norm(front)

    if render_bev:
        up = np.array([1, 0, 0])
    else:
        up = np.array([0, 0, -1])

    # Set camera
    view_ctl.set_lookat(lookat_custom.tolist())
    view_ctl.set_front(front.tolist())
    view_ctl.set_up(up.tolist())
    view_ctl.set_zoom(0.7)  # between 0.5 and 0.8

    # Update view and capture screenshot
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(save_file))

    vis.destroy_window()


# === MAIN LOOP ===
file_list = sorted(os.listdir(opt.folder))

renderer = o3d.visualization.rendering.OffscreenRenderer(256, 256)
renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
renderer.scene.view.set_post_processing(False)

for i, filename in enumerate(file_list):
    file_path = os.path.join(opt.folder, filename)
    scene_idx = os.path.basename(file_path).split('.')[0].split('_')[1]
    if 'v1' in file_path and scene_idx not in ['4', '11', '14', '15', '16']:
        continue
    elif 'v2' in file_path and scene_idx not in ['1', '8', '11', '14']:
        continue
    else:
        pass
    print(f"Visualizing: {file_path}")
    # print(scene_idx)

    if os.path.getsize(file_path) == 0:
        print("Skipping empty file.")
        continue

    points_colors = np.loadtxt(file_path, delimiter=' ')
    if points_colors.shape[1] != 4:
        print(f"Invalid format in file: {filename}. Expected x y z label.")
        continue

    points = points_colors[:, -3:]
    labels = points_colors[:, 0]
    colors = np.array([pritti_colors[int(l)] for l in labels])

    # For visualizing voxel grids
    voxel_grid, line_set = voxel_grid_to_cubes_with_wireframes(points, colors, voxel_dims=opt.voxel_dims, voxel_size=opt.voxel_size, origin=opt.origin)
    # normal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    # o3d.visualization.draw_geometries([voxel_grid, line_set], window_name=f"Scene {os.path.basename(file_path)}")

    lookat_custom = np.array([32, 0, 0], dtype=float)
    eye_custom = np.array([-20, 0, -50], dtype=float)
    save_screenshot([voxel_grid, line_set], lookat_custom, eye_custom, save_file=Path(opt.save_path) / f"{scene_idx}.png")
