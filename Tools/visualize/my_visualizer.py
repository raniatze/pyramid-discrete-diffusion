import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import os
import argparse

from typing import Optional
from features.image_feature import Image

def infer_voxel_params(folder: str):
    if "s_1" in folder:
        stage = "s_1"
        voxel_size = 2.0
        voxel_dims = [32, 32, 4]
        origin = np.array([1.0, -31.0, -3.5])
    elif "s_2" in folder:
        stage = "s_2"
        voxel_size = 1.0
        voxel_dims = [64, 64, 8]
        origin = np.array([0.5, -31.5, -3.5])
    elif "s_3" in folder:
        stage = "s_3"
        voxel_size = 0.25
        voxel_dims = [256, 256, 16]
        origin = np.array([0.125, -31.875, -3.5])
    else:
        raise ValueError(f"Unknown voxel size: {folder}")

    return voxel_size, voxel_dims, origin, stage

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='/media/raniatze/Elements/PhD/Research/pyramid-discrete-diffusion/generated/s_3_50K/GeneratedFusion')
# parser.add_argument('--folder', default='/home/raniatze/Documents/PhD/Research/pyramid-discrete-diffusion/generated/s_1_50K/Generated')
parser.add_argument('--voxel_grid', action='store_false')

opt = parser.parse_args()
opt.voxel_size, opt.voxel_dims, opt.origin, opt.stage = infer_voxel_params(opt.folder)

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


def open3d_camera_setup(renderer, voxel_dims, voxel_size: float = 0.25):

    front_extent = voxel_dims[0] * voxel_size
    side_extent = (voxel_dims[1] / 2) * voxel_size

    # Camera setup when no remapping
    camera_target = np.array([32, 32, 0])
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
    renderer, voxel_mesh, voxel_dims, voxel_size: float = 0.25
) -> Optional[Image]:
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    renderer.scene.add_geometry("voxel_grid", voxel_mesh, material)

    open3d_camera_setup(renderer, voxel_dims=voxel_dims, voxel_size=voxel_size)
    rendered_image = renderer.render_to_image()
    rendered_image = np.asarray(rendered_image)

    plt.imshow(rendered_image)
    plt.show()

    # Filter out blank/empty renders
    if np.all(rendered_image == 0) or np.all(rendered_image == 255):
        return None

    renderer.scene.clear_geometry()
    return Image(data=rendered_image)

def make_voxel_grid_from_points(points, colors, remap=False):
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = opt.voxel_size
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
            max_x = opt.voxel_dims[1]  # was Y direction
            center_y = opt.voxel_dims[0] / 2  # was X direction

            # Step 3: Shift world so that this point becomes the origin
            world_x = -(rotated_x - max_x) * opt.voxel_size  # move back to 0
            world_y = (rotated_y - center_y) * opt.voxel_size  # move to center
            world_z = rotated_z * opt.voxel_size  # Z stays the same

            # Step 4: Create voxel at new position
            grid_index = (
                int(round(world_x / opt.voxel_size)),
                int(round(world_y / opt.voxel_size)),
                int(round(world_z / opt.voxel_size)),
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

def voxel_grid_to_cubes_with_wireframes(
    points, colors, voxel_dims, origin, stage, voxel_size=0.25, voxel_z_offset=0.5
):
    cubes = []
    wireframes = []
    if stage == 's_3':
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


# === MAIN LOOP ===
file_list = sorted(os.listdir(opt.folder))

renderer = o3d.visualization.rendering.OffscreenRenderer(256, 256)
renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
renderer.scene.view.set_post_processing(False)

for i, filename in enumerate(file_list):
    file_path = os.path.join(opt.folder, filename)
    scene_idx = os.path.basename(file_path).split('.')[0].split('_')[1]
    if scene_idx not in ['100', '10000', '10002']:
        continue
    print(f"Visualizing: {file_path}")
    print(scene_idx)

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

    # For BEV semantic map rendering only
    # voxel_grid = make_voxel_grid_from_points(points, colors, remap=False)
    # normal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    # o3d.visualization.draw_geometries([voxel_grid, normal_frame], window_name=f"Scene {i} before remapping")

    # voxel_grid = make_voxel_grid_from_points(points, colors, remap=True)
    # normal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    # o3d.visualization.draw_geometries([voxel_grid, normal_frame], window_name=f"Scene {i} after remapping")
    # bev_voxel_grid_rendering(renderer, voxel_grid, voxel_dims=opt.voxel_dims, voxel_size=opt.voxel_size)


    # For visualizing voxel grids
    voxel_grid, line_set = voxel_grid_to_cubes_with_wireframes(points, colors, voxel_dims=opt.voxel_dims, voxel_size=opt.voxel_size, origin=opt.origin, stage=opt.stage)
    normal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    o3d.visualization.draw_geometries([voxel_grid, line_set, normal_frame], window_name=f"Scene {os.path.basename(file_path)}")
