import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import os
import argparse

from typing import Optional
from features.image_feature import Image

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='/home/raniatze/Documents/PhD/Research/pyramid-discrete-diffusion/generated/s_1_to_s_2_50K/PrevSceneContext')
parser.add_argument('--voxel_grid', action='store_false')
parser.add_argument('--voxel_size', type=float, default=2.0)  # ADJUST
parser.add_argument('--voxel_dims', type=int, nargs=3, default=[32, 32, 4], help='Dimensions of the voxel grid as [X Y Z]')  # ADJUST

opt = parser.parse_args()

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

def make_voxel_grid_from_points(points, colors, voxel_dims, voxel_size=0.25):
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = voxel_size
    voxel_grid.origin = [0.0, 0.0, 0.0]

    for i in range(points.shape[0]):
        x, y, z = points[i]
        r, g, b = colors[i]
        color = np.array([r, g, b]) / 255.0

        # new_x = y  # forward
        # new_y = (voxel_dims[0] / 2) - x  # right (centered)
        # new_z = z - (voxel_dims[2] - 1)  # down
        grid_index = (
            int(round(x)),
            int(round(y)),
            int(round(z))
        )

        voxel = o3d.geometry.Voxel(grid_index=grid_index, color=color)
        voxel_grid.add_voxel(voxel)

    return voxel_grid

def voxel_grid_to_cubes_with_wireframes(
    points, colors, voxel_dims, voxel_size=0.25, voxel_z_offset=0.5
):
    cubes = []
    wireframes = []

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
        x, y, z = points[i]
        r, g, b = colors[i]
        color = np.array([r, g, b]) / 255.0
        wire_color = color * 0.8  # Slightly darker version of fill color
        wire_color = np.clip(wire_color, 0, 1)

        # new_x = y  # forward
        # new_y = (voxel_dims[0] / 2) - x  # right (centered)
        # new_z = z - (voxel_dims[2] - 1)  # down
        center = np.array([x, y, z]) * voxel_size
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


# === MAIN LOOP ===
file_list = sorted(os.listdir(opt.folder))

renderer = o3d.visualization.rendering.OffscreenRenderer(256, 256)
renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
renderer.scene.view.set_post_processing(False)

for i, filename in enumerate(file_list):
    # if i != 3:  # TODO: Fix so that it works in a loop
        # continue
    file_path = os.path.join(opt.folder, filename)
    print(f"Visualizing: {file_path}")

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
    # voxel_grid = make_voxel_grid_from_points(points, colors, voxel_dims=opt.voxel_dims, voxel_size=opt.voxel_size)
    # bev_voxel_grid_rendering(renderer, voxel_grid, voxel_dims=opt.voxel_dims, voxel_size=opt.voxel_size)


    # For visualizing voxel grids
    voxel_grid, line_set = voxel_grid_to_cubes_with_wireframes(points, colors, voxel_dims=opt.voxel_dims, voxel_size=opt.voxel_size)
    normal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    o3d.visualization.draw_geometries([voxel_grid, line_set, normal_frame], window_name=f"Scene {i}")
