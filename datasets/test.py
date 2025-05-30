from pathlib import Path
import gzip
import pickle
import open3d as o3d
import numpy as np
from tqdm import tqdm

skitti_colors = {
    0: (0, 0, 0),
    1: (81, 0, 81),  # ground
    2: (152, 251, 152),  # terrain
    3: (244, 35, 232),  # sidewalk
    4: (250, 170, 160),  # parking
    5: (128, 64, 128),  # road
    6: (107, 142, 35),  # vc
    7: (107, 142, 35),  # ve
    8: (0, 60, 100),  # vb
    9: (0, 0, 142),  # vs
    10: (119, 11, 32),  # tw
    11: (220, 20, 60),  # h
    12: (70, 70, 70),  # cb
    13: (102, 102, 156),  # cs
    14: (153, 153, 153),  # p
    15: (250, 170, 30),  # tc
    16: (0, 128, 192),  # o
}

def voxel_grid_to_cubes_with_wireframes(
    voxel_grid_data, voxel_size=2.0, voxel_z_offset=0.5
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
                    color = np.array(skitti_colors[label]) / 255
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

def with_extension(feature_file: Path) -> Path:
    return feature_file.with_suffix(".gz")

def load_computed_feature_from_folder(feature_file: Path) -> np.ndarray:
    try:
        with gzip.open(with_extension(feature_file), "rb") as f:
            data = pickle.load(f)
            voxel_label = data["data"]
        return voxel_label
    except Exception as e:
        raise RuntimeError(f"Corrupted or unreadable file: {feature_file}") from e

# Start directory
data_dir = Path("/home/raniatze/Documents/skitti_workspace/cache/pdd_cache/voxel_cache_256")

# Iterate over all .gz files
all_files = list(data_dir.rglob("*.gz"))

print(f"Found {len(all_files)} files.")

for gz_file in tqdm(all_files):
    sequence = gz_file.parts[-4]
    split = gz_file.parts[-3]
    frame = gz_file.parts[-2]
    result = f"{sequence}/{split}/{frame}"
    # print(result)
    try:
        # Try to load file
        with gzip.open(gz_file, "rb") as f:
            data = pickle.load(f)
            voxel_grid = data["data"]
            # print(voxel_grid.shape)
    except Exception as e:
        print(f"[ERROR] File corrupted or failed to load: {gz_file}")
        print(f"        Reason: {e}")

    # combined_cubes, combined_wireframes = voxel_grid_to_cubes_with_wireframes(voxel_grid)
    # normal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    # o3d.visualization.draw_geometries(
        # [combined_cubes, combined_wireframes, normal_frame], window_name=f"{result} Full Grid"
    # )
