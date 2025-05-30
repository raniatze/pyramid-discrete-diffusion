import os
import gzip
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from utils.spilt_scenes import split_scenes
from utils.mask_scene import mask_scene


class PrittiDataset(Dataset):
    def __init__(
        self,
        directory="/home/raniatze/Documents/skitti_workspace/cache/pdd_cache/voxel_cache_32",
        quantized_directory=None,
        split="train",
        data_argumentation=False,
        num_frames=1,
        mode="train",
        prev_stage="none",
        next_stage="s_1",
        prev_data_size=(32, 32, 4),
        next_data_size=(128, 128, 16),
        prev_scene_path=None,
        infer_data_source="dataset",
        mask_ratio=0.0625,
        mask_prob=[0.25, 0.25, 0.25, 0.25],
        model_type="con",
    ):
        self._directory = directory
        self.quantized_directory = quantized_directory
        self._num_frames = num_frames
        self.data_argumentation = data_argumentation
        self.infer_data_source = infer_data_source
        self.prev_data_size = prev_data_size
        self.next_data_size = next_data_size
        self.mode = mode
        self.prev_stage = prev_stage
        self.next_stage = next_stage
        self.prev_scene_path = prev_scene_path
        self.mask_ratio = mask_ratio
        self.mask_prob = mask_prob
        self.model_type = model_type
        self._scenes = sorted(
            [os.path.join(sequence, split) for sequence in os.listdir(self._directory)]
        )

        self._num_scenes = len(self._scenes)
        self._num_frames_scene = []
        self._eval_labels = []
        self._frames_list = []
        self._completion_list = []
        self._quantized_eval_labels = []

        for scene in self._scenes:
            eval_dir = os.path.join(self._directory, scene)
            quantized_dir = os.path.join(self.quantized_directory, scene)
            frames_list = [filename for filename in sorted(os.listdir(eval_dir))]

            self._num_frames_scene.append(len(os.listdir(eval_dir)))
            self._frames_list.extend(frames_list)

            self._eval_labels.extend(
                [Path(eval_dir) / str(frame) / "voxel_grid.gz" for frame in frames_list]
            )
            self._quantized_eval_labels.extend(
                [
                    Path(quantized_dir) / str(frame) / "voxel_grid.gz"
                    for frame in frames_list
                ]
            )

        assert (
            len(self._eval_labels)
            == len(self._quantized_eval_labels)
            == len(self._frames_list)
            == sum(self._num_frames_scene)
        )

        if (
            self.mode in ["inference", "infinity_gen"]
            and self.prev_stage != "none"
            and self.infer_data_source == "generation"
        ):
            for root_, _, files in os.walk(self.prev_scene_path):
                for file in files:
                    if file.endswith(".txt"):
                        self._completion_list.append(os.path.join(root_, file))
            self._completion_list.sort(
                key=lambda f: int(f.split("_")[-1].split(".")[0])
            )

        self.check_data_dimensions()

    def check_data_dimensions(self):
        expected_quantized_dims = {"s_1": (32, 32, 4), "s_2": (64, 64, 8)}
        expected_eval_dims = {
            "s_1": (32, 32, 4),
            "s_2": (64, 64, 8),
            "s_3": (256, 256, 16),
        }

        if self.prev_stage in expected_quantized_dims:
            sample_quantized = self.load_computed_feature_from_folder(
                self._quantized_eval_labels[0]
            )
            if sample_quantized.shape != expected_quantized_dims[self.prev_stage]:
                raise ValueError(
                    f"Incorrect dimensions for _quantized_eval_labels. Expected {expected_quantized_dims[self.prev_stage]}, got {sample_quantized.shape}"
                )

        if self.next_stage in expected_eval_dims:
            sample_eval = self.load_computed_feature_from_folder(self._eval_labels[0])
            if sample_eval.shape != expected_eval_dims[self.next_stage]:
                raise ValueError(
                    f"Incorrect dimensions for _eval_labels. Expected {expected_eval_dims[self.next_stage]}, got {sample_eval.shape}"
                )

    def __len__(self):
        return (
            len(self._completion_list)
            if self.mode in ["inference", "infinity_gen"]
            and self.prev_stage != "none"
            and self.infer_data_source == "generation"
            else sum(self._num_frames_scene)
        )

    def collate_fn(self, data):
        voxel_batch = [bi[0] for bi in data]
        output_batch = [bi[1] for bi in data]
        counts_batch = [bi[2] for bi in data]
        return voxel_batch, output_batch, counts_batch

    def with_extension(self, feature_file: Path) -> str:
        return str(feature_file.with_suffix(".gz"))

    def load_computed_feature_from_folder(self, feature_file: Path) -> np.ndarray:
        try:
            with gzip.open(self.with_extension(feature_file), "rb") as f:
                data = pickle.load(f)
                voxel_label = data["data"]
            return voxel_label
        except EOFError as e:
            raise RuntimeError(
                f"Corrupted or truncated .gz file: {feature_file}"
            ) from e

    def __getitem__(self, idx):
        idx_range = self.find_horizon(idx)
        if idx != idx_range[-1]:
            raise Exception("Found different idx_range!")

        if (
            self.next_stage == "s_1"
            and self.prev_stage == "none"
            and self.infer_data_source == "dataset"
        ):
            next_stage_data = self.load_computed_feature_from_folder(
                self._quantized_eval_labels[idx_range[-1]]
            )
            prev_stage_data = next_stage_data.copy()
            quantized_output = np.zeros_like(next_stage_data)
            counts = np.zeros_like(next_stage_data)
        else:
            quantized_output = self.load_computed_feature_from_folder(
                self._quantized_eval_labels[idx_range[-1]]
            )
            next_stage_data = self.load_computed_feature_from_folder(
                self._eval_labels[idx_range[-1]]
            )
            counts = next_stage_data.copy()
            prev_stage_data = next_stage_data.copy()

        if self.data_argumentation:
            next_stage_data, counts, quantized_output = self.apply_data_augmentation(
                next_stage_data, counts, quantized_output
            )

        if (
            self.mode in ["inference", "infinity_gen"]
            and self.prev_stage != "none"
            and self.infer_data_source == "generation"
        ):
            _output = self.load_generated_output(idx_range[-1])

        if self.next_stage in ["s_2", "s_3"] and self.prev_stage != "none":
            if self.infer_data_source == "dataset":
                low_resolution = quantized_output
            elif self.infer_data_source == "generation":
                low_resolution = _output

            sliding_split = self.mode == "inference" and self.next_stage == "s_3"
            next_stage_data, prev_stage_data, counts = self.split_scenes(
                next_stage_data, low_resolution, counts, sliding=sliding_split
            )  # (num_sub_scenes, next_shape[0], next_shape[1], next_shape[2]), (num_sub_scenes, prev_shape[0], prev_shape[1], prev_shape[2]), (num_sub_scenes, next_shape[0], next_shape[1], next_shape[2])

        if (
            self.mode == "train"
            and self.next_stage == "s_3"
            and self.model_type == "con"
        ):
            next_stage_data = self.apply_mask(next_stage_data)  # (2 * num_sub_scenes, 128, 128, 16) where num_sub_scenes is 4

        return prev_stage_data, next_stage_data, counts

    def apply_data_augmentation(self, next_stage_data, counts, quantized_output):
        if np.random.randint(2):
            next_stage_data = np.flip(next_stage_data, axis=0)
            counts = np.flip(counts, axis=0)
            quantized_output = np.flip(quantized_output, axis=0)

        if np.random.randint(2):
            next_stage_data = np.flip(next_stage_data, axis=1)
            counts = np.flip(counts, axis=1)
            quantized_output = np.flip(quantized_output, axis=1)

        rotation_type = np.random.randint(4)
        if rotation_type > 0:
            k = rotation_type
            next_stage_data = np.rot90(next_stage_data, k, axes=(0, 1))
            counts = np.rot90(counts, k, axes=(0, 1))
            quantized_output = np.rot90(quantized_output, k, axes=(0, 1))

        return next_stage_data, counts, quantized_output

    def load_generated_output(self, idx):
        _output = np.zeros(self.prev_data_size)
        with open(self._completion_list[idx], "r") as file:
            for line in file:
                label, x, y, z = map(int, map(float, line.strip().split()))
                _output[x, y, z] = label
        return _output.astype(np.uint8)

    def split_scenes(self, next_stage_data, low_resolution, counts, sliding=False):
        sub_scenes_high, sub_scenes_low, sub_scenes_high_counts = split_scenes(
            high_resolution=next_stage_data,
            low_resolution=low_resolution,
            high_resolution_counts=counts,
            target_resolution_high=self.next_data_size,
            sliding_split=sliding,
            sliding_ratio=1 - self.mask_ratio,
        )
        return sub_scenes_high, sub_scenes_low, sub_scenes_high_counts

    def apply_mask(self, next_stage_data):
        masked_scene = mask_scene(
            input=next_stage_data, mask_ratio=self.mask_ratio, mask_prob=self.mask_prob
        )  # list with len equal to num_sub_scenes (i.e., next_stage_data.shape[0])
        return np.concatenate((next_stage_data, masked_scene))

    # no enough frame
    def find_horizon(self, idx):
        end_idx = idx
        idx_range = np.arange(idx - self._num_frames, idx) + 1
        diffs = np.asarray(
            [
                int(self._frames_list[end_idx]) - int(self._frames_list[i])
                for i in idx_range
            ]
        )
        good_difs = -1 * (np.arange(-self._num_frames, 0) + 1)

        idx_range[good_difs != diffs] = -1

        return idx_range
