from pathlib import Path
from prettytable import PrettyTable
import torch
import os
import gzip
import pickle
import numpy as np
import torch.nn.functional as F
from typing import Type
from utils.dicts import clean_dict
from torch.utils.tensorboard import SummaryWriter
from features.voxel_feature import VoxelGrid
from features.abstract_feature import AbstractFeature

def with_extension(feature_file: Path) -> str:
    """Inherited, see superclass."""
    return str(feature_file.with_suffix(".gz"))

def store_computed_feature_to_folder(
    feature_file: Path, feature: AbstractFeature
) -> bool:
    """Inherited, see superclass."""
    serializable_dict = feature.serialize()
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(with_extension(feature_file), "wb", compresslevel=1) as f:
        pickle.dump(serializable_dict, f)
    return True

def load_computed_feature_from_folder(
    feature_file: Path, feature_type: Type[AbstractFeature]
) -> AbstractFeature:
    """Inherited, see superclass."""
    with gzip.open(with_extension(feature_file), "rb") as f:
        data = pickle.load(f)
    return feature_type.deserialize(data)

def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table

def create_folders(mode, log_path):
    if mode == 'train':
        os.makedirs(log_path+'/models', exist_ok=True)
    elif mode in ['inference', 'infinity_gen']:
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(log_path+'/Generated', exist_ok=True)
        os.makedirs(log_path+'/GeneratedFusion', exist_ok=True)
        os.makedirs(log_path+'/PrevSceneContext', exist_ok=True)
        os.makedirs(log_path+'/PrevSceneContextFusion', exist_ok=True)
        os.makedirs(log_path+'/GroundTruth', exist_ok=True)
        os.makedirs(log_path+'/GroundTruthFusion', exist_ok=True)
        os.makedirs(log_path+'/MaskedSceneContext', exist_ok=True)
        os.makedirs(log_path+'/InfiniteScene', exist_ok=True)
        
    print("Storing logs in:", log_path)


def visualization(args, generated, prev_data_voxels, next_data_voxels, iteration, sub_scenes=None, second_context=None):

    if second_context is None:
        loop_iterator = zip(generated, prev_data_voxels, next_data_voxels)
    else:
        loop_iterator = zip(generated, prev_data_voxels, next_data_voxels, second_context)

    for batch, items in enumerate(loop_iterator):
        if second_context is None:
            generated_i, prev_data_i, next_data_i = items
        else:
            generated_i, prev_data_i, next_data_i, masked_context_i = items

        generated_index = []
        next_data_index = []
        prev_data_index = []
        masked_context_index = []

        for i in range(1, args.num_classes):
            index = torch.nonzero(generated_i == i ,as_tuple=False)  # (N, 3)
            generated_index.append(F.pad(index,(1,0),'constant',value = i))  # (N, 4)

            if isinstance(next_data_i, torch.Tensor):
                out_color = torch.nonzero(next_data_i == i, as_tuple=False)  # (N', 3)
                next_data_index.append(F.pad(out_color,(1,0),'constant',value=i))  # (N', 4)
            if args.prev_stage != 'none' and isinstance(prev_data_i, torch.Tensor):
                sub_index = torch.nonzero(prev_data_i == i, as_tuple=False)
                prev_data_index.append(F.pad(sub_index,(1,0),'constant',value = i))

            if second_context is not None and isinstance(masked_context_i, torch.Tensor):
                second_index = torch.nonzero(masked_context_i == i, as_tuple=False)
                masked_context_index.append(F.pad(second_index,(1,0),'constant',value = i))

        generated_indexes = torch.cat(generated_index, dim = 0).cpu().numpy()
        if next_data_index:
            next_data_indexes = torch.cat(next_data_index, dim=0).cpu().numpy()
        else:
            next_data_indexes = np.empty((0, 4))

        if args.prev_stage != 'none':
            prev_data_indexes = torch.cat(prev_data_index, dim = 0).cpu().numpy()
            if second_context is not None:
                masked_context_indexes = torch.cat(masked_context_index, dim = 0).cpu().numpy()

        # generated_voxel_grid = VoxelGrid(data=generated_i.cpu().numpy())
        # conditioning_voxel_grid = VoxelGrid(data=prev_data_i.cpu().numpy())
        # ground_truth_voxel_grid = VoxelGrid(data=next_data_i.cpu().numpy())

        if args.prev_stage != 'none':

            # save_path = Path(args.log_path) / "Voxels" / "Generated"
            # save_path.mkdir(parents=True, exist_ok=True)
            # store_computed_feature_to_folder(save_path / f"result_{(iteration * args.batch_size) + batch}_{sub_scenes}", generated_voxel_grid)

            # save_path = Path(args.log_path) / "Voxels" / "PrevSceneContext"
            # save_path.mkdir(parents=True, exist_ok=True)
            # store_computed_feature_to_folder(save_path / f"prev_{(iteration * args.batch_size) + batch}_{sub_scenes}", conditioning_voxel_grid)

            np.savetxt(args.log_path+'/Generated/result_{}_{}.txt'.format((iteration * args.batch_size) + batch, sub_scenes), generated_indexes)
            np.savetxt(args.log_path+'/PrevSceneContext/prev_{}_{}.txt'.format((iteration * args.batch_size) + batch, sub_scenes), prev_data_indexes)
            if args.infer_data_source == 'dataset' and args.prev_stage != 'none':
                # save_path = Path(args.log_path) / "Voxels" / "GroundTruth"
                # save_path.mkdir(parents=True, exist_ok=True)
                # store_computed_feature_to_folder(save_path / f"gt_{(iteration * args.batch_size) + batch}_{sub_scenes}", ground_truth_voxel_grid)
                np.savetxt(args.log_path+'/GroundTruth/gt_{}_{}.txt'.format((iteration * args.batch_size) + batch, sub_scenes), next_data_indexes)
            if second_context is not None:
                np.savetxt(args.log_path+'/MaskedSceneContext/masked_{}_{}.txt'.format((iteration * args.batch_size) + batch, sub_scenes), masked_context_indexes)
        else:
            # save_path = Path(args.log_path) / "Voxels" / "Generated"
            # save_path.mkdir(parents=True, exist_ok=True)
            # store_computed_feature_to_folder(save_path / f"result_{(iteration * args.batch_size) + batch}", generated_voxel_grid)
            np.savetxt(args.log_path+'/Generated/result_{}.txt'.format((iteration * args.batch_size) + batch), generated_indexes)  # generated_indexes: (total_num_fg_voxels, 4)
            if args.infer_data_source == 'dataset':
                # save_path = Path(args.log_path) / "Voxels" / "GroundTruth"
                # save_path.mkdir(parents=True, exist_ok=True)
                # store_computed_feature_to_folder(save_path / f"gt_{(iteration * args.batch_size) + batch}", ground_truth_voxel_grid)
                np.savetxt(args.log_path+'/GroundTruth/gt_{}.txt'.format((iteration * args.batch_size) + batch), next_data_indexes)  # next_data_indexes: (total_num_fg_voxels', 4)
        

def save_args(args, log_path):
    with open(os.path.join(log_path, 'args.pickle'), "wb") as f:
        pickle.dump(log_path, f)

    args_table = get_args_table(vars(args))
    with open(os.path.join(log_path,'args_table.txt'), "w") as f:
        f.write(str(args_table))

def create_writer(args, log_path, no_log_keys):
    args_dict = clean_dict(vars(args), keys=no_log_keys)
    writer = SummaryWriter(os.path.join(log_path, 'tb'))
    writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)
