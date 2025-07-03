import os
import glob
import argparse
import matplotlib.pyplot as plt

from pathlib import Path
from features.image_feature import Image
from utils.tables import load_computed_feature_from_folder

def run_visualization(args):

    target_paths = glob.glob(os.path.join(args.target_path, "*.gz"))

    for idx, target_path in enumerate(target_paths):

        semantic_map = load_computed_feature_from_folder(
            Path(target_path), Image
        )

        plt.imshow(semantic_map.data)
        plt.show()

    return


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.target_path = Path("/home/raniatze/Documents/PhD/Research/pyramid-discrete-diffusion/generated/s_1_to_s_2/Voxels/Generated/Rendering")
    run_visualization(args)


if __name__ == "__main__":
    main()
