import re
import os
import glob
import sys
import logging
import random
import argparse
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("select_common_ids")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ID_RE = re.compile(r"bev_semantic_map_(.+)\.gz$")  # handles numeric or hashed IDs

def choose_numeric_ids(total: int, k: int, seed: int) -> list[str]:
    if k > total:
        raise ValueError(f"Requested {k} but total is {total}.")
    rnd = random.Random(seed)
    return [str(i) for i in rnd.sample(range(total), k)]

def write_manifest(ids: List[str], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Manifest already exists: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in ids:
            f.write(f"{i}\n")

def infer_num_samples(mode):
    if mode == "fps_1000":
        num_samples = 1000
    elif mode == "fps_5000":
        num_samples = 5000
    elif mode == "fps_10000":
        num_samples = 10000
    elif mode == "distance_1":
        num_samples = 32203
    elif mode == "distance_2":
        num_samples = 17487
    elif mode == "distance_5":
        num_samples = 7447
    elif mode == "distance_10":
        num_samples = 3855
    else:
        raise ValueError(f"Unknown mode {mode}")
    return num_samples

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select common generated sample IDs across stages.")
    p.add_argument("--total", type=int, default=50000, help="Total number of generated samples per stage (IDs in [0, total-1]).")
    p.add_argument("--manifest", type=Path, default="/media/raniatze/Elements/PhD/Research/pyramid-discrete-diffusion/generated", help="Output path for the selected ID list (one ID per line).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for deterministic sampling.")
    p.add_argument("--mode", type=str, default="distance_10")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    num_samples = infer_num_samples(args.mode)

    # numeric fast path
    selected = choose_numeric_ids(args.total, num_samples, args.seed)
    logger.info(f"Selected {len(selected)} numeric IDs out of total={args.total}")

    save_dir = Path(args.manifest) / f"selected_ids_{args.mode}.txt"
    write_manifest(selected, save_dir)
    logger.info(f"Wrote {len(selected)} IDs to: {save_dir}")

if __name__ == "__main__":
    main()