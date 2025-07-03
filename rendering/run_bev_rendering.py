import logging

import hydra
from omegaconf import DictConfig

from rendering.bev_rendering import run_semantic_map_rendering
from multithreading.worker_pool_builder import build_worker

# logging.getLogger("numba").setLevel(logging.WARNING)  # turns off Numba DEBUG messages?
logger = logging.getLogger(__name__)


CONFIG_PATH = "."
CONFIG_NAME = "default_rendering"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:

    # Build worker
    worker = build_worker(cfg)

    run_semantic_map_rendering(cfg=cfg, worker=worker)


if __name__ == "__main__":
    main()
