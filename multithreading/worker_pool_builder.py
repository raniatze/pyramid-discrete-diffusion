import logging
from hydra._internal.utils import _locate
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Union, Any, Type, Callable

from multithreading.worker_pool import WorkerPool
from multithreading.worker_ray import RayDistributed


def validate_type(instantiated_class: Any, desired_type: Type[Any]) -> None:
    """
    Validate that constructed type is indeed the desired one
    :param instantiated_class: class that was created
    :param desired_type: type that the created class should have
    """
    assert isinstance(
        instantiated_class, desired_type
    ), f"Class to be of type {desired_type}, but is {type(instantiated_class)}!"


def is_target_type(cfg: DictConfig, target_type: Union[Type[Any], Callable[..., Any]]) -> bool:
    """
    Check whether the config's resolved type matches the target type or callable.
    :param cfg: config
    :param target_type: Type or callable to check against.
    :return: Whether cfg._target_ matches the target_type.
    """
    return bool(_locate(cfg._target_) == target_type)

logger = logging.getLogger(__name__)


def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    Builds the worker.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of WorkerPool.
    """
    logger.info("Building WorkerPool...")
    worker: WorkerPool = (
        instantiate(cfg.worker, output_dir=cfg.output_dir)
        if is_target_type(cfg.worker, RayDistributed)
        else instantiate(cfg.worker)
    )
    validate_type(worker, WorkerPool)

    logger.info("Building WorkerPool...DONE!")
    return worker
