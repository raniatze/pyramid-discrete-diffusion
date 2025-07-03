from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data.dataloader import default_collate

logger = logging.getLogger(__name__)

FeatureDataType = Union[npt.NDArray[np.float32], torch.Tensor]


def to_tensor(data: FeatureDataType) -> torch.Tensor:
    """
    Convert data to tensor
    :param data which is either numpy or Tensor
    :return torch.Tensor
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        raise ValueError(f"Unknown type: {type(data)}")


class AbstractFeature(ABC):

    @classmethod
    def collate(cls, batch: List[AbstractFeature]) -> AbstractFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        serialized = [sample.serialize() for sample in batch]
        return cls.deserialize(default_collate(serialized))

    @abstractmethod
    def to_feature_tensor(self) -> AbstractFeature:
        """
        :return object which will be collated into a batch
        """
        pass

    @abstractmethod
    def to_device(self, device: torch.device) -> AbstractFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        pass

    def serialize(self) -> Dict[str, Any]:
        """
        :return: Return dictionary of data that can be serialized
        """
        return dataclasses.asdict(self)

    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractFeature:
        """
        :return: Return dictionary of data that can be serialized
        """
        pass

    @abstractmethod
    def unpack(self) -> List[AbstractFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        pass

    @property
    def is_valid(self) -> bool:
        """
        :return: Whether the feature is valid (e.g. non empty). By default all features are valid unless overridden.
        """
        return True