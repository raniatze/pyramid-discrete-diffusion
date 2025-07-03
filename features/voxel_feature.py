from __future__ import annotations
from enum import Enum
import numpy as np
import torchvision

from typing import Any, Dict, List, Type, Union
import numpy.typing as npt
from dataclasses import dataclass
import torch

from features.abstract_feature import AbstractFeature


FeatureDataType = Union[npt.NDArray[np.float32], torch.Tensor]

def validate_type(instantiated_class: Any, desired_type: Type[Any]) -> None:
    """
    Validate that constructed type is indeed the desired one
    :param instantiated_class: class that was created
    :param desired_type: type that the created class should have
    """
    assert isinstance(
        instantiated_class, desired_type
    ), f"Class to be of type {desired_type}, but is {type(instantiated_class)}!"



@dataclass
class VoxelGrid(AbstractFeature):
    """Feature class of latent variable."""

    data: FeatureDataType

    def to_device(self, device: torch.device) -> VoxelGrid:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return VoxelGrid(data=self.data.to(device=device))

    def to_feature_tensor(self) -> VoxelGrid:
        """Implemented. See interface."""
        to_tensor_torchvision = torchvision.transforms.ToTensor()
        data = to_tensor_torchvision(np.asarray(self.data))
        return VoxelGrid(data=data)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VoxelGrid:
        """Implemented. See interface."""
        return VoxelGrid(data=data["data"])

    def unpack(self) -> List[VoxelGrid]:
        """Implemented. See interface."""
        return [VoxelGrid(data[None]) for data in self.data]

class VoxelGridClass(Enum):
    """Enum of semantic voxel grid classes."""

    EMPTY = 0
    GROUND = 1
    TERRAIN = 2
    SIDEWALK = 3
    PARKING = 4
    ROAD = 5
    VEGETATION_CUBOID = 6
    VEGETATION_ELLIPSOID = 7
    VEHICLE_BIG = 8
    VEHICLE_SMALL = 9
    TWO_WHEELERS = 10
    HUMAN = 11
    CONSTRUCTION_BIG = 12
    CONSTRUCTION_SMALL = 13
    POLE = 14
    TRAFFIC_CONTROL = 15
    OBJECT = 16