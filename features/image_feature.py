from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import torch

from features.abstract_feature import AbstractFeature

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
class Image(AbstractFeature):
    """Feature class of latent variable."""

    data: FeatureDataType

    def to_device(self, device: torch.device) -> Image:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return Image(data=self.data.to(device=device))

    def to_feature_tensor(self) -> Image:
        """Inherited, see superclass."""
        return Image(data=to_tensor(self.data))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Image:
        """Implemented. See interface."""
        return Image(data=data["data"])

    def unpack(self) -> List[Image]:
        """Implemented. See interface."""
        return [Image(data) for data in self.data]
