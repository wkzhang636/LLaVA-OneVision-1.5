""" PackedCaptioningSample """

from dataclasses import dataclass, field
from typing import List, Optional
from megatron.energon.flavors.base_dataset import Sample
import torch
import numpy as np

@dataclass
class PackedCaptioningSample(Sample):
    """Sample type for packed captioning."""
    # sample_id: str
    images: List[torch.Tensor]
    prompts: Optional[List[str]]
    captions: List[str]
    patch_positions: Optional[List[List[np.ndarray]]] = None  # [sample_idx][img_idx] -> np.ndarray