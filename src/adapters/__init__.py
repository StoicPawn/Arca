"""Adapter modules for different input modalities."""

from .audio import AudioAdapter, AudioAdapterConfig
from .vision import VisionAdapter, VisionAdapterConfig

__all__ = [
    "AudioAdapter",
    "AudioAdapterConfig",
    "VisionAdapter",
    "VisionAdapterConfig",
]
