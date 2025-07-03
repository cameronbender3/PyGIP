from .base import BaseDefense
from .OptimizedWatermarkDefense import OptimizedWatermarkDefense
from .WatermarkDefense import (
    WatermarkByRandomGraph,
)
from .OwnerWatermarkingDefense import OwnerWatermarkingDefense


__all__ = [
    'BaseDefense',
    'WatermarkByRandomGraph',
    'OptimizedWatermarkDefense'
]
