from .base import BaseDefense
from .OptimizedWatermarkDefense import OptimizedWatermarkDefense
from .WatermarkDefense import (
    WatermarkByRandomGraph,
    BaseDefense
)


__all__ = [
    'BaseDefense',
    'WatermarkByRandomGraph',
    'OptimizedWatermarkDefense'
]
