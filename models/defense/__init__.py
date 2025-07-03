from .base import BaseDefense
from .SurviveWM2 import OptimizedWatermarkDefense
from .WatermarkDefense import (
    WatermarkByRandomGraph,
)
from .ImperceptibleWM import OwnerWatermarkingDefense


__all__ = [
    'BaseDefense',
    'WatermarkByRandomGraph',
    'OptimizedWatermarkDefense'
]
