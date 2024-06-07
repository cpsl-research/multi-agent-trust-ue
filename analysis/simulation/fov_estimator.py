from typing import Any

from avstack.config import MODELS


@MODELS.register_module()
class ConcaveHullLidarFOVEstimator:
    def __init__(self, concavity: int = 1, length_threshold: float = 4):
        self.concavity = concavity
        self.length_threshold = length_threshold

    def __call__(self, pc, in_global: bool, *args: Any, **kwds: Any) -> Any:
        fov = pc.concave_hull_bev(
            concavity=self.concavity,
            length_threshold=self.length_threshold,
            in_global=in_global,
        )
        return fov
