from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime

from avstack.config import AGENTS, MODELS, ConfigDict


class Agent:
    def __init__(
        self,
        ID: int,
        t_start: "datetime",
        perception: ConfigDict,
        tracking: ConfigDict,
    ):
        self.ID = ID
        self.t_start = t_start
        self.perception = MODELS.build(perception)
        self.tracking = MODELS.build(tracking, default_args={"t0": t_start})


@AGENTS.register_module()
class MobileAgent(Agent):
    def __init__(
        self,
        ID: int,
        t_start: "datetime",
        perception: ConfigDict = {
            "type": "MMDetObjectDetector3D",
            "model": "pointpillars",
            "dataset": "carla-vehicle",
            "gpu": 0,
            "thresh_duplicate": 1.0,
        },
        tracking: ConfigDict = {
            "type": "StoneSoupKalmanTracker3DBox",
        },
    ):
        super().__init__(ID, t_start, perception, tracking)


@AGENTS.register_module()
class StaticAgent(Agent):
    def __init__(
        self,
        ID: int,
        t_start: "datetime",
        perception: ConfigDict = {
            "type": "MMDetObjectDetector3D",
            "model": "pointpillars",
            "dataset": "carla-infrastructure",
            "gpu": 0,
            "thresh_duplicate": 1.0,
        },
        tracking: ConfigDict = {
            "type": "StoneSoupKalmanTracker3DBox",
        },
    ):
        super().__init__(ID, t_start, perception, tracking)


@AGENTS.register_module()
class CommandCenter:
    def __init__(
        self,
        t_start: "datetime",
        tracking: ConfigDict = {
            "type": "MeasurementBasedMultiTracker",
            "tracker": {"type": "StoneSoupKalmanTracker3DBox"},
        },
    ):
        tracking["tracker"]["t0"] = t_start
        self.tracking = MODELS.build(tracking)
