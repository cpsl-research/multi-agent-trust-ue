import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime

from avstack.config import AGENTS, MODELS, ConfigDict
from avstack.geometry import GlobalOrigin3D


class Agent:
    def __init__(
        self,
        ID: int,
        t_start: "datetime",
        fov_estimator: ConfigDict,
        perception: ConfigDict,
        tracking: ConfigDict,
        log_dir: str,
    ):
        self.ID = ID
        self.t_start = t_start

        # add logging hooks
        out_folder = os.path.join(log_dir, f"agent-{ID}", "{}")
        perception["post_hooks"] = [
            {
                "type": "DetectionsLogger",
                "output_folder": out_folder.format("detections"),
            }
        ]
        tracking["post_hooks"] = [
            {
                "type": "StoneSoupTracksLogger",
                "output_folder": out_folder.format("tracks"),
            }
        ]

        # build models
        self.fov_estimator = MODELS.build(fov_estimator)
        self.perception = MODELS.build(perception)
        self.tracking = MODELS.build(tracking, default_args={"t0": t_start})

        # presets
        self.reference = None
        self.fov = None
        self.detections = []
        self.tracks = []

    def pipeline(self, sensor_data, platform, calibration):
        self.reference = platform
        self.fov = self.fov_estimator(sensor_data, in_global=False)
        self.detections = self.perception(sensor_data)
        self.tracks = self.tracking(
            self.detections, platform=platform, calibration=calibration
        )

    def get_detections_global(self):
        dets_global = self.detections.apply_and_return(
            "change_reference", GlobalOrigin3D, inplace=False
        )
        return dets_global

    def get_fov_global(self):
        if self.fov is not None:
            fov_global = self.fov.change_reference(GlobalOrigin3D, inplace=False)
        else:
            raise RuntimeError("FOV is not properly set yet")
        return fov_global

    def get_tracks_global(self):
        boxes = self.tracks.apply_and_return("getattr", "box3d")
        tracks_global = boxes.apply_and_return(
            "change_reference", GlobalOrigin3D, inplace=False
        )
        return tracks_global


@AGENTS.register_module()
class MobileAgent(Agent):
    def __init__(
        self,
        ID: int,
        t_start: "datetime",
        fov_estimator: ConfigDict = {"type": "ConcaveHullLidarFOVEstimator"},
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
        log_dir: str = "last_run",
    ):
        super().__init__(ID, t_start, fov_estimator, perception, tracking, log_dir)


@AGENTS.register_module()
class StaticAgent(Agent):
    def __init__(
        self,
        ID: int,
        t_start: "datetime",
        fov_estimator: ConfigDict = {"type": "ConcaveHullLidarFOVEstimator"},
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
        log_dir: str = "last_run",
    ):
        super().__init__(ID, t_start, fov_estimator, perception, tracking, log_dir)


@AGENTS.register_module()
class CommandCenter:
    def __init__(
        self,
        t_start: "datetime",
        tracking: ConfigDict = {
            "type": "MeasurementBasedMultiTracker",
            "tracker": {"type": "StoneSoupKalmanTracker3DBox"},
        },
        log_dir: str = "last_run",
    ):
        # add logging hooks
        out_folder = os.path.join(log_dir, f"command-center", "{}")
        tracking["post_hooks"] = [
            {
                "type": "StoneSoupTracksLogger",
                "output_folder": out_folder.format("tracks"),
            }
        ]

        # build models
        tracking["tracker"]["t0"] = t_start
        self.tracking = MODELS.build(tracking)

    def pipeline(self, agent_dets, agent_fovs, agent_platforms):
        self.tracks = self.tracking(
            detections=agent_dets, fovs=agent_fovs, platforms=agent_platforms
        )

    def predict_tracks(self, timestamp: "datetime"):
        tracks_predicted = self.tracking.predict_tracks(
            timestamp, platform=GlobalOrigin3D, check_reference=False
        )
        return tracks_predicted
