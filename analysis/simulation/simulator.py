from datetime import datetime
from typing import List

from avstack.config import AGENTS, ConfigDict
from mate.config import MATE


class TrustSimulation:
    def __init__(
        self,
        t0: datetime,
        agents: List[ConfigDict],
        command_center: ConfigDict,
        trust_estimator: ConfigDict,
        video_folder: str = "last_videos",
    ):
        self.t0 = t0
        self.agents = {agent["ID"]: AGENTS.build(agent) for agent in agents}
        self.command_center = AGENTS.build(command_center)
        self.trust_estimator = MATE.build(trust_estimator)
        self.video_folder = video_folder

    def __call__(self, data):
        """Step the simulation forward in time"""

        # run local agent pipelines
        agent_poses = {}
        agent_fovs_global = {}
        agent_dets_global = {}
        agent_tracks_global = {}
        for agent_ID in self.agents:
            # get sensor data
            # HACK: fix that it is lidar data replayed
            sensor_data = data["agent_data"][agent_ID]["sensor_data"]["lidar"]
            calibration = sensor_data.calibration
            platform = calibration.reference

            # run agent pipeline
            self.agents[agent_ID].pipeline(
                sensor_data=sensor_data, platform=platform, calibration=calibration
            )

            # store necessary data
            agent_poses[agent_ID] = platform
            agent_fovs_global[agent_ID] = self.agents[agent_ID].get_fov_global()
            agent_dets_global[agent_ID] = self.agents[agent_ID].get_detections_global()
            agent_tracks_global[agent_ID] = self.agents[agent_ID].get_tracks_global()

        # predict command center tracks
        timestamp = self.t0 + data["timestamp_dt"]
        tracks_cc = self.command_center.predict_tracks(timestamp=timestamp)

        # run trust estimation
        self.trust_estimator(
            agent_poses=agent_poses,
            agent_fovs=agent_fovs_global,
            agent_dets=agent_dets_global,
            agent_tracks=agent_tracks_global,
            cc_tracks=tracks_cc,
        )

        # run command center pipelines in global
        self.command_center.pipeline(
            agent_dets=agent_dets_global,
            agent_fovs=agent_fovs_global,
            agent_platforms=agent_poses,
        )
