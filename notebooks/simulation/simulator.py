from typing import List

from avstack.config import AGENTS, DATASETS, ConfigDict

from .replayer import DatasetReplayer


class TrustSimulation:
    def __init__(
        self,
        agents: List[ConfigDict],
        command_center: ConfigDict,
        dataset_replayer: ConfigDict,
        video_folder: str = "last_videos",
    ):
        self.agents = [AGENTS.build(agent) for agent in agents]
        self.command_center = AGENTS.build(command_center)
        self.dataset_replayer = (
            dataset_replayer
            if isinstance(dataset_replayer, DatasetReplayer)
            else DATASETS.build(dataset_replayer)
        )
        self.video_folder = video_folder
