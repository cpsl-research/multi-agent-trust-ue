from datetime import datetime

import avapi  # noqa # pylint: disable=unused-import; to set the registries
import avstack  # noqa # pylint: disable=unused-import; to set the registries
import simulation  # noqa # pylint: disable=unused-import; to set the registries
from simulation import DatasetReplayer, TrustSimulation
from tqdm import tqdm


def main():
    t_start = datetime.now()
    agents = [
        {"type": "MobileAgent", "ID": 0, "t_start": t_start},
        {"type": "StaticAgent", "ID": 1, "t_start": t_start},
        {"type": "StaticAgent", "ID": 2, "t_start": t_start},
        {"type": "StaticAgent", "ID": 3, "t_start": t_start},
    ]
    command_center = {"type": "CommandCenter", "t_start": t_start}
    replayer = {"type": "DatasetReplayer", "scene_index": 0}
    trust_estimator = {"type": "TrustEstimator"}
    simulator = TrustSimulation(
        t0=t_start,
        agents=agents,
        command_center=command_center,
        trust_estimator=trust_estimator,
    )
    replayer = DatasetReplayer(scene_index=0)

    # run through simulator
    for data_input in tqdm(replayer):
        data_output = simulator(data_input)


if __name__ == "__main__":
    main()
