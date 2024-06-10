import cProfile
import json
import os
from argparse import ArgumentParser
from datetime import datetime

import avapi  # noqa # pylint: disable=unused-import; to set the registries
import avstack  # noqa # pylint: disable=unused-import; to set the registries
import simulation  # noqa # pylint: disable=unused-import; to set the registries
from simulation import DatasetReplayer, TrustSimulation
from tqdm import tqdm


def main(args):
    t_start = datetime.now()
    agents = [
        {"type": "MobileAgent", "ID": 0, "t_start": t_start, "log_dir": args.log_dir},
        {"type": "StaticAgent", "ID": 1, "t_start": t_start, "log_dir": args.log_dir},
        {"type": "StaticAgent", "ID": 2, "t_start": t_start, "log_dir": args.log_dir},
        {"type": "StaticAgent", "ID": 3, "t_start": t_start, "log_dir": args.log_dir},
    ]
    command_center = {"type": "CommandCenter", "t_start": t_start}
    trust_estimator = {"type": "TrustEstimator"}
    simulator = TrustSimulation(
        t0=t_start,
        agents=agents,
        command_center=command_center,
        trust_estimator=trust_estimator,
        log_dir=args.log_dir,
    )
    replayer = DatasetReplayer(scene_index=args.scene_index)

    all_results = []
    try:
        # run through simulator
        for data_input in tqdm(replayer(load_perception=True)):
            data_output = simulator(data_input)
            all_results.append(
                {
                    "frame": data_input["frame"],
                    "data": data_output,
                }
            )
    except KeyboardInterrupt:
        pass
    finally:
        # save metadata
        metadata = {"t_start": t_start.timestamp(), "replayer": replayer.metadata}
        with open(os.path.join(args.log_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # save all results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene", dest="scene_index", type=int, default=0)
    parser.add_argument("--log", dest="log_dir", type=str, default="last_run")
    args = parser.parse_args()

    pr = cProfile.Profile()
    pr.enable()
    main(args)
    pr.disable()
    pr.dump_stats(os.path.join(args.log_dir, "last_run.prof"))
