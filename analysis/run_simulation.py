from datetime import datetime, timedelta

import avapi  # noqa # pylint: disable=unused-import; to set the registries
import avstack  # noqa # pylint: disable=unused-import; to set the registries
import simulation  # noqa # pylint: disable=unused-import; to set the registries
from simulation import DatasetReplayer, TrustSimulation
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from tqdm import tqdm


def object_to_stone_soup_truth(obj, t_start):
    ts = t_start + timedelta(seconds=obj.t)
    xx, xy, xz = obj.position.x
    h, w, l = obj.box.size
    vx, vy, vz = obj.velocity.x
    er, ep, ey = obj.attitude.euler
    state = [xx, vx, xy, vy, xz, vz, h, w, l, er, ep, ey]
    metadata = {
        "object_type": obj.obj_type,
        "object_ID": obj.ID,
        "occlusion": obj.occlusion,
    }
    return GroundTruthState(state, timestamp=ts, metadata=metadata)


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
    objs_in_view = set()
    truth = {"objects_visible": {}, "objects": {}, "agent": {agent: {} for agent in agents}}
    visible_times = {"first": {}, "last": {}}
    for data_input in tqdm(replayer):
        data_output = simulator(data_input)


if __name__ == "__main__":
    main()
