import cProfile
import os

from avapi.carla import CarlaScenesManager
from avstack.datastructs import DataContainer
from avstack.modules.perception.object3d import MMDetObjectDetector3D
from avstack.modules.tracking.multisensor import MeasurementBasedMultiTracker
from avstack.modules.tracking.tracker3d import BasicBoxTracker3D
from tqdm import tqdm


def main():
    cpath = os.path.join("/data/shared/CARLA/multi-agent-intersection/")
    CSM = CarlaScenesManager(cpath)
    idx = 0
    CDM = CSM.get_scene_dataset_by_index(idx)
    vid_folder = f"videos_intersection_{idx}"

    print(CSM.scenes)
    print(f"{len(CDM)} frames")

    # init models
    # agents = list(range(len(CDM.get_agents(frame=1))))
    agents = list(range(4))
    agent_is_static = {
        i: "static" in CDM.get_agent(frame=1, agent=i).obj_type for i in agents
    }
    n_static = sum(list(agent_is_static.values()))
    print(
        "There are {} agents\n   {} mobile, {} static".format(
            len(agents), len(agents) - n_static, n_static
        )
    )
    percep_veh = MMDetObjectDetector3D(
        model="pointpillars", dataset="carla-vehicle", gpu=0
    )
    percep_inf = MMDetObjectDetector3D(
        model="pointpillars", dataset="carla-infrastructure", gpu=0
    )
    percep_col = None
    trackers = {agent: BasicBoxTracker3D() for agent in agents}
    # trackers = {agent: StoneSoupKalmanTracker3DBox() for agent in agents}
    trackers["central"] = MeasurementBasedMultiTracker(tracker=BasicBoxTracker3D())
    trackers["collab"] = BasicBoxTracker3D()

    # init data structures
    ss_tracks = {}
    dets = {}
    tracks = {}
    timestamps_all = {agent: [] for agent in agents}
    imgs_all = {agent: [] for agent in agents}
    pcs_all = {agent: [] for agent in agents}
    dets_all = {agent: [] for agent in agents}
    dets_all["collab"] = []
    tracks_all = {agent: [] for agent in agents}
    tracks_all["central"] = []
    tracks_all["collab"] = []
    agent_0_frames = CDM.get_frames(sensor="lidar-0", agent=0)[1:-1]
    platforms_all = {agent: [] for agent in agents}

    # flags for this run
    run_distributed_perception = True
    run_distributed_tracking = True
    run_centralized_tracking = True
    run_collaborative_perception = False
    run_collaborative_tracking = False

    # run loop
    n_frames_max = 100
    ego_agent = agents[0]
    for frame in tqdm(agent_0_frames[: min(n_frames_max, len(agent_0_frames))]):
        found_data = False
        fovs = {}
        platforms = {}
        perception_input = {}
        for agent in agents:
            ###############################################
            # GET DATA
            ###############################################
            lidar_sensor = "lidar-0"
            camera_sensor = "camera-0"
            img = CDM.get_image(frame=frame, sensor=camera_sensor, agent=agent)
            pc = CDM.get_lidar(frame=frame, sensor=lidar_sensor, agent=agent)
            imgs_all[agent].append(img)
            pcs_all[agent].append(pc)
            objs = CDM.get_objects(frame=frame, sensor=lidar_sensor, agent=agent)
            calib = CDM.get_calibration(frame=frame, sensor=lidar_sensor, agent=agent)
            fovs[agent] = pc.concave_hull_bev(
                concavity=1, length_threshold=4, in_global=False
            )
            # fovs[agent] = Sphere(radius=100)
            platforms[agent] = calib.reference
            platforms_all[agent].append(calib.reference)

            ###############################################
            # DISTRIBUTED PERCEPTION
            ###############################################
            found_data = True
            if run_distributed_perception:
                if agent_is_static[agent]:
                    dets[agent] = percep_inf(pc)
                else:
                    dets[agent] = percep_veh(pc)
                dets_all[agent].append(dets[agent])

            ###############################################
            # DISTRIBUTED TRACKING USING DISTRIBUTED PERCEP
            ###############################################
            if run_distributed_tracking:
                assert run_distributed_perception
                tracks[agent] = trackers[agent](
                    dets[agent], platform=calib.reference, calibration=calib
                )
                if not isinstance(tracks[agent], DataContainer):
                    raise
                tracks_all[agent].append(
                    [track.box3d.copy() for track in tracks[agent]]
                )

        ###############################################
        # COLLABORATIVE PERCEPTION/TRACKING
        ###############################################
        if run_collaborative_perception:
            raise NotImplementedError

        ###############################################
        # CENTRALIZED TRACKING USING DISTRIBUTED PERCEP
        ###############################################
        # run central tracker on all detections
        if found_data:
            if run_centralized_tracking:
                # Run trust model

                # Run tracking
                tracks["central"] = trackers["central"](
                    detections=dets,
                    fovs=fovs,
                    platforms=platforms,
                )
                tracks_all["central"].append(
                    [track.box3d.copy() for track in tracks["central"]]
                )


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        pr.disable()
        pr.dump_stats("last_run.prof")
