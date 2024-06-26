import datetime
import os


_save_folder = "sim_results/run-{date:%Y-%m-%d_%H:%M:%S}-intersection".format(
    date=datetime.datetime.now()
)
_sensor_data_logger = {
    "type": "SensorDataLogger",
    "output_folder": os.path.join(_save_folder, "data"),
}
_object_data_logger = {
    "type": "ObjectStateLogger",
    "output_folder": os.path.join(_save_folder, "objects"),
}
_observation_cameras = [
    {
        "type": "CarlaRgbCamera",
        "name": "camera-0",
        "reference": {
            "type": "CarlaReferenceFrame",
            "camera": True,
        },
        "image_size_x": 1280,
        "image_size_y": 1280,
        "post_hooks": [_sensor_data_logger],
    },
]
_infra_sensor_suite = [
    {
        "type": "CarlaRgbCamera",
        "name": "camera-0",
        "reference": {
            "type": "CarlaReferenceFrame",
            "camera": True,
        },
        "post_hooks": [_sensor_data_logger],
    },
    {
        "type": "CarlaLidar",
        "name": "lidar-0",
        "horizontal_fov": 120.0,
        "upper_fov": 30.0,
        "lower_fov": -30.0,
        "channels": 64,
        "points_per_second": 2240000,  # 1750 * 64 * 60 * 120/360
        "reference": {
            "type": "CarlaReferenceFrame",
            "camera": False,
        },
        "post_hooks": [_sensor_data_logger],
    },
]
_mobile_sensor_suite = [
    {
        "type": "CarlaRgbCamera",
        "name": "camera-0",
        "post_hooks": [_sensor_data_logger],
    },
    {
        "type": "CarlaRgbCamera",
        "name": "camera-1",
        "reference": {
            "type": "CarlaReferenceFrame",
            "location": [-1.5, 0, 1.6],
            "rotation": [0, 0, 180],
            "camera": True,
        },
        "post_hooks": [_sensor_data_logger],
    },
    {
        "type": "CarlaLidar",
        "name": "lidar-0",
        "post_hooks": [_sensor_data_logger],
    },
    {
        "type": "CarlaGnss",
        "name": "gnss-0",
        "post_hooks": [_sensor_data_logger],
    },
    {
        "type": "CarlaImu",
        "name": "imu-0",
        "post_hooks": [_sensor_data_logger],
    },
]
_empty_pipeline = {"type": "SerialPipeline", "modules": []}

actor_manager = {
    "type": "CarlaObjectManager",
    "subname": "actors",
    "objects": [
        {
            "type": "CarlaMobileActor",
            "spawn": 1,
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": [-24, 0, 0],
                "camera": False,
            },
            "vehicle": 1,
            "destination": None,
            "autopilot": True,
            "sensors": _mobile_sensor_suite,
            "pipeline": _empty_pipeline,
        },
        {
            "type": "CarlaStaticActor",
            "spawn": 1,
            "sensors": _infra_sensor_suite,
            "pipeline": _empty_pipeline,
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": [0, 0, 20],
                "rotation": [0, 30, 0],
                "camera": False,
            },
        },
        {
            "type": "CarlaStaticActor",
            "spawn": 1,
            "sensors": _infra_sensor_suite,
            "pipeline": _empty_pipeline,
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": [16, -22, 20],
                "rotation": [0, 30, 90],
                "camera": False,
            },
        },
        {
            "type": "CarlaStaticActor",
            "spawn": 1,
            "sensors": _infra_sensor_suite,
            "pipeline": _empty_pipeline,
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": [16, 30, 20],
                "rotation": [0, 30, -90],
                "camera": False,
            },
        },
        {
            "type": "CarlaStaticActor",
            "spawn": 1,
            "sensors": _observation_cameras,
            "pipeline": _empty_pipeline,
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": [10, 5, 70],
                "rotation": [0, 90, 90],
                "camera": False,
            },
        },
        {
            "type": "CarlaStaticActor",
            "spawn": 1,
            "sensors": _observation_cameras,
            "pipeline": _empty_pipeline,
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": [60, 5, 140],
                "rotation": [0, 90, 90],
                "camera": False,
            },
        },
    ],
    "post_hooks": [_object_data_logger],
}

_n_npcs = 50
npc_manager = {
    "type": "CarlaObjectManager",
    "subname": "npcs",
    "objects": [
        # these go with random seed XX
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.volkswagen.t2",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": [-22, 3.6, 0],
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.nissan.patrol_2021",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": [-26, 3.6, 0],
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.mercedes.sprinter",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": [-24, 0, 0],
                "camera": False,
            },
        },
        *[
            {"type": "CarlaNpc", "spawn": "random", "npc_type": "vehicle"}
            for _ in range(_n_npcs)
        ],
    ],
    "post_hooks": [_object_data_logger],
}
