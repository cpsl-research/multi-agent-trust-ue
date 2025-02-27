import numpy as np


_base_ = ["./base_intersect.py"]

_n_npcs = 0  # only the npcs we specify below

spawn1 = np.array([-65, 27, 0.60])
ego = np.array([-87, 27, 0.60])
spawn1_to_ego = ego - spawn1

forward = np.array([1, 0, 0])
left = np.array([0, 1, 0])

npc_manager = {
    "type": "CarlaObjectManager",
    "subname": "npcs",
    "objects": [
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.nissan.patrol_2021",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 18 * forward,
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.mercedes.coupe",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 9 * forward,
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.ford.mustang",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 40 * forward + 30 * left,
                "rotation": [0, 0, -90],
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.ford.crown",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 40 * forward + 25 * left,
                "rotation": [0, 0, -90],
                "camera": False,
            },
        },
        *[
            {"type": "CarlaNpc", "spawn": "random", "npc_type": "vehicle"}
            for _ in range(_n_npcs)
        ],
    ],
}
