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
            "npc_type": "vehicle.jeep.wrangler_rubicon",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 4 * forward + 3 * left,  # [-16, 3.6, 0],
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.nissan.patrol",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 5 * forward,  # [-18, 0, 0],
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.mercedes.coupe",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 43 * forward - 22 * left,  # [25, -22, 0],
                "rotation": [0, 0, 90],
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.tesla.model3",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 36 * forward + 45 * left,  # [18, 45, 0],
                "rotation": [0, 0, -90],
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.bmw.grandtourer",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 36 * forward + 35 * left,  # [18, 35, 0],
                "rotation": [0, 0, -90],
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.dodge.charger_2020",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 70 * forward + 14 * left,
                "rotation": [0, 0, -180],
                "camera": False,
            },
        },
        {
            "type": "CarlaNpc",
            "spawn": 1,
            "npc_type": "vehicle.ford.mustang",
            "reference_to_spawn": {
                "type": "CarlaReferenceFrame",
                "location": spawn1_to_ego + 90 * forward + 14 * left,
                "rotation": [0, 0, -180],
                "camera": False,
            },
        },
        *[
            {"type": "CarlaNpc", "spawn": "random", "npc_type": "vehicle"}
            for _ in range(_n_npcs)
        ],
    ],
}