"""
configs.py

Defines per-dataset configuration (kwargs) for each dataset in Open-X Embodiment.

Configuration adopts the following structure:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB

    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth

    # Always 8-dim =>> changes based on `StateEncoding`
    state_obs_keys:
        StateEncoding.POS_EULER:    EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
        StateEncoding.POS_QUAT:     EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
        StateEncoding.JOINT:        Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)

    state_encoding: Type of `StateEncoding`
    action_encoding: Type of action encoding (e.g., EEF Position vs. Joint Position)
"""

from kinder_openpi.dataloader.utils.data_utils import ActionEncoding
from kinder_openpi.dataloader.utils.data_utils import StateEncoding

# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    "planning_dataset": {
        "image_obs_keys": {
            "primary": "base_image",
            "wrist_right": "overview_image",
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
}


OXE_DATASET_METADATA = {
    "planning_dataset": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
}

for dataset_name, metadata in OXE_DATASET_METADATA.items():
    if dataset_name in OXE_DATASET_CONFIGS:
        OXE_DATASET_CONFIGS[dataset_name].update(metadata)
