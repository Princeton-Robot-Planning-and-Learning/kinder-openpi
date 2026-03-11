"""
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""

from typing import Any


def planning_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Default planning dataset transform using EE_POSE control mode."""
    return trajectory


# === Registry ===
# Default transforms (backward compatible, using EE_POSE control mode)
OXE_STANDARDIZATION_TRANSFORMS = {
    "planning_dataset": planning_dataset_transform,
}
