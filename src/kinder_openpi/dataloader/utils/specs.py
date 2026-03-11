"""Data specifications and constants for RLDS datasets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RldsDatasetSpec:
    primary_image_key: str = "base_0_rgb"
    wrist_image_key: str = "left_wrist_0_rgb"
    wrist_image_right_key: str = "right_wrist_0_rgb"
