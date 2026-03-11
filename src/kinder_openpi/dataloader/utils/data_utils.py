"""
data_utils.py

Additional RLDS-specific data utilities.
"""

from collections.abc import Callable
from copy import deepcopy
from enum import Enum
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


# Note: Both DROID and OXE use roll-pitch-yaw convention (extrinsic XYZ).
# Note: quaternion is in xyzw order.
# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    NONE = -1  # No Proprioceptive State
    POS_EULER = 1  # EEF XYZ (3) + Roll-Pitch-Yaw extrinsic XYZ (3) + Gripper Open/Close (1)  Note: no <PAD>
    POS_QUAT = 2  # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3  # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4  # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    EEF_R6 = 5  # EEF XYZ (3) + R6 (6) + Gripper Open/Close (1)


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    EEF_POS = 1  # EEF Delta XYZ (3) + Roll-Pitch-Yaw extrinsic XYZ (3) + Gripper Open/Close (1).
    JOINT_POS = 2  # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4  # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    ABS_EEF_POS = 5  # EEF Absolute XYZ (3) + Roll-Pitch-Yaw extrinsic XYZ (3) + Gripper Open/Close (1)


class ControlMode(str, Enum):
    """Control mode for action representation.

    - joint_pos: Absolute joint positions (pass-through, no transformation)
    - delta_joint_pos: Delta joint positions computed from absolute positions
    """

    JOINT_POS = "joint_pos"  # Absolute joint positions (no transformation)
    DELTA_JOINT_POS = "delta_joint_pos"  # Delta joint positions from absolute


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


def state_encoding_to_type(encoding: StateEncoding) -> str:
    """Map StateEncoding to human-readable state type string.

    Args:
        encoding: The StateEncoding enum value

    Returns:
        State type string: "none", "joint_pos", or "eef_pose"
    """
    if encoding == StateEncoding.NONE:
        return "none"
    if encoding in (StateEncoding.JOINT, StateEncoding.JOINT_BIMANUAL):
        return "joint_pos"
    if encoding in (StateEncoding.POS_EULER, StateEncoding.POS_QUAT, StateEncoding.EEF_R6):
        return "eef_pose"
    raise ValueError(f"Unknown StateEncoding: {encoding}")


def tree_map(fn: Callable, tree: dict) -> dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_merge(*trees: dict) -> dict:
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    if tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")


def make_decode_images_fn(
    *,
    primary_key: str,
    wrist_key: str | None,
    wrist_right_key: str | None = None,
    resize_to: tuple[int, int] | None = (224, 224),
):
    """Return a frame_map function that decodes encoded image bytes to uint8 tensors.
    Preserves aspect ratio, pads symmetrically, and returns the original dtype semantics
    (uint8 clamped 0-255, float32 clamped to [-1, 1]).
    """

    def _tf_resize_with_pad(image: tf.Tensor, target_h: int, target_w: int) -> tf.Tensor:
        # Compute resized dimensions preserving aspect ratio
        in_h = tf.shape(image)[0]
        in_w = tf.shape(image)[1]
        orig_dtype = image.dtype

        h_f = tf.cast(in_h, tf.float32)
        w_f = tf.cast(in_w, tf.float32)
        th_f = tf.cast(target_h, tf.float32)
        tw_f = tf.cast(target_w, tf.float32)

        ratio = tf.maximum(w_f / tw_f, h_f / th_f)
        resized_h = tf.cast(tf.math.floor(h_f / ratio), tf.int32)
        resized_w = tf.cast(tf.math.floor(w_f / ratio), tf.int32)

        # Resize in float32
        img_f32 = tf.cast(image, tf.float32)
        resized_f32 = tf.image.resize(img_f32, [resized_h, resized_w], method=tf.image.ResizeMethod.BILINEAR)

        # Dtype-specific postprocess (python conditional on static dtype)
        if orig_dtype == tf.uint8:
            resized = tf.cast(tf.clip_by_value(tf.round(resized_f32), 0.0, 255.0), tf.uint8)
            const_val = tf.constant(0, dtype=resized.dtype)
        else:
            resized = tf.clip_by_value(resized_f32, -1.0, 1.0)
            const_val = tf.constant(-1.0, dtype=resized.dtype)

        # Compute symmetric padding
        pad_h_total = target_h - resized_h
        pad_w_total = target_w - resized_w
        pad_h0 = pad_h_total // 2
        pad_h1 = pad_h_total - pad_h0
        pad_w0 = pad_w_total // 2
        pad_w1 = pad_w_total - pad_w0

        padded = tf.pad(resized, [[pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]], constant_values=const_val)
        return padded

    def _decode_single(img_bytes):
        # If already numeric, cast to uint8 and return
        if img_bytes.dtype != tf.string:
            img = tf.cast(img_bytes, tf.uint8)
        else:
            # Guard against empty placeholders (e.g., padding "")
            has_data = tf.greater(tf.strings.length(img_bytes), 0)
            img = tf.cond(
                has_data,
                lambda: tf.io.decode_image(
                    img_bytes,
                    channels=3,
                    expand_animations=False,
                    dtype=tf.uint8,
                ),
                lambda: tf.zeros([1, 1, 3], dtype=tf.uint8),
            )
        # Optional resize-with-pad to ensure batching shape compatibility
        if resize_to is not None:
            h, w = resize_to
            img = _tf_resize_with_pad(img, h, w)
        return img

    def decode_with_time_dim(img_tensor):
        """Decode images that may have time dimension.

        Handles:
        - Rank 0: scalar encoded string (single image)
        - Rank 1: [T] vector of encoded strings (prediction mode with multiple frames)
        - Rank 3: [H, W, C] single decoded image
        - Rank 4: [T, H, W, C] decoded images with time dimension
        """
        rank = len(img_tensor.shape)

        if rank == 1:  # [T] - multiple encoded strings (prediction mode)
            # Decode each encoded string separately
            # Output: [T, H, W, C] after decoding
            decoded_frames = tf.map_fn(_decode_single, img_tensor, fn_output_signature=tf.uint8)
            # Set explicit shape for downstream processing
            if resize_to is not None:
                h, w = resize_to
                decoded_frames.set_shape([None, h, w, 3])
            return decoded_frames
        if rank == 4:  # [T, H, W, C] - already decoded with time dimension
            # Apply resize if needed (shouldn't normally happen in this path)
            return img_tensor
        # rank == 0 (scalar string) or rank == 3 ([H, W, C])
        # Single frame: decode if string, otherwise return as-is
        return _decode_single(img_tensor)

    def _decode_frame(traj: dict) -> dict:
        traj["observation"][primary_key] = decode_with_time_dim(traj["observation"][primary_key])
        traj["observation"][wrist_key] = decode_with_time_dim(traj["observation"][wrist_key])
        traj["observation"][wrist_right_key] = decode_with_time_dim(traj["observation"][wrist_right_key])

        return traj

    return _decode_frame


# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_names: list[str], dataset_weights: list[int]) -> None:
    print("\n######################################################################################")
    print(f"# Loading the following {len(dataset_names)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_name, weight in zip(dataset_names, dataset_weights):
        pad = 80 - len(dataset_name)
        print(f"# {dataset_name}: {weight:=>{pad}f} #")
    print("######################################################################################\n")


def allocate_threads(n: int | None, weights: np.ndarray):
    """
    Allocates an integer number of threads across datasets based on weights.

    The final array sums to `n`, but each element is no less than 1. If `n` is None, then every dataset is assigned a
    value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)

        # Recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1

    return allocation


def load_dataset_kwargs(
    rlds_data_dir: Path,
    load_camera_views: tuple[str] = ("primary", "wrist"),
) -> dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    from kinder_openpi.dataloader.utils.configs import OXE_DATASET_CONFIGS
    from kinder_openpi.dataloader.utils.transforms import OXE_STANDARDIZATION_TRANSFORMS

    ds_name = "planning_dataset"
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[ds_name])
    if dataset_kwargs["action_encoding"] not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6]:
        raise ValueError(f"Cannot load `{ds_name}`; only EEF_POS & EEF_R6 actions supported!")

    language_annotations = dataset_kwargs.get("language_annotations")
    if not language_annotations or language_annotations.lower() == "none":
        raise ValueError(f"Cannot load `{ds_name}`; language annotations required!")

    if (
        dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS
        or dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6
    ):
        pass
    else:
        raise ValueError(f"Cannot load `{ds_name}`; only EEF_POS & EEF_R6 actions supported!")

    # Filter
    dataset_kwargs["image_obs_keys"] = {k: dataset_kwargs["image_obs_keys"].get(k, None) for k in load_camera_views}

    for k, v in dataset_kwargs["image_obs_keys"].items():
        if k == "primary":
            assert v is not None, f"primary image is required for {ds_name}"

    dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS.get(ds_name)

    return {"name": ds_name, "data_dir": str(rlds_data_dir), **dataset_kwargs}
