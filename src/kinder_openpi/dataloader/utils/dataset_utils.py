"""Generic dataset utilities for RLDS datasets."""

import logging
import os

import psutil
import tensorflow as tf

from kinder_openpi.dataloader.utils.data_utils import make_decode_images_fn


def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # in MB
    logging.info(f"[{label}] Memory usage: {mem:.2f} MB")


def gather_with_padding(
    data: tf.Tensor,
    sequence_length: tf.Tensor,
    window_size: int | tf.Tensor,
    per_timestep_windows: tf.Tensor | None = None,
    pad_with_last: bool = True,
) -> tf.Tensor:
    """Gather sliding windows with proper padding.

    Args:
        data: Source tensor to gather from, shape [T, ...] where T is sequence length
        sequence_length: Scalar tensor, length of the sequence
        window_size: Scalar or tensor, size of the window to gather. If per_timestep_windows
                    is provided, this should be the maximum window size.
        per_timestep_windows: Optional [T] tensor specifying variable window size per timestep.
                             If None, uses fixed window_size for all timesteps.
        pad_with_last: If True, pad with the last valid value in the sequence.
                      If False, pad with zeros.

    Returns:
        Gathered windows with shape [T, window_size, ...], properly padded.
    """
    # Create base indices [T, window_size]
    if isinstance(window_size, int):
        window_size_tensor = tf.constant(window_size, dtype=tf.int32)
    else:
        window_size_tensor = tf.cast(window_size, tf.int32)

    base = tf.broadcast_to(tf.range(window_size_tensor)[None], [sequence_length, window_size_tensor])
    offsets = tf.broadcast_to(tf.range(sequence_length)[:, None], [sequence_length, window_size_tensor])
    indices = base + offsets  # [T, window_size], can exceed sequence_length - 1

    # Clamp indices for gathering (to avoid TF errors)
    # This naturally pads with last value since out-of-bounds indices get clamped to last index
    clamped_indices = tf.minimum(indices, sequence_length - 1)

    # Gather data
    gathered = tf.gather(data, clamped_indices)  # [T, window_size, ...]

    if not pad_with_last:
        # Zero out invalid positions (original behavior)
        # Create validity mask
        if per_timestep_windows is not None:
            # Variable window sizes: check both sequence bounds and per-timestep window size
            sequence_valid = indices < sequence_length  # [T, window_size]
            window_valid = base < tf.expand_dims(per_timestep_windows, -1)  # [T, window_size]
            valid_mask = tf.logical_and(sequence_valid, window_valid)
        else:
            # Fixed window size: just check sequence bounds
            valid_mask = indices < sequence_length  # [T, window_size]

        # Expand mask to match gathered shape
        mask_expanded = tf.cast(valid_mask, gathered.dtype)
        if len(gathered.shape) > 2:  # Has additional dimensions beyond [T, window]
            for _ in range(len(gathered.shape) - 2):
                mask_expanded = tf.expand_dims(mask_expanded, -1)

        gathered = gathered * mask_expanded

    return gathered


def dataset_size(ds: tf.data.Dataset) -> int:
    """Helper: try cardinality; fall back to counting if UNKNOWN/INFINITE."""
    c = ds.cardinality().numpy()  # returns int64 or negative sentinel
    if c >= 0:
        return int(c)
    # Count explicitly (works after .filter/.flat_map, etc.)
    return int(ds.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1).numpy())


def prepare_batched_dataset(
    dataset,
    shuffle,
    shuffle_buffer_size,
    seed,
    max_samples,
    batch_size,
    resize_resolution,
    primary_image_key,
    wrist_image_key,
    wrist_image_right_key=None,
):
    # Apply standard pipeline operations
    if shuffle and max_samples is None:
        dataset = dataset.repeat().shuffle(shuffle_buffer_size, seed=seed)
    else:
        raise NotImplementedError("Mode with max_samples and no val not implemented")

    decode_fn = make_decode_images_fn(
        primary_key=primary_image_key,
        wrist_key=wrist_image_key,
        wrist_right_key=wrist_image_right_key,
        resize_to=resize_resolution,
    )
    num_parallel_calls = tf.data.AUTOTUNE
    dataset = dataset.frame_map(decode_fn, num_parallel_calls)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Apply device-specific and buffering operations
    try:
        dataset = dataset.prefetch_to_device(2)
    except Exception:
        dataset = dataset.prefetch(2)
    dataset = dataset.with_ram_budget(1)

    return dataset
