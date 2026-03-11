"""Multi-dataset orchestration for RLDS datasets."""

import logging
import os
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import numpy as np
import tensorflow as tf

from kinder_openpi.dataloader.oxe_datasets import PlanningDataset
from kinder_openpi.dataloader.utils.data_utils import NormalizationType
from kinder_openpi.dataloader.utils.data_utils import allocate_threads
from kinder_openpi.dataloader.utils.data_utils import pprint_data_mixture
from kinder_openpi.dataloader.utils.data_utils import state_encoding_to_type
from kinder_openpi.dataloader.utils.dataset_utils import prepare_batched_dataset
from kinder_openpi.dataloader.utils.mixtures import OXE_NAMED_MIXTURES
from kinder_openpi.dataloader.utils.specs import RldsDatasetSpec
from kinder_openpi.transforms import NormalizeActionAndProprio

if TYPE_CHECKING:
    from kinder_openpi.training.config import DataConfig


class OXEDatasets:
    spec: ClassVar[RldsDatasetSpec] = RldsDatasetSpec()
    # Use centralized VQA dataset registry

    def __init__(
        self,
        config: "DataConfig",
        data_dir: str,
        action_dim: int = 32,
        action_horizon: int = 16,
        seed: int = 0,
        split: str = "train",
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        balance_weights: bool = True,  # noqa: FBT001, FBT002
        hash_tables: dict = None,
        standalone=True,
    ):
        self.hash_tables = hash_tables
        self.batch_size = batch_size

        if config.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[config.data_mix]
            dataset_names = [l[0] for l in mixture_spec]
            sample_weights = [l[1] for l in mixture_spec]

        else:
            dataset_names = [config.data_mix]
            sample_weights = [1.0]

        self.config = config

        total_threads = len(os.sched_getaffinity(0))
        total_read_threads = int(total_threads * 0.4)
        total_transform_threads = int(total_threads * 0.4)
        logging.info(f"Total read threads, {total_read_threads}")
        logging.info(f"Total transform threads, {total_transform_threads}")
        logging.info(f"Length of sample weights: {len(sample_weights)}")

        # Allocate Threads based on Weights
        threads_per_dataset = allocate_threads(total_transform_threads, np.array(sample_weights))
        reads_per_dataset = allocate_threads(total_read_threads, np.array(sample_weights))

        logging.info("Threads per Dataset: %s", threads_per_dataset)
        logging.info("Reads per Dataset: %s", reads_per_dataset)

        datasets, dataset_sizes, all_dataset_statistics = [], [], {}
        dataset_state_encodings = {}  # Track state encoding for each dataset
        logging.info("Constructing datasets...")
        for dataset_name, threads, reads in zip(  # noqa: B905
            dataset_names,
            threads_per_dataset,
            reads_per_dataset,
        ):
            assert threads != tf.data.AUTOTUNE, "threads should not be AUTOTUNE"
            assert reads != tf.data.AUTOTUNE, "reads should not be AUTOTUNE"
            logging.info(f"Creating dataset {dataset_name} with control_mode={config.control_mode}")
            kwargs = dict(
                data_dir=data_dir,
                config=config,
                action_horizon=action_horizon,
                action_dim=action_dim,
                seed=seed,
                split=split,
                action_proprio_normalization_type=action_proprio_normalization_type,
                control_mode=config.control_mode,
                num_parallel_reads=threads,
                num_parallel_calls=threads,
                standalone=False,
            )

            if dataset_name.startswith("planning"):
                ds = PlanningDataset(
                    dataset_name=dataset_name,
                    **kwargs,
                )
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented in OXEDatasets")
            # Don't restrict RAM budget on individual datasets
            # Let the final mixed dataset handle memory management
            datasets.append(ds.dataset)
            dataset_statistics = ds.dataset_statistics
            dataset_sizes.append(dataset_statistics["state"].num_transitions)
            all_dataset_statistics[dataset_name] = dataset_statistics
            dataset_state_encodings[dataset_name] = config.state_encoding  # Track state encoding

        # Get the indices of the "primary" datasets (i.e., datasets with sample_weight == 1.0)
        # primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])
        primary_dataset_indices = np.array(list(range(len(sample_weights))))

        # Balance and Normalize Weights
        if balance_weights:
            sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
        unnormalized_sample_weights = sample_weights.copy()
        sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
        #   =>> Note :: Only counting the "primary" datasets (i.e., datasets with sample_weight == 1.0)
        dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())

        self.sample_weights = sample_weights
        self.unnormalized_sample_weights = unnormalized_sample_weights
        self.dataset_statistics = all_dataset_statistics
        self.dataset_length = dataset_len

        pprint_data_mixture(dataset_names, sample_weights)

        global_stats_dir = data_dir
        global_stats = self._compute_or_load_global_stats(
            datasets=datasets,
            dataset_names=dataset_names,
            all_dataset_statistics=all_dataset_statistics,
            dataset_state_encodings=dataset_state_encodings,
            save_dir=global_stats_dir,
            action_dim=action_dim,
        )
        logging.info("Applying global normalization with stats: %s", global_stats)

        # Apply state-type-specific normalization to each dataset BEFORE interleaving
        # This avoids tf.case/tf.cond issues entirely
        normalized_datasets = []
        for ds_name, ds in zip(dataset_names, datasets):
            state_enc = dataset_state_encodings[ds_name]
            state_type = state_encoding_to_type(state_enc)

            # Create normalizer for this state type
            stats = {"actions": global_stats["actions"]}
            state_key_name = f"state_{state_type}"
            if state_key_name in global_stats:
                stats["state"] = global_stats[state_key_name]

            normalizer = NormalizeActionAndProprio(
                norm_stats=stats,
                normalization_type=action_proprio_normalization_type,
                action_key="actions",
                state_key="state",
            )

            # Apply normalizer to this dataset
            normalized_datasets.append(ds.map(normalizer, num_parallel_calls=tf.data.AUTOTUNE))

        # Interleave the normalized datasets
        self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
            normalized_datasets, self.sample_weights, rerandomize_each_iteration=True, seed=seed
        )
        self.global_statistics = global_stats

        # Store the pre-batched dataset for creating checkpointable versions
        self._pre_batched_dataset = self.dataset

        self.dataset = prepare_batched_dataset(
            dataset=self.dataset,
            shuffle=shuffle,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
            batch_size=batch_size,
            resize_resolution=config.resize_resolution,
            primary_image_key=self.spec.primary_image_key,
            wrist_image_key=self.spec.wrist_image_key,
            wrist_image_right_key=self.spec.wrist_image_right_key,
        )

    def _compute_or_load_global_stats(
        self,
        datasets: list[dl.DLataset],
        dataset_names: list[str],
        all_dataset_statistics: dict,
        dataset_state_encodings: dict,
        save_dir: str,
        action_dim: int,
    ) -> dict:
        """Compute or load global normalization statistics across all datasets.

        When using global normalization, we compute mean/std across all datasets
        weighted by their sample counts. Statistics are computed separately for
        each state type (joint_pos, eef_pose, none).

        Note: The statistics are padded to action_dim to match the padded tensors.
        """
        from kinder_openpi.shared.normalize_adapter import ExtendedNormStats

        # # Try to load cached global stats
        # try:
        #     global_stats = load(save_dir)
        #     logging.info(f"Loaded cached global normalization stats from {save_dir}")
        #     return global_stats
        # except FileNotFoundError:
        #     logging.info("Computing global normalization statistics from scratch...")

        # Group datasets by state type
        datasets_by_state_type = {"joint_pos": [], "eef_pose": [], "none": []}
        for dataset_name in dataset_names:
            state_encoding = dataset_state_encodings[dataset_name]
            state_type = state_encoding_to_type(state_encoding)
            datasets_by_state_type[state_type].append(dataset_name)

        # Compute weighted global statistics for actions
        # Note: Action stats are shared across ALL datasets regardless of state type
        total_action_n = 0
        action_weighted_sum = np.zeros(action_dim, dtype=np.float32)

        for dataset_name, stats in all_dataset_statistics.items():
            action_n = stats["actions"].num_transitions
            total_action_n += action_n
            # Pad each dataset's mean to action_dim before accumulating
            action_mean_padded = np.pad(
                stats["actions"].mean, (0, action_dim - len(stats["actions"].mean)), mode="constant"
            )
            action_weighted_sum += action_mean_padded * action_n

        action_global_mean = action_weighted_sum / total_action_n

        # Compute weighted variance using parallel axis theorem
        action_var_sum = np.zeros_like(action_global_mean)

        for dataset_name, stats in all_dataset_statistics.items():
            action_n = stats["actions"].num_transitions

            # Pad local stats to action_dim for comparison with global stats
            action_local_mean = np.pad(
                stats["actions"].mean, (0, action_dim - len(stats["actions"].mean)), mode="constant"
            )
            action_local_std = np.pad(
                stats["actions"].std, (0, action_dim - len(stats["actions"].std)), mode="constant", constant_values=0.0
            )

            # var_i + (mean_i - global_mean)^2
            action_local_var = np.square(action_local_std)
            action_mean_diff_sq = np.square(action_local_mean - action_global_mean)
            action_var_sum += action_n * (action_local_var + action_mean_diff_sq)

        action_global_var = action_var_sum / total_action_n
        action_global_std = np.sqrt(action_global_var)

        # For quantiles, use conservative bounds (global min/max across all datasets)
        # Pad each dataset's quantiles to action_dim first, then compute min/max
        action_q01_padded = [
            np.pad(
                stats["actions"].q01, (0, action_dim - len(stats["actions"].q01)), mode="constant", constant_values=0
            )
            for stats in all_dataset_statistics.values()
        ]
        action_q99_padded = [
            np.pad(
                stats["actions"].q99, (0, action_dim - len(stats["actions"].q99)), mode="constant", constant_values=0
            )
            for stats in all_dataset_statistics.values()
        ]
        action_q01 = np.min(action_q01_padded, axis=0)
        action_q99 = np.max(action_q99_padded, axis=0)

        # Compute actual global min/max from per-dataset statistics
        action_min_padded = [
            np.pad(
                stats["actions"].min, (0, action_dim - len(stats["actions"].min)), mode="constant", constant_values=0
            )
            for dataset_name, stats in all_dataset_statistics.items()
        ]
        action_max_padded = [
            np.pad(
                stats["actions"].max, (0, action_dim - len(stats["actions"].max)), mode="constant", constant_values=0
            )
            for dataset_name, stats in all_dataset_statistics.items()
        ]
        action_global_min = np.min(action_min_padded, axis=0)
        action_global_max = np.max(action_max_padded, axis=0)

        global_stats = {
            "actions": ExtendedNormStats(
                mean=action_global_mean,
                std=action_global_std,
                q01=action_q01,
                q99=action_q99,
                min=action_global_min,
                max=action_global_max,
                num_transitions=total_action_n,
                num_trajectories=sum(stats["actions"].num_trajectories for stats in all_dataset_statistics.values()),
            ),
        }

        # Log global action statistics per dimension
        logging.info("=" * 80)
        logging.info("Global Action Statistics per Dimension:")
        logging.info("-" * 80)
        for dim_idx in range(action_dim):
            logging.info(
                f"Action dim {dim_idx:2d}: "
                f"min={action_global_min[dim_idx]:9.4f}, "
                f"max={action_global_max[dim_idx]:9.4f}, "
                f"q01={action_q01[dim_idx]:9.4f}, "
                f"q99={action_q99[dim_idx]:9.4f}, "
                f"mean={action_global_mean[dim_idx]:9.4f}, "
                f"std={action_global_std[dim_idx]:8.4f}"
            )
        logging.info("=" * 80)

        # Compute separate state statistics for each state type. VQA datasets already skipped above.
        for state_type, ds_names in datasets_by_state_type.items():
            if not ds_names:
                continue  # Skip if no datasets of this type

            # Skip normalization for "none" state type (empty state)
            if state_type == "none":
                continue

            # Compute weighted statistics for this state type
            state_stats_subset = {name: all_dataset_statistics[name] for name in ds_names}
            total_state_n = sum(stats["state"].num_transitions for stats in state_stats_subset.values())

            if total_state_n == 0:
                continue  # Skip if no transitions

            # Initialize with action_dim size (all states will be padded to this size)
            state_weighted_sum = np.zeros(action_dim, dtype=np.float32)

            for dataset_name, stats in state_stats_subset.items():
                state_n = stats["state"].num_transitions
                # Pad state mean to action_dim before accumulating
                state_mean_padded = np.pad(
                    stats["state"].mean, (0, action_dim - len(stats["state"].mean)), mode="constant"
                )
                state_weighted_sum += state_mean_padded * state_n

            state_global_mean = state_weighted_sum / total_state_n

            # Pad global mean to action_dim
            state_global_mean = np.pad(state_global_mean, (0, action_dim - len(state_global_mean)), mode="constant")

            # Compute weighted variance
            state_var_sum = np.zeros_like(state_global_mean)

            for dataset_name, stats in state_stats_subset.items():
                state_n = stats["state"].num_transitions

                # Pad local stats to action_dim
                state_local_mean = np.pad(
                    stats["state"].mean, (0, action_dim - len(stats["state"].mean)), mode="constant"
                )
                state_local_std = np.pad(
                    stats["state"].std, (0, action_dim - len(stats["state"].std)), mode="constant", constant_values=0.0
                )

                # var_i + (mean_i - global_mean)^2
                state_local_var = np.square(state_local_std)
                state_mean_diff_sq = np.square(state_local_mean - state_global_mean)
                state_var_sum += state_n * (state_local_var + state_mean_diff_sq)

            state_global_var = state_var_sum / total_state_n
            state_global_std = np.sqrt(state_global_var)

            # For quantiles, use conservative bounds
            # Pad each dataset's quantiles to action_dim first, then compute min/max
            state_q01_padded = [
                np.pad(
                    stats["state"].q01, (0, action_dim - len(stats["state"].q01)), mode="constant", constant_values=0
                )
                for stats in state_stats_subset.values()
            ]
            state_q99_padded = [
                np.pad(
                    stats["state"].q99, (0, action_dim - len(stats["state"].q99)), mode="constant", constant_values=0
                )
                for stats in state_stats_subset.values()
            ]
            state_q01 = np.min(state_q01_padded, axis=0)
            state_q99 = np.max(state_q99_padded, axis=0)

            # Compute actual global min/max from per-dataset statistics
            state_min_padded = [
                np.pad(
                    stats["state"].min, (0, action_dim - len(stats["state"].min)), mode="constant", constant_values=0
                )
                for stats in state_stats_subset.values()
            ]
            state_max_padded = [
                np.pad(
                    stats["state"].max, (0, action_dim - len(stats["state"].max)), mode="constant", constant_values=0
                )
                for stats in state_stats_subset.values()
            ]
            state_global_min = np.min(state_min_padded, axis=0)
            state_global_max = np.max(state_max_padded, axis=0)

            # Store with state type-specific key
            global_stats[f"state_{state_type}"] = ExtendedNormStats(
                mean=state_global_mean,
                std=state_global_std,
                q01=state_q01,
                q99=state_q99,
                min=state_global_min,
                max=state_global_max,
                num_transitions=total_state_n,
                num_trajectories=sum(stats["state"].num_trajectories for stats in state_stats_subset.values()),
            )

            # Log global state statistics per dimension
            logging.info("=" * 80)
            logging.info(f"Global State Statistics per Dimension (type: {state_type}):")
            logging.info("-" * 80)
            for dim_idx in range(action_dim):
                logging.info(
                    f"State dim {dim_idx:2d}: "
                    f"min={state_global_min[dim_idx]:9.4f}, "
                    f"max={state_global_max[dim_idx]:9.4f}, "
                    f"q01={state_q01[dim_idx]:9.4f}, "
                    f"q99={state_q99[dim_idx]:9.4f}, "
                    f"mean={state_global_mean[dim_idx]:9.4f}, "
                    f"std={state_global_std[dim_idx]:8.4f}"
                )
            logging.info("=" * 80)

        return global_stats

    def __iter__(self):
        it = self.dataset.as_numpy_iterator()
        while True:
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            yield batch

    def __len__(self):
        return self.dataset_length
