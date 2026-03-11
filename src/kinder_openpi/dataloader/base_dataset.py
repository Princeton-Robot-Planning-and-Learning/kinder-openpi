"""Base dataset class for RLDS datasets."""

import contextlib
import logging
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import psutil
import tensorflow as tf
import tensorflow_datasets as tfds

from kinder_openpi.dataloader.utils.data_utils import ControlMode
from kinder_openpi.dataloader.utils.data_utils import NormalizationType
from kinder_openpi.dataloader.utils.data_utils import load_dataset_kwargs
from kinder_openpi.dataloader.utils.dataset_utils import gather_with_padding
from kinder_openpi.dataloader.utils.dataset_utils import prepare_batched_dataset
from kinder_openpi.dataloader.utils.specs import RldsDatasetSpec
from kinder_openpi.shared.normalize_adapter import check_dataset_statistics
from kinder_openpi.shared.normalize_adapter import get_dataset_statistics

if TYPE_CHECKING:
    from kinder_openpi.training.config import DataConfig


class SingleDataset:
    spec: ClassVar[RldsDatasetSpec] = RldsDatasetSpec()

    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str,
        data_dir: str,
        config: "DataConfig",
        action_dim: int = 32,
        action_horizon: int = 16,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        control_mode: ControlMode = ControlMode.JOINT_POS,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        seed: int = 0,
        split: str = "train",
        standalone: bool = False,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
    ):
        self.config = config
        self.seed = seed
        self.dataset_name = dataset_name
        self.action_dim = action_dim
        self.action_proprio_normalization_type = action_proprio_normalization_type
        self.control_mode = control_mode
        self.use_wrist_image = bool(config.use_wrist_image)
        self.standalone = standalone
        dataset_kwargs = load_dataset_kwargs(
            data_dir,
            load_camera_views=("primary", "wrist", "wrist_right"),
        )

        logging.info(f"Dataset kwargs: {dataset_kwargs}")
        self.control_frequency: int = int(dataset_kwargs["control_frequency"])  # constant for this dataset
        self.standardize_fn = dataset_kwargs["standardize_fn"]
        self.image_obs_keys = dataset_kwargs["image_obs_keys"]
        self.state_obs_keys = dataset_kwargs["state_obs_keys"]
        self.state_encoding = dataset_kwargs["state_encoding"]
        self.action_encoding = dataset_kwargs["action_encoding"]
        self.is_bimanual = dataset_kwargs.get("is_bimanual", False)

        self.num_parallel_reads = tf.data.AUTOTUNE if num_parallel_reads == -1 else num_parallel_reads
        self.num_parallel_calls = tf.data.AUTOTUNE if num_parallel_calls == -1 else num_parallel_calls

        # ------------------------------------------------------------------
        # Configure Tensorflow with no GPU/TPU devices to avoid clobbering JAX/TPU runtime
        # ------------------------------------------------------------------
        tf.config.set_visible_devices([], "GPU")
        with contextlib.suppress(Exception):
            tf.config.set_visible_devices([], "TPU")

        # Set global seed for file-level operations (shuffle, interleave)
        # Data-level randomness uses stateless ops with explicit seeds
        tf.random.set_seed(self.seed)

        self.builder = self.build_dataset_builder(dataset_name, data_dir)

        # Check if we have cached statistics
        cached_stats, _, _ = check_dataset_statistics(self.builder.data_dir)

        # If no cached stats, compute them first
        if cached_stats is None or self.config.force_recompute_stats:
            logging.info(f"No cached statistics found for {dataset_name} or force recompute. Computing statistics...")
            # Build temporary dataset for stats computation (with sharding - multi-host aggregation handles it)
            self.dataset = self.build_dataset(self.builder)
            logging.info(f"Stats computation: built dataset, cardinality={self.dataset.cardinality().numpy()}")
            self.get_traj_identifier()
            logging.info(
                f"Stats computation: applied get_traj_identifier, cardinality={self.dataset.cardinality().numpy()}"
            )
            self.apply_restructure()
            logging.info(
                f"Stats computation: applied apply_restructure, cardinality={self.dataset.cardinality().numpy()}"
            )
            # Apply trajectory transforms (including action chunking) before computing stats
            self.apply_traj_transforms(action_horizon=action_horizon)
            logging.info(
                f"Stats computation: applied apply_traj_transforms, cardinality={self.dataset.cardinality().numpy()}"
            )

            # Compute and save statistics
            cached_stats = get_dataset_statistics(
                self.dataset,
                save_dir=self.builder.data_dir,
                action_key="actions",
                state_key="state",
                action_dim=action_dim,
            )
            logging.info(f"Statistics computed and saved for {dataset_name}")

        # Now rebuild dataset using cached stats path for consistent ordering
        self.dataset = self.build_dataset(self.builder)
        self.get_traj_identifier()

        # Set statistics before filtering (needed for dataset-specific filters)
        self.dataset_statistics = cached_stats

        # Apply operations in consistent order: filter -> split -> restructure
        self.apply_traj_filters(action_key="action")
        self.apply_restructure()

        self.apply_traj_transforms(
            action_horizon=action_horizon,
        )

        self.apply_repack_transforms()

        # self.dataset = self.dataset.shuffle(60_000, seed=self.seed)

        self.apply_flatten()

        self.apply_frame_filters()

        if standalone:
            # Store the pre-batched dataset for creating checkpointable versions
            self._pre_batched_dataset = self.dataset

            # Apply common shuffling/take/cache behavior
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

    def build_dataset_builder(self, ds_name, data_dir):
        if ds_name == "fmb":
            ds_name = "fmb:1.0.0"
        if ds_name == "dobbe":
            ds_name = "dobbe:0.0.1"
        return tfds.builder(ds_name, data_dir=data_dir)

    def build_dataset(self, builder):
        opts = tf.data.Options()
        # Always use deterministic operations for reproducibility
        # File interleaving will be deterministic but still provide good mixing
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_fusion = True
        cpu_count = psutil.cpu_count(logical=True) or 16
        opts.experimental_threading.private_threadpool_size = int(max(16, cpu_count))
        dataset = dl.DLataset.from_rlds(
            builder,
            split="all",
            shuffle=True,
            num_parallel_reads=self.num_parallel_reads,
        )
        # dataset = dataset.shard(jax.process_count(), jax.process_index())
        # Repeat early to increase interleaving across files/episodes
        dataset = dataset.with_options(opts)
        return dataset

    def apply_traj_transforms(
        self,
        action_horizon: int,
        summation_steps: int = 30,
        action_key: str = "actions",
        state_key: str = "state",
    ):
        """
        Compare to original transforms, we omit the following:
        - skip_unlabeled
        - max_action
        - max_proprio
        - goal_relabeling
        - drop_goal_or_instruction
        - subsample_length
        """

        def pad_action_state(traj):
            # Pad actions to action_dim (only if not already padded)
            action_last_dim = tf.shape(traj[action_key])[-1]
            pad_amount_action = tf.maximum(0, self.action_dim - action_last_dim)
            traj[action_key] = tf.pad(traj[action_key], [[0, 0], [0, pad_amount_action]])
            # Ensure static shape is preserved
            traj[action_key].set_shape([None, self.action_dim])

            # Pad state to action_dim (only if not already padded)
            state_last_dim = tf.shape(traj["observation"][state_key])[-1]
            pad_amount_state = tf.maximum(0, self.action_dim - state_last_dim)
            traj["observation"][state_key] = tf.pad(
                traj["observation"][state_key],
                [[0, 0], [0, pad_amount_state]],
            )
            # Ensure static shape is preserved
            traj["observation"][state_key].set_shape([None, self.action_dim])
            return traj

        self.dataset = self.dataset.traj_map(pad_action_state, self.num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks with proper last-value padding.

            For delta_joint_pos mode:
            - Calculate delta only for action[3:10] (joint positions)
            - Delta is computed w.r.t. state[3:10] at the start of each chunk
            - Keep action[0:3] and action[10:] as absolute values
            """
            traj_len = tf.shape(traj[action_key])[0]

            # Standard chunking with last-value padding
            traj[action_key] = gather_with_padding(
                data=traj[action_key],
                sequence_length=traj_len,
                window_size=action_horizon,
                pad_with_last=True,
            )

            # Ensure static shape is preserved: [T, action_horizon, action_dim]
            traj[action_key].set_shape([None, action_horizon, self.action_dim])
            return traj

        self.dataset = self.dataset.traj_map(chunk_actions, self.num_parallel_calls)

    def apply_repack_transforms(self):
        def common(traj):
            # Add empty caption field for robot datasets (VQA datasets will populate this)
            traj_len = tf.shape(traj["actions"])[0]
            traj["caption"] = tf.repeat(tf.constant("", dtype=tf.string), traj_len)
            return traj

        self.dataset = self.dataset.traj_map(common, self.num_parallel_calls)

    def get_traj_identifier(self):
        raise NotImplementedError

    def apply_restructure(self):
        raise NotImplementedError

    def apply_traj_filters(self):
        raise NotImplementedError

    def apply_frame_filters(self):
        raise NotImplementedError

    def apply_flatten(self):
        # Flatten: map from trajectory dataset to dataset of individual action chunks
        self.dataset = self.dataset.flatten(num_parallel_calls=self.num_parallel_calls)

    def __iter__(self):
        assert self.standalone, "This dataset is not standalone"
        it = self.dataset.as_numpy_iterator()
        while True:
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            yield batch

    def __len__(self):
        return self.dataset_statistics["state"].num_transitions
