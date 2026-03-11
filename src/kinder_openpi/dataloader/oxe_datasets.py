"""OXE dataset implementations for CoT RLDS datasets."""

from typing import TYPE_CHECKING

import tensorflow as tf

from kinder_openpi.dataloader.base_dataset import SingleDataset
from kinder_openpi.dataloader.utils.data_utils import ControlMode
from kinder_openpi.dataloader.utils.data_utils import NormalizationType
from kinder_openpi.dataloader.utils.data_utils import state_encoding_to_type
from kinder_openpi.dataloader.utils.dataset_utils import gather_with_padding

if TYPE_CHECKING:
    from kinder_openpi.training.config import DataConfig


class _SingleOXEDataset(SingleDataset):
    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str,
        data_dir: str,
        config: "DataConfig",
        action_horizon: int = 16,
        action_dim: int = 32,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        control_mode: ControlMode = ControlMode.JOINT_POS,
        seed: int = 0,
        split: str = "train",
        standalone: bool = True,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
    ):
        self.use_json_actions = False
        self.control_mode = control_mode

        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            config=config,
            action_dim=action_dim,
            action_horizon=action_horizon,
            action_proprio_normalization_type=action_proprio_normalization_type,
            control_mode=control_mode,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            seed=seed,
            split=split,
            standalone=standalone,
            shuffle=shuffle,
            batch_size=batch_size,
            max_samples=max_samples,
        )

    def apply_restructure(self):
        def restructure(traj):
            # extracts images, depth images and proprio from the "observation" dict
            traj_len = tf.shape(traj["action"])[0]
            old_obs = traj["observation"]
            new_obs = {}

            for new, old in self.image_obs_keys.items():
                if new == "primary":
                    img_key = self.spec.primary_image_key
                elif new == "wrist_right":
                    img_key = self.spec.wrist_image_right_key
                elif new == "wrist":
                    img_key = self.spec.wrist_image_key
                else:
                    raise ValueError(f"Unknown image key: {new}")
                # Check if key exists in observation dict
                if old is None or old not in old_obs:
                    new_obs[img_key] = tf.repeat("", traj_len)  # padding
                else:
                    new_obs[img_key] = old_obs[old]

            if self.state_obs_keys:
                # Note: instead of padding with zeros, we drop the key if it is None
                new_obs["state"] = tf.concat(
                    [tf.cast(old_obs[key], tf.float32) for key in self.state_obs_keys if key is not None],
                    axis=1,
                )
            else:
                new_obs["state"] = tf.zeros((traj_len, 0), dtype=tf.float32)  # Empty state

            # Determine state type from state encoding
            state_type_str = state_encoding_to_type(self.state_encoding)

            # Build a deterministic per-trajectory identifier using a strong hash
            # of the dataset name and the serialized action tensor. This avoids
            # relying on per-dataset metadata with inconsistent schemas.

            traj = {
                "observation": new_obs,
                "language_instruction": traj["language_instruction"],
                "actions": tf.cast(traj["action"], tf.float32),
                "dataset_name": tf.repeat(self.dataset_name, traj_len),
                "trajectory_id": traj["trajectory_id"],
                "raw_action": tf.cast(traj["action"], tf.float32),
                "control_frequency": tf.fill([traj_len], tf.cast(self.control_frequency, tf.int32)),
                "is_bimanual": tf.fill([traj_len], tf.constant(self.is_bimanual)),
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
            }

            return traj

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    def get_traj_identifier(self):
        def _get_traj_identifier(traj):
            # apply a standardization function, if provided
            if self.standardize_fn is not None:
                traj = self.standardize_fn(traj)
            traj_len = tf.shape(traj["action"])[0]
            max_steps = 128
            action_for_hash = tf.cond(
                max_steps >= traj_len,
                lambda: traj["action"],
                lambda: tf.concat([traj["action"][:64], traj["action"][-64:]], axis=0),
            )
            serialized_action = tf.io.serialize_tensor(action_for_hash)
            name_tensor = tf.constant(self.dataset_name, dtype=tf.string)
            sep1 = tf.constant("::", dtype=tf.string)
            sep2 = tf.constant("-", dtype=tf.string)
            to_hash = tf.strings.join([name_tensor, sep1, serialized_action])
            hashed = tf.strings.to_hash_bucket_strong(to_hash, 2147483647, key=[self.seed, 1337])
            traj_uid = tf.strings.join([name_tensor, sep2, tf.strings.as_string(hashed)])
            traj["trajectory_id"] = tf.repeat(traj_uid, traj_len)
            return traj

        self.dataset = self.dataset.traj_map(_get_traj_identifier, self.num_parallel_calls)

    def apply_traj_filters(self, action_key):
        def is_nonzero_length(traj):
            return tf.shape(traj[action_key])[0] > 0

        def has_any_instruction(traj):
            instr = traj["language_instruction"]
            instr = tf.reshape(instr, [-1])
            instr = tf.strings.strip(instr)
            return tf.reduce_any(tf.strings.length(instr) > 0)

        self.dataset = self.dataset.filter(has_any_instruction)

        self.dataset = self.dataset.filter(is_nonzero_length)

    def apply_frame_filters(self):
        """
        Optionally applied *per-dataset* transforms that happen at a frame level.
        """

        # Always drop frames with empty/whitespace-only prompts
        def _non_empty_prompt(frame: dict) -> tf.Tensor:
            p = tf.strings.strip(frame["prompt"])  # scalar tf.string after flatten
            return tf.strings.length(p) > 0

        self.dataset = self.dataset.filter(_non_empty_prompt)

    def apply_repack_transforms(self):
        super().apply_repack_transforms()

        def _pop_and_rename_keys(traj):
            # traj.pop("trajectory_id")
            traj["prompt"] = traj["language_instruction"]
            traj.pop("language_instruction")
            traj.pop("raw_action")
            return traj

        self.dataset = self.dataset.traj_map(_pop_and_rename_keys, self.num_parallel_calls)


class PlanningDataset(_SingleOXEDataset):
    """Dataset for planning tasks loaded from HDF5 via TFDS.

    The planning dataset contains:
    - Images: base_image (84x84x3), wrist_image (84x84x3)
    - State: 10D [arm_pos(3), arm_r6(6), gripper_pos(1)]
    - Actions: 10D action vector
    - Language: Fixed instruction per demo
    """

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
        import logging

        control_mode = self.control_mode
        logging.info(
            f"PlanningDataset.apply_traj_transforms: control_mode={control_mode}, is_delta={control_mode == ControlMode.DELTA_JOINT_POS}"
        )

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
            # traj[action_key] shape: [T, action_horizon, action_dim]

            if control_mode == ControlMode.DELTA_JOINT_POS:
                # Get state at start of each chunk (current timestep's state)
                # state shape: [T, action_dim], we need [T, 1, action_dim] for broadcasting
                current_state = traj["observation"][state_key][:, 3:10]  # [T, 7]
                current_state = tf.expand_dims(current_state, axis=1)  # [T, 1, 7]

                # Extract action components
                action_prefix = traj[action_key][:, :, :3]  # [T, action_horizon, 3] - keep absolute
                action_joints = traj[action_key][:, :, 3:10]  # [T, action_horizon, 7] - compute delta
                action_suffix = traj[action_key][:, :, 10:]  # [T, action_horizon, remaining] - keep absolute

                # Compute delta for joint positions w.r.t. current state
                delta_joints = action_joints - current_state  # [T, action_horizon, 7]

                # Reconstruct action with delta joints
                traj[action_key] = tf.concat([action_prefix, delta_joints, action_suffix], axis=-1)

            # Ensure static shape is preserved: [T, action_horizon, action_dim]
            traj[action_key].set_shape([None, action_horizon, self.action_dim])
            return traj

        self.dataset = self.dataset.traj_map(chunk_actions, self.num_parallel_calls)
