"""Policy transforms for planning dataset.

This module provides input and output transforms for the planning dataset,
which uses a fixed language instruction and loads data from HDF5 via TFDS.
"""

import dataclasses

import numpy as np
from openpi import transforms as upstream_transforms
from openpi.models.model import ModelType

from kinder_openpi.policies.utils import parse_image

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


@dataclasses.dataclass(frozen=True)
class PlanningInputs(upstream_transforms.DataTransformFn):
    """Transform planning dataset to model inputs.

    The planning dataset has:
    - Images: base_image (84x84x3), wrist_image (84x84x3) from HDF5
    - State: 8D [arm_pos(3), arm_quat(4), gripper_pos(1)]
    - Actions: 10D action vector
    - Language: Fixed instruction per demo
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    # Determines which model will be used.
    model_type: ModelType = ModelType.PI05
    has_time_dim: bool = False

    def _prepare_inputs(self, data: dict) -> dict:
        """Prepare inputs from planning dataset."""
        assert self.model_type == ModelType.PI05
        assert "observation" in data

        # Parse base image
        assert IMAGE_KEYS[0] in data["observation"], f"Missing {IMAGE_KEYS[0]} in observation"
        base_image = parse_image(data["observation"][IMAGE_KEYS[0]])
        if len(base_image.shape) == 3 and self.has_time_dim:
            base_image = base_image[None]

        if base_image is None:
            raise ValueError("Base image missing from observation")
        base_image_mask = np.False_ if np.all(base_image == 0) else np.True_

        # Parse wrist image
        if IMAGE_KEYS[1] in data["observation"]:
            wrist_image = parse_image(data["observation"][IMAGE_KEYS[1]])
            wrist_image_mask = np.False_ if np.all(wrist_image == 0.0) else np.True_
            if len(wrist_image.shape) == 3 and self.has_time_dim:
                wrist_image = wrist_image[None]
        else:
            wrist_image = np.zeros_like(base_image)
            wrist_image_mask = np.False_

        # For planning dataset, we don't have a right wrist camera
        # Use zeros as placeholder
        if IMAGE_KEYS[2] in data["observation"]:
            right_wrist_image = parse_image(data["observation"][IMAGE_KEYS[2]])
            right_wrist_image_mask = np.False_ if np.all(right_wrist_image == 0.0) else np.True_
            if len(right_wrist_image.shape) == 3 and self.has_time_dim:
                right_wrist_image = right_wrist_image[None]
        else:
            right_wrist_image = np.zeros_like(base_image)
            right_wrist_image_mask = np.False_

        images = [base_image, wrist_image, right_wrist_image]
        image_masks = [base_image_mask, wrist_image_mask, right_wrist_image_mask]

        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(IMAGE_KEYS, images[: len(IMAGE_KEYS)], strict=True)),
            "image_mask": dict(zip(IMAGE_KEYS, image_masks[: len(IMAGE_KEYS)], strict=True)),
        }

        # Get language instruction
        prompt = data.get("prompt")
        assert prompt is not None, "Prompt missing from data"
        if isinstance(prompt, bytes):  # training time
            prompt_str = prompt.decode("utf-8")
        elif isinstance(prompt, str):  # inference time
            prompt_str = prompt
        else:
            raise ValueError(f"Prompt is not a string or bytes: {prompt}")

        inputs["prompt"] = prompt_str

        # Add actions if available
        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        inputs["sample_mask"] = True

        return inputs

    def __call__(self, data: dict) -> dict:
        """Transform planning dataset sample to model input format."""
        return self._prepare_inputs(data)


@dataclasses.dataclass(frozen=True)
class PlanningOutputs(upstream_transforms.DataTransformFn):
    """Transform model outputs for planning dataset.

    For planning dataset, we primarily use the action predictions directly.
    Reasoning/language outputs are optional for debugging.
    """

    def __call__(self, data: dict) -> dict:
        """Transform model outputs.

        Args:
            data: Dict containing model outputs, typically:
                - "actions": predicted actions

        Returns:
            Dict with processed outputs
        """
        # For planning dataset, we primarily care about action predictions
        # Return as-is for now, can add post-processing if needed
        output = {}
        output["actions"] = data["actions"][:, :11]

        return output
