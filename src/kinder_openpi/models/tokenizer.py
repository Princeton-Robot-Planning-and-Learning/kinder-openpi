import dataclasses
import logging
from typing import Literal

import numpy as np
from openpi.models import tokenizer as _tokenizer
import sentencepiece

import kinder_openpi.shared.download as download


@dataclasses.dataclass
class StateTemplate:
    """Template for formatting discretized state values.

    Allows flexible representation of state vectors with custom labels and formatting.
    """

    # Dimension labels in order (e.g., ["x", "y", "z", "rot1x", ...])
    # If None or shorter than values, uses generic labels
    dim_labels: list[str] | None = None

    # Format string for each dimension value
    # Can use {label} and {value} placeholders
    # e.g., "{label}={value:03d}" for "x=134" or just "{value}" for "134"
    dim_format: str = "{value}"

    # Separator between dimensions
    separator: str = " "

    def format_state(self, values: np.ndarray) -> str:
        """Format discretized state values according to template.

        Args:
            values: Array of discretized state values

        Returns:
            Formatted string representation
        """
        parts = []
        for i, val in enumerate(values):
            # Use provided label or generate generic one
            if self.dim_labels and i < len(self.dim_labels):
                label = self.dim_labels[i]
            else:
                label = f"dim{i}"

            parts.append(self.dim_format.format(label=label, value=int(val)))

        return self.separator.join(parts)


@dataclasses.dataclass
class StateDiscretizationConfig:
    """Configuration for discretizing state vectors into text."""

    bins: int = 256
    min_dim: int = 7  # Minimum number of dimensions to include (avoid over-trimming)
    range_min: float = -1.0
    range_max: float = 1.0
    template: StateTemplate | None = None  # If None, uses default space-separated format

    def discretize_state(self, state: np.ndarray) -> str:
        """Discretize state vector into string representation.

        Trims trailing zero-padded dimensions and discretizes to bins.
        Uses the configured StateTemplate if provided, otherwise defaults to space-separated values.

        Args:
            state: State vector to discretize

        Returns:
            Formatted string representation of discretized state
        """
        state_arr = np.asarray(state)
        eps = 1e-8

        # Trim zero-padded dimensions
        if state_arr.ndim == 1:
            non_zero_mask = np.abs(state_arr) > eps
            last_idx = int(np.nonzero(non_zero_mask)[0][-1]) + 1 if np.any(non_zero_mask) else 0
            last_idx = max(last_idx, self.min_dim)
            trimmed = state_arr[:last_idx]
        else:
            flat = state_arr.reshape(-1, state_arr.shape[-1])
            non_zero_cols = np.any(np.abs(flat) > eps, axis=0)
            last_idx = int(np.nonzero(non_zero_cols)[0][-1]) + 1 if np.any(non_zero_cols) else 0
            last_idx = max(last_idx, self.min_dim)
            trimmed = state_arr[..., :last_idx].reshape(-1)

        if trimmed.size > 0:
            bins = np.linspace(self.range_min, self.range_max, self.bins + 1)[:-1]
            discretized_state = np.digitize(trimmed, bins=bins) - 1

            # Use template if provided, otherwise default to space-separated
            if self.template is not None:
                return self.template.format_state(discretized_state)
            return " ".join(map(str, discretized_state))
        return ""


@dataclasses.dataclass
class PromptComponent:
    """A modular component of a prompt.

    Each component can be one of:
    - task_prefix: Format for task instruction (e.g., "Task: {prompt}")
    - state_prefix: Format for state (e.g., "State ({state_label}): {state}")
    - schema: Schema/instruction text (e.g., coordinate system description)
    - action_prefix: Prefix before action output (e.g., "Action: ")
    """

    type: Literal["task_prefix", "state_prefix", "schema", "action_prefix"]
    template: str
    # Whether to include state type label in state prefix
    include_state_type: bool = True


@dataclasses.dataclass
class PromptFormat:
    """Defines how to format prompts for tokenization using modular components.

    This allows easy extension to support different prompt formats by composing
    components in different ways.
    """

    name: str
    components: list[PromptComponent]
    state_config: StateDiscretizationConfig | None = None
    # Separator between components (e.g., ", " or "\n")
    separator: str = ""

    @property
    def include_state(self) -> bool:
        """Check if this format includes state."""
        return any(c.type == "state_prefix" for c in self.components)

    def format_prompt(self, prompt: str, state: np.ndarray | None = None, state_type: str | None = None) -> str:
        """Format the prompt with optional state and state type.

        Args:
            prompt: The task prompt/instruction
            state: Optional state vector to discretize and include
            state_type: Optional state type ("joint_pos", "eef_pose", "none")

        Returns:
            Formatted prompt string ready for tokenization
        """
        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ").rstrip(".")

        # Prepare state-related variables
        state_str = ""
        state_label = ""
        if state is not None and state_type != "none":
            # Map state_type to human-readable label
            state_type_labels = {
                "joint_pos": "joint position",
                "eef_pose": "end-effector pose",
            }
            state_label = state_type_labels.get(state_type, state_type) if state_type else ""

            if self.state_config is not None:
                state_str = self._discretize_state(state)

        # Build prompt by chaining components
        parts = []
        for component in self.components:
            if component.type == "task_prefix":
                parts.append(component.template.format(prompt=cleaned_prompt))
            elif component.type == "state_prefix":
                if state is None or state_type == "none":
                    # Skip state component if no state
                    if component.include_state_type:
                        parts.append(component.template.format(state="", state_label="None"))
                    else:
                        parts.append(component.template.format(state="", state_label=""))
                else:
                    if self.state_config is None:
                        raise ValueError(f"State config required for prompt format '{self.name}'")
                    if component.include_state_type:
                        parts.append(component.template.format(state=state_str, state_label=state_label))
                    else:
                        parts.append(component.template.format(state=state_str, state_label=""))
            elif component.type == "schema" or component.type == "action_prefix":
                parts.append(component.template)
        return self.separator.join(parts)

    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretize state vector into string representation.

        Trims trailing zero-padded dimensions and discretizes to bins.
        Uses the configured StateTemplate if provided, otherwise defaults to space-separated values.
        """
        assert self.state_config is not None
        return self.state_config.discretize_state(state)


# Predefined prompt formats - easily extensible by adding new instances
PI05_PROMPT_FORMAT = PromptFormat(
    name="pi05",
    components=[
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "State{state_label}: {state}", include_state_type=False),
        PromptComponent("action_prefix", "Action: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7),
    separator=", ",
)

# Registry for easy lookup
PROMPT_FORMAT_REGISTRY = {
    "pi05": PI05_PROMPT_FORMAT,
}


class PaligemmaTokenizer(_tokenizer.PaligemmaTokenizer):
    def __init__(
        self,
        max_len: int = 48,
        prompt_format: Literal["pi05",] | PromptFormat = "pi05",
    ):
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._max_len = max_len
        self._stop_token_id = self._tokenizer.eos_id()

        # Support both string and PromptFormat instance
        if isinstance(prompt_format, str):
            if prompt_format not in PROMPT_FORMAT_REGISTRY:
                raise ValueError(
                    f"Unknown prompt format: {prompt_format}. Available formats: {list(PROMPT_FORMAT_REGISTRY.keys())}"
                )
            self._prompt_format = PROMPT_FORMAT_REGISTRY[prompt_format]
        else:
            self._prompt_format = prompt_format

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        state_type: str | None = None,
        prompt_format: PromptFormat | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Resolve prompt format

        if prompt_format is None:
            fmt = self._prompt_format
        elif isinstance(prompt_format, str):
            if prompt_format not in PROMPT_FORMAT_REGISTRY:
                raise ValueError(
                    f"Unknown prompt format: {prompt_format}. Available formats: {list(PROMPT_FORMAT_REGISTRY.keys())}"
                )
            fmt = PROMPT_FORMAT_REGISTRY[prompt_format]
        else:
            fmt = prompt_format

        formatted_prompt = fmt.format_prompt(prompt, state, state_type)

        # Tokenize
        pad_id = self._tokenizer.pad_id()

        tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)

        if len(tokens) > self._max_len:
            logging.warning(
                f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                "Consider increasing the `max_token_len` in your model config if this happens frequently."
            )
            tokens = tokens[: self._max_len]

        # Left pad to max length for generation/training
        pad_count = self._max_len - len(tokens)
        if pad_count > 0:
            tokens = [pad_id] * pad_count + tokens

        # Create masks
        attn_mask = np.zeros(self._max_len, dtype=bool)

        # Mark all non-pad positions as valid for attention
        attn_mask[pad_count:] = True

        return (
            np.asarray(tokens, dtype=np.int32),
            attn_mask,
        )

    def decode(self, tokens: np.ndarray) -> str:
        """Decode tokens back to a string, skipping special tokens and placeholders."""
        if not isinstance(tokens, list):
            tokens = tokens.tolist()

        return self._tokenizer.decode(tokens).strip()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
        """Encode a string to tokens."""
        return self._tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
