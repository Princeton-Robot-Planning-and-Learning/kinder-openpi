"""See _CONFIGS for the list of available configs."""

import abc
import dataclasses
import difflib
import logging
import pathlib
from typing import Literal, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as upstream_config
import openpi.training.optimizer as _optimizer
import openpi.transforms as upstream_transforms
from typing_extensions import override
import tyro

from kinder_openpi.dataloader.utils.data_utils import ActionEncoding
from kinder_openpi.dataloader.utils.data_utils import ControlMode
from kinder_openpi.dataloader.utils.data_utils import NormalizationType
from kinder_openpi.dataloader.utils.data_utils import StateEncoding
from kinder_openpi.models.tokenizer import PaligemmaTokenizer
import kinder_openpi.policies.planning_policy as planning_policy
from kinder_openpi.shared.download import maybe_download
import kinder_openpi.shared.normalize_adapter as _normalize_adapter
import kinder_openpi.training.weight_loaders as weight_loaders
from kinder_openpi.transforms import TokenizePrompt

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


def _to_path(base: str | pathlib.Path, *extra: str) -> pathlib.Path | epath.Path:
    """
    Join `base` with any `extra` segments, returning:
      • `pathlib.Path` for normal file-system paths
      • `epath.Path`   for `gs://` URIs
    """
    base = str(base)  # in case the attr is already a Path object
    if base.startswith("gs://"):
        # epath.Path already mimics pathlib semantics (`/`, `.joinpath`, etc.)
        return epath.Path(base).joinpath(*extra)  # no `.resolve()` on GCS
    return (pathlib.Path(base).joinpath(*extra)).resolve()


def build_cosine_lr(
    *,
    warmup_steps: int = 1_000,
    peak_lr: float = 1e-4,
    decay_steps: int = 1_000_000,
    decay_lr: float = 1e-4,
) -> _optimizer.LRScheduleConfig:
    """Shared cosine LR schedule used by most experiments."""
    return _optimizer.CosineDecaySchedule(
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        decay_steps=decay_steps,
        decay_lr=decay_lr,
    )


@dataclasses.dataclass(frozen=True)
class DataConfig(upstream_config.DataConfig):
    shuffle_buffer_size: int = 250_000
    # Optional cap on number of unique flattened samples for overfitting tests
    max_samples: int | None = None
    use_wrist_image: bool = True
    wrist_image_dropout_prob: float = 0.0
    # One of {"droid", "oxe", "combined"}; used by the RLDS loader switch.
    state_encoding: StateEncoding = StateEncoding.POS_EULER
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS
    control_mode: ControlMode = ControlMode.JOINT_POS
    # Normalization type for actions and proprioceptive state.
    # CLI: --data.action_proprio_normalization_type {normal|bounds|bounds_q99}
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL
    resize_resolution: tuple[int, int] = (224, 224)

    force_recompute_stats: bool = False

    data_mix: str | None = "planning_dataset"


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(upstream_config.ModelTransformFactory):
    """Creates model transforms for standard pi0 models."""

    prompt_format: Literal["pi05"] = "pi05"

    def __call__(self, model_config: _model.BaseModelConfig) -> upstream_transforms.Group:
        if model_config.model_type == ModelType.PI05:
            assert isinstance(model_config, pi0_config.Pi0Config)
            return upstream_transforms.Group(
                inputs=[
                    upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                    # upstream_transforms.ResizeImages(224, 224),
                    TokenizePrompt(
                        PaligemmaTokenizer(
                            model_config.max_token_len,
                            prompt_format=self.prompt_format,
                        ),
                        discrete_state_input=model_config.discrete_state_input,
                    ),
                    upstream_transforms.PadStatesAndActions(model_config.action_dim),
                ],
                outputs=[],
            )
        raise ValueError(f"Unsupported model type for ModelTransformFactory: {model_config.model_type}")


@dataclasses.dataclass(frozen=True)
class BaseDataConfigFactory(DataConfig, upstream_config.DataConfigFactory, abc.ABC):
    """Base class for all CoT data config factories.

    Provides common implementations for:
    - create_base_config: Extract CoT fields and set up base configuration
    - _load_norm_stats: Load normalization statistics from assets directory

    Subclasses must implement:
    - _create_data_transforms: Policy-specific data transformations
    - _create_model_transforms: Model-specific transformations
    """

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create base CoT config with common fields."""
        cot_fields = DataConfig.__dataclass_fields__.keys()
        data = {k: getattr(self, k) for k in cot_fields}
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        data.update(
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=None,  # Note: Normalization is handled on dataset level
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )
        return DataConfig(**data)

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, upstream_transforms.NormStats] | None:
        """Load normalization statistics from assets directory."""
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize_adapter.load(maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

    @abc.abstractmethod
    def _create_data_transforms(
        self, base_cfg: DataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        """Create policy-specific data transforms. Must be implemented by subclasses."""

    @abc.abstractmethod
    def _create_model_transforms(
        self, base_cfg: DataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        """Create model-specific transforms. Must be implemented by subclasses."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Template method that orchestrates config creation."""
        base_cfg = self.create_base_config(assets_dirs, model_config)
        data_transforms = self._create_data_transforms(base_cfg, model_config)
        model_transforms = self._create_model_transforms(base_cfg, model_config)

        return dataclasses.replace(
            base_cfg,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class PlanningDataConfig(BaseDataConfigFactory):
    """
    Config for training on planning dataset, using RLDS format loaded from TFDS.
    """

    has_time_dim: bool = False
    rlds_data_dir: str = "<your_data_dir>"

    @override
    def _create_data_transforms(
        self, base_cfg: DataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return upstream_transforms.Group(
            inputs=[
                planning_policy.PlanningInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    has_time_dim=self.has_time_dim,
                )
            ],
            outputs=[planning_policy.PlanningOutputs()],
        )

    @override
    def _create_model_transforms(
        self, base_cfg: DataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return ModelTransformFactory()(model_config)


@dataclasses.dataclass(frozen=True)
class TrainConfig(upstream_config.TrainConfig):
    # Overide
    project_name: str = "kinder-openpi"
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoaderChoice = dataclasses.field(
        default_factory=weight_loaders.WeightLoaderChoice
    )
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=build_cosine_lr)
    num_train_steps: int = 100_000
    save_interval: int = 500
    log_interval: int = 50
    keep_period: int | None = 10000
    resume: bool = True
    ema_decay: float | None = 0.999
    # New field
    checkpoint_async_timeout_secs: int | None = 7200
    checkpoint_async_enable: bool = True
    checkpoint_max_retries: int = 1
    checkpoint_retry_delay_secs: float = 30.0
    checkpoint_retry_backoff: float = 2.0
    checkpoint_fallback_to_sync: bool = True

    def _prefix_from_rlds(self) -> str | None:
        """Derive <prefix> from data.rlds_data_dir treated as <prefix>/data.

        Returns None when data is not a PlanningDataConfig or rlds_data_dir is
        the placeholder default value.
        """
        if not isinstance(self.data, PlanningDataConfig):
            return None
        rlds = self.data.rlds_data_dir
        if rlds == "<your_data_dir>":
            return None
        # Strip a trailing "/data" segment to get the prefix.
        rlds = rlds.rstrip("/")
        if rlds.endswith("/data"):
            return rlds[: -len("/data")]
        # If the path doesn't end in "data", use the parent directory.
        if rlds.startswith("gs://"):
            parts = rlds.split("/")
            return "/".join(parts[:-1]) if len(parts) > 3 else rlds
        return str(pathlib.Path(rlds).parent)

    @property
    @override
    def assets_dirs(self) -> pathlib.Path | epath.Path:
        """Assets directory (works for local paths and gs://…).

        When data.rlds_data_dir is set, assets_base_dir defaults to
        <prefix>/assets where <prefix> is the parent of the data directory.
        """
        prefix = self._prefix_from_rlds()
        base = f"{prefix}/assets" if prefix is not None else self.assets_base_dir
        return _to_path(base, self.name)

    @property
    @override
    def checkpoint_dir(self) -> pathlib.Path | epath.Path:
        """Checkpoint directory (local or Cloud Storage).

        When data.rlds_data_dir is set, checkpoint_base_dir defaults to
        <prefix>/checkpoints where <prefix> is the parent of the data directory.
        """
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        prefix = self._prefix_from_rlds()
        base = f"{prefix}/checkpoints" if prefix is not None else self.checkpoint_base_dir
        return _to_path(base, self.name, self.exp_name)


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="pi05_kinder_finetune",
        model=pi0_config.Pi0Config(
            action_horizon=10,
            max_token_len=180,
            pi05=True,
            discrete_state_input=True,
        ),
        data=PlanningDataConfig(
            repo_id="planning_dataset",
            asset_id="planning",
            shuffle_buffer_size=1_000_000,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
            rlds_data_dir="data",
        ),
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="gs://openpi-assets/checkpoints/pi05_base/params",
        ),
        fsdp_devices=8,
        batch_size=256,
        num_train_steps=50001,
        save_interval=5000,
        keep_period=5000,
        log_interval=100,
    ),
    *upstream_config._CONFIGS,  # noqa: SLF001
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name.

    Args:
        config_name: Name of the config to retrieve

    Returns:
        The requested TrainConfig

    Examples:
        get_config("lap")
    """
    if config_name in _CONFIGS_DICT:
        return _CONFIGS_DICT[config_name]

    # Config not found - provide helpful error message
    closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=3, cutoff=0.0)
    closest_str = f" Did you mean one of: {', '.join(repr(c) for c in closest)}?" if closest else ""

    raise ValueError(f"Config '{config_name}' not found.{closest_str}")
