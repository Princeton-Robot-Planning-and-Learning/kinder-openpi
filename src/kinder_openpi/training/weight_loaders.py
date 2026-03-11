import dataclasses
import logging
import pathlib
import re
from typing import Literal, Protocol, runtime_checkable

from flax import traverse_util
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.shared.array_typing as at
import orbax.checkpoint as ocp

import kinder_openpi.shared.download as download

logger = logging.getLogger(__name__)


def recover_dtype(a: np.ndarray) -> np.ndarray:
    """Numpy's `save` stores bfloat16 type as "void" type, so we recover it."""
    if hasattr(a, "dtype") and a.dtype.type is np.void:
        assert a.itemsize == 2, "Unknown dtype!"
        return a.view(jax.numpy.bfloat16)
    return a


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        # Preferred logic: use remote cache if present, otherwise mirror upstream into cache and load from there.
        params_path_str = str(self.params_path)

        if params_path_str.startswith("gs://"):
            # If this is already a cache path, try it; if missing or incomplete, fall back to upstream and mirror.
            if "/cache/" in params_path_str:
                cache_candidate = params_path_str
                upstream = params_path_str.split("/cache/", 1)[1]
                upstream = upstream if upstream.startswith("gs://") else f"gs://{upstream}"
                # Prefer existing cache; if present, ensure commit_success marker and use it.
                # Otherwise, mirror upstream into cache and use the mirror.
                try:
                    download.ensure_commit_success(cache_candidate)
                    params_source = cache_candidate
                except Exception:
                    params_source = str(download.maybe_download(upstream))
            else:
                # Not in cache yet; mirror upstream into cache to standardize layout.
                params_source = str(download.maybe_download(params_path_str))
        else:
            params_source = str(download.maybe_download(params_path_str))

        loaded_params = _model.restore_params(params_source, restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class PaliGemma2WeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma2 checkpoint."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(self.params_path, gs={"token": "anon"})
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        loaded_params = jax.tree.map(recover_dtype, loaded_params)
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=metadata,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), metadata
                ),
            ),
        )

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)


@dataclasses.dataclass(frozen=True)
class WeightLoaderChoice(WeightLoader):
    """CLI-friendly wrapper to choose a weight loader without nested subcommands.

    This class implements the WeightLoader protocol and forwards to a concrete
    loader based on the selected kind. It allows setting the loader type and its
    arguments via flat flags like:

      --weight-loader.kind=checkpoint --weight-loader.params-path=gs://...
      --weight-loader.kind=paligemma
      --weight-loader.kind=gemma3 --weight-loader.target-pos-emb-grid-size='(16,16)'
      --weight-loader.kind=pi05_action_expert --weight-loader.params-path=gs://base/checkpoint --weight-loader.pi05-params-path=gs://...
      --weight-loader.kind=pi05_base --weight-loader.pi05-params-path=gs://...
      --weight-loader.kind=paligemma_with_pi05_action_expert --weight-loader.params-path=gs://paligemma/checkpoint --weight-loader.pi05-params-path=gs://...
      --weight-loader.kind=none
    """

    # Which loader to use.
    kind: Literal[
        "none",
        "checkpoint",
        "paligemma",
        "paligemma2",
    ] = "paligemma"
    # Used when kind == "checkpoint", "paligemma2", "gemma3", "pi05_action_expert", or "paligemma_with_pi05_action_expert".
    params_path: str | None = None
    # Only used when kind == "gemma3" - target grid size for positional embeddings.
    target_pos_emb_grid_size: tuple[int, int] | None = None
    # Only used when kind == "pi05_action_expert", "pi05_base", or "paligemma_with_pi05_action_expert" - path to Pi0.5 checkpoint.
    pi05_params_path: str | None = "gs://openpi-assets/checkpoints/pi05_base/params"

    def _resolve(self) -> WeightLoader:
        match self.kind:
            case "checkpoint":
                if not self.params_path:
                    raise ValueError("--weight-loader.params-path must be set when kind=checkpoint")
                return CheckpointWeightLoader(self.params_path)
            case "paligemma":
                return PaliGemmaWeightLoader()
            case "paligemma2":
                if not self.params_path:
                    raise ValueError("--weight-loader.params-path must be set when kind=paligemma2")
                return PaliGemma2WeightLoader(self.params_path)
            case "none":
                return NoOpWeightLoader()
            case _:
                raise ValueError(f"Unknown weight loader kind: {self.kind}")

    def load(self, params: at.Params) -> at.Params:
        return self._resolve().load(params)


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
