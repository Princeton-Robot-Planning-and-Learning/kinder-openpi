import dataclasses
import datetime
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import numpy as np
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for remote environments
from openpi.models import model as _model
from openpi.models.model import Observation
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import optax
import psutil
from rail_tpu_utils import prevent_cross_region
import tqdm_loggable.auto as tqdm
import wandb

import kinder_openpi.dataloader.data_loader as _data_loader
import kinder_openpi.training.checkpoints as _checkpoints
import kinder_openpi.training.config as _config
import kinder_openpi.training.mh_sharding as sharding
import kinder_openpi.training.utils as training_utils
import kinder_openpi.training.weight_loaders as _weight_loaders


def log_mem(msg: str):
    """
    Returns:
        ram (float): Host RAM usage in GB (RSS).
        tpu_mem (float): TPU HBM memory usage in GB (sum across devices).
    """
    # --- Host RAM (Resident Set Size) ---
    proc = psutil.Process(os.getpid())
    ram_gb = proc.memory_info().rss / (1024**3)  # RSS in GB
    logging.info(f"{msg}: RAM: {ram_gb:.2f}GB")


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
    rewind_to_step: int | None = None,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    # Only initialize wandb in the main process
    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        if rewind_to_step is not None:
            # Use wandb's rewind feature to resume from a specific step
            wandb.init(
                resume_from=f"{run_id}?_step={rewind_to_step}",
                project=config.project_name,
            )
        else:
            wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def init_tpu(config: _config.TrainConfig):
    def _is_tpu_runtime() -> bool:
        try:
            return any(d.platform == "tpu" for d in jax.devices())
        except Exception:
            return False

    try:
        jax.distributed.initialize()
        logging.info("jax.distributed.initialize() succeeded")
    except Exception as e:
        logging.info("jax.distributed.initialize() skipped: %s", e)
    if "local" in config.name:
        os.environ["CURL_CA_BUNDLE"] = (
            "/etc/pki/tls/certs/ca-bundle.crt"  # Ensure the CA bundle is set for SSL verification
        )

    data_dir = save_dir = config.data.rlds_data_dir
    cache_dir = os.environ.get("OPENPI_DATA_HOME", None)
    if _is_tpu_runtime() and (str(data_dir).startswith("gs://") or str(save_dir).startswith("gs://")):
        prevent_cross_region(data_dir, save_dir)
        if cache_dir is not None:
            prevent_cross_region(cache_dir, save_dir)
    # Determine effective FSDP devices for single-process GPU/CPU runs.
    process_count = getattr(jax, "process_count", lambda: 1)()
    local_devices = getattr(jax, "local_device_count", lambda: 1)()
    global_devices = getattr(jax, "device_count", lambda: local_devices)()
    logging.info(f"Local devices: {local_devices}, Global devices: {global_devices}, Process count: {process_count}")
    if process_count == 1:
        # Choose the largest divisor of available devices not exceeding configured fsdp_devices
        target = min(config.fsdp_devices, local_devices)
        effective_fsdp_devices = 1
        for d in range(target, 0, -1):
            if global_devices % d == 0:
                effective_fsdp_devices = d
                break
        if effective_fsdp_devices != config.fsdp_devices:
            logging.info(
                "Using fsdp_devices=%d for single-process run (available devices=%d)",
                effective_fsdp_devices,
                global_devices,
            )
    else:
        effective_fsdp_devices = config.fsdp_devices

    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    return effective_fsdp_devices


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    del partial_params
    import gc

    gc.collect()

    return train_state, state_sharding


class TrainingStepRunner:
    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[Observation | Observation, _model.Actions],
    ) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
        model = nnx.merge(state.model_def, state.params)
        model.train()

        @at.typecheck
        def loss_fn(
            model: _model.BaseModel,
            rng: at.KeyArrayLike,
            observation: Observation | Observation,
            actions: _model.Actions,
        ):
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss)

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        diff_state = nnx.DiffState(0, self.config.trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

        params = state.params.filter(self.config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_params = nnx.state(model)

        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )

        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )

        info = {
            "loss": jnp.mean(loss),
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(kernel_params),
        }

        return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    effective_fsdp_devices = init_tpu(config)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # Human-readable mesh overview
    sharding.log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", sharding.format_sharding(data_sharding))
    logging.info("Replicated sharding spec: %s", sharding.format_sharding(replicated_sharding))

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        async_timeout_secs=config.checkpoint_async_timeout_secs,
        async_enable=config.checkpoint_async_enable,
    )
    init_wandb(
        config,
        resuming=resuming,
        enabled=config.wandb_enabled,
        rewind_to_step=getattr(config, "rewind_to_step", None),
    )

    # Log training start timestamp and preemption tracking info
    training_start_timestamp = datetime.datetime.now().isoformat()
    logging.info(f"Training started at: {training_start_timestamp}")
    logging.info(f"Resuming from checkpoint: {resuming}")

    log_mem("Before init train state")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)

    log_mem("After init train state")

    logging.info(f"Initialized train state (param shapes):\n{training_utils.array_tree_to_info(train_state.params)}")
    sharding.log_param_sharding_planned(train_state_sharding)
    sharding.log_param_sharding_actual(train_state.params)

    # Restore checkpoint BEFORE creating iterator to ensure dataloader state is restored correctly
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, None)
        logging.info("Successfully restored checkpoint and dataloader state")

    start_step = int(train_state.step)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
        persistent_iterator=True,
    )

    data_iter = iter(data_loader)
    log_mem("Before getting batch")
    batch = next(data_iter)
    log_mem("After getting batch")
    logging.info(f"Initialized data loader (shapes):\n{training_utils.array_tree_to_info(batch)}")

    # Log camera images from first batch to wandb
    if jax.process_index() == 0:
        obs = batch[0]
        wandb_images = {}
        for key, img in obs.images.items():
            sample_images = []
            for t in range(min(img.shape[0], 4)):
                sample_img = np.asarray(img[t])
                sample_img_uint8 = ((sample_img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                sample_images.append(wandb.Image(sample_img_uint8, caption=f"{key}_sample{t}"))
            if sample_images:
                wandb_images[f"batch_vis/{key}"] = sample_images
        if wandb_images:
            wandb.log(wandb_images, step=0)
            logging.info(f"Logged first-batch images to wandb: {list(wandb_images.keys())}")
    sharding.log_batch_sharding(batch)

    train_runner = TrainingStepRunner(config)
    ptrain_step = jax.jit(
        train_runner,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)

        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps:
            checkpoint_manager = _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                max_retries=config.checkpoint_max_retries,
                retry_delay_secs=config.checkpoint_retry_delay_secs,
                retry_backoff=config.checkpoint_retry_backoff,
                fallback_to_sync=config.checkpoint_fallback_to_sync,
                async_timeout_secs=config.checkpoint_async_timeout_secs,
                keep_period=config.keep_period,
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
