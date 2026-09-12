__copyright__ = """MIT License

Copyright (c) 2024 - IBM Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import logging
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Generic, Iterator, Literal, Sequence, TypeVar, cast
from uuid import uuid4

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from mblm.data.datasets import DistributedDataset
from mblm.data.types import ModelMode
from mblm.model.utils import count_params
from mblm.train.core.config import (
    CSVLossEntry,
    CSVTimeAndMemSnapshotEntry,
    GenericEntryConfig,
    GenericOutputConfig,
    ResumeMetadata,
    SummaryStats,
    TIoConfig,
    TModelParams,
    TTrainConfig,
)
from mblm.train.core.iter import epoch_cycler
from mblm.train.core.startup import STAGE_CHECKPOINT_SAVE, bootstrap_log, required_stage
from mblm.utils.cuda import IS_BF16_AVAILABLE, cuda_memory_snapshot, cuda_properties
from mblm.utils.distributed import ElasticRunVars
from mblm.utils.io import (
    CheckpointCursor,
    CSVWriter,
    StateDict,
    dump_yml,
    load_training_checkpoint_state,
    save_model_state,
    save_training_checkpoint_state,
)
from mblm.utils.logging import create_logger
from mblm.utils.top_n import TopN

FRESH = "fresh"
RESUME = "resume"

TModel = TypeVar("TModel", bound=torch.nn.Module)
TBatch = TypeVar("TBatch", bound=torch.Tensor | Sequence[torch.Tensor])


@dataclass
class CoreTrainerOptions:
    config_file_name: str = "config.yaml"
    loss_file_name: str = "loss.csv"
    timemem_file_name: str = "timemem.csv"
    skip_validation: bool = False
    display_progress: bool = sys.stdout.isatty()
    train_prog_min_interval_seconds: int = 1
    valid_prog_min_interval_seconds: int = 1
    track_first_fw_bw_exec_times: int | None = 30  # for 30 first passes, track fw/bw time
    amp_dtype: torch.dtype = torch.bfloat16 if IS_BF16_AVAILABLE else torch.half


class CoreTrainer(ABC, Generic[TModel, TBatch, TModelParams, TTrainConfig, TIoConfig]):
    """
    An abstract core trainer that provides a set of utility methods for
    training, evaluating and testing. It is held generic to enforce type-safety
    when implementing the abstract methods.

    - All methods that may or may not be implemented are `@classmethod`

    - Methods that _must_ be implemented are concerned with creating the
    model, which is the job of the instantiator. They are abstract and
    type-checkers complain if they are not implemented correctly

    - Methods that _may_ be implemented, i.e., overwritten, have the prefix
    `with_`. For example, for the optimizer, an `Adam` optimizer with sensible
    defaults is provided but it can be overwritten

    """

    # public config
    config: GenericEntryConfig[TModelParams, TTrainConfig, TIoConfig]

    # overridable options with sensible defaults
    options: CoreTrainerOptions

    # private var
    _local_rank: int
    _global_rank: int
    _world_size: int
    _is_main_writer: bool
    _device: str
    _device_type: Literal["cuda", "cpu"]
    _is_cuda: bool

    # the four components of a run, built by the startup stages
    _model: TModel
    _model_dist: TModel | DistributedDataParallel
    _optimizer: Optimizer
    _scheduler: LRScheduler
    _grad_scaler: torch.GradScaler

    # misc - attached once the run owns an output directory
    _output_dir: Path | None
    _resume_metadata: ResumeMetadata
    _running_summary_stats: SummaryStats
    _top_n_models: TopN[StateDict]
    _csv_loss_writer: CSVWriter[CSVLossEntry]
    _csv_timemem_writer: CSVWriter[CSVTimeAndMemSnapshotEntry]
    _log: logging.Logger
    _resume_cursor: CheckpointCursor | None
    _last_latest_checkpoint_step: int
    _last_latest_checkpoint_time: float

    def __init__(
        self,
        config: GenericEntryConfig[TModelParams, TTrainConfig, TIoConfig],
        run_vars: ElasticRunVars,
        options: CoreTrainerOptions | None = None,
    ):
        """
        Set up a trainer. Nothing is built and nothing is written here: the
        startup stages (`build_model`, `build_training_components`,
        `preflight_checkpoint`, `create_output_dir`, `initialize_outputs`) run
        after the process group is up, so that no rank creates an artefact, or
        enters a collective of its own, while another rank is still constructing.
        """
        self.config = config
        self.options = options or CoreTrainerOptions()
        self._resume_cursor = None
        self._last_latest_checkpoint_step = 0
        self._last_latest_checkpoint_time = time()
        self._world_size = run_vars.world_size
        self._local_rank = run_vars.local_rank
        # the global rank owns the shared artefacts; the local rank only picks
        # this process' device and data shard
        self._global_rank = run_vars.global_rank
        self._is_main_writer = run_vars.global_rank == 0
        # used for sending tensors/models to a device
        self._device = f"cuda:{self._local_rank}" if run_vars.is_cuda else "cpu"
        # used for mixed-precision
        self._device_type = "cuda" if run_vars.is_cuda else "cpu"
        self._is_cuda = not self._device == "cpu"

        self._output_dir = None
        self._log = bootstrap_log()
        self._top_n_models = TopN(
            config.io.num_models_to_save,
            deep_copy=True,  # module state_dicts are references
        )

    """ Startup stages """

    def build_model(self) -> None:
        """
        Build the raw model for this rank and move it to its device.
        """
        self._model = self.init_model().to(self._device)

    def local_batch_iters(self) -> int:
        """
        The number of micro-batches this rank trains on, derived from the
        resolved training config and the world size. A data loader is never the
        source of this number.
        """
        train_conf = self.config.train
        local_target_elements = train_conf.target_elements // self._world_size
        if train_conf.target_elements_strategy == "batch":
            elements_per_batch = train_conf.batch_size
        else:
            elements_per_batch = train_conf.batch_size * self.config.params.input_seq_len
        return math.ceil(local_target_elements / elements_per_batch)

    def local_gradient_steps(self) -> int:
        """
        The number of optimizer steps this rank takes over the whole run.
        """
        return self.local_batch_iters() // self.config.train.gradient_accumulate_every

    def build_training_components(self) -> None:
        """
        Wrap the model for distributed training and build the optimizer,
        scheduler and gradient scaler around it.
        """
        self._validate_config()
        self._model_dist = self._init_distributed_model(self._model)
        self._optimizer = self.configure_optimizer(self._model_dist.parameters())
        self._scheduler = self.configure_scheduler(self._optimizer, self.local_gradient_steps())
        self._grad_scaler = torch.GradScaler(device=self._device_type)

    def preflight_checkpoint(self) -> str:
        """
        Decide whether this run starts from scratch or restores an explicitly
        configured checkpoint, and restore it into the already built model,
        optimizer, scheduler and scaler.

        A configured checkpoint must load completely: there is no partial
        restore and no fallback to a fresh run. Restoring is decided solely by
        the presence of `resume.checkpoint_file`; the run never looks for a
        checkpoint on its own.

        Returns:
            str: `fresh` or `resume`
        """
        if self.config.resume is None:
            self._log.info("Creating new model")
            return FRESH

        checkpoint_file = self.config.resume.checkpoint_file
        self._log.info(f"Initiating model loading from checkpoint {checkpoint_file}")
        cursor, model_loss = load_training_checkpoint_state(
            checkpoint_file,
            self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            grad_scaler=self._grad_scaler,
            gradient_accumulate_every=self.config.train.gradient_accumulate_every,
            map_location="cpu",
            optimizer_device=self._device,
        )
        # the checkpoint carries the training position in full training state
        self._resume_cursor = cursor
        if model_loss is None:
            self._log.warning(
                f"Checkpoint {checkpoint_file} carries no loss, so nothing is added to "
                "the model candidate board from it"
            )
        else:
            # keep the restored model as a candidate so a run that only makes
            # things worse does not lose the model it started from
            self._top_n_models.add((model_loss, self._model.state_dict()))
            self._log.info(f"Loaded model with loss {model_loss:.4f} from checkpoint")
        self._log.info(
            f"Resuming from epoch {cursor.epoch}, batch {cursor.batch}, "
            f"cumulative micro-batch {cursor.cum_batch}"
        )
        return RESUME

    def create_output_dir(self) -> str | None:
        """
        Create this run's unique output directory. Only the global rank 0
        writer creates it; every other rank returns `None` and receives the
        created path from the stage's unified outcome.
        """
        if not self._is_main_writer:
            return None
        run_id = self.configure_run_id()
        output_dir = Path(self.config.io.output_dir) / f"{self.config.io.name_model}_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=False)
        return str(output_dir)

    def initialize_outputs(self, output_dir: str) -> None:
        """
        Attach this run's output artefacts to every rank. Only the global rank 0
        writer creates or opens a file: every other rank gets a noop logger and
        noop CSV writers.
        """
        self._output_dir = Path(output_dir)
        self._log = self.configure_logger(self._output_dir, self._is_main_writer)
        self._csv_loss_writer = CSVWriter(
            self._output_dir, self.options.loss_file_name, noop=not self._is_main_writer
        )
        self._csv_timemem_writer = CSVWriter(
            self._output_dir, self.options.timemem_file_name, noop=not self._is_main_writer
        )

        # the output config only records provenance; the cursor stays in the
        # checkpoint and is never written back to a config file
        self._resume_metadata = ResumeMetadata(
            parent_checkpoint=self.config.resume.checkpoint_file if self.config.resume else None,
        )
        cuda_info = cuda_properties()
        main_model_params, submodule_params = self.configure_count_parameters(self._model)

        self._running_summary_stats = SummaryStats(
            parameter_count=main_model_params,
            num_workers=self._world_size,
            cuda_devices=cuda_info.cuda_devices,
            training_start="",  # temporary, updated when training starts
            training_end="",  # temporary, updated when training ends
            error=None,  # temporary, updated when training fails
        )
        self._dump_output_config()
        self._log.info("Trainer initialized successfully")
        self._log.info(f"Model parameters: {main_model_params}, ({submodule_params})")
        self._log.info(f"Configuration: {self.config}")
        self._log.info(f"CUDA: {cuda_info}")

    def share_test_model(self, model: TModel) -> TModel:
        """
        Distribute the state chosen for testing from the global rank 0 writer to
        every rank, so that all ranks evaluate the same weights.

        Choosing which state to test belongs to the caller; this only moves the
        chosen state across ranks.
        """
        if self._world_size == 1:
            return model
        for tensor in model.state_dict().values():
            dist.broadcast(tensor.data, src=0)
        return model

    def _validate_config(self) -> None:
        assert self.config.io.validate_amount > 0, "Validate amount must be strictly positive"
        assert self.config.io.num_models_to_save >= 0, "num_models_to_save cant be negative"
        if self.config.io.num_models_to_save == 0:
            self._log.warning("No model of this training will be saved!")

        if self.config.io.validate_amount < self.config.io.num_models_to_save:
            self._log.warning(
                f"Validate amount ({self.config.io.validate_amount}) \
                is less than number of models to save ({self.config.io.num_models_to_save}).\
                Saving only {self.config.io.validate_amount} models"
            )

        accumulation = self.config.train.gradient_accumulate_every
        batch_iters = self.local_batch_iters()
        if batch_iters < accumulation:
            raise ValueError(
                f"the run's {batch_iters} micro-batches are fewer than "
                f"gradient_accumulate_every ({accumulation}): no optimizer step would be taken"
            )
        if batch_iters % accumulation != 0:
            raise ValueError(
                f"the run's {batch_iters} micro-batches are not a multiple of "
                f"gradient_accumulate_every ({accumulation}): it would end with a "
                "pending accumulation window"
            )

    """ Abstract methods that must be implemented """

    @abstractmethod
    def init_model(self) -> TModel:
        """
        Initialize a model of the specified type `TModel`.
        """
        ...

    @abstractmethod
    def model_forward(
        self,
        model: TModel,
        batch: TBatch,
        device: str,
    ) -> torch.Tensor:
        """
        A single forward pass of the model. Both the model and batch are
        generic, their types are inferred according to the type instantiation
        defined when subclassing `CoreTrainer`.

        Args:
            model (TModel): The model (already on the device)
            batch (TBatch): One batch (MUST be put to device)
            device (str): `cuda:n` for the `n`-th GPU or `cpu`

        Returns:
            torch.Tensor: A Tensor with a single element that is the loss
            of the forward pass

        **Example**::

            # here, batch is a tuple of data, target
            @classmethod
            def model_forward(cls, model, batch, device):
                x, y = batch
                output = model.forward(x.to(device))
                loss_function = torch.nn.MSELoss()
                loss: torch.Tensor = loss_function(output, y)
                return loss


            # in other scenarios, batch might be a single Tensor
            @classmethod
            def model_forward(cls, model, batch, device):
                batch = batch.to(device).long()
                loss: torch.Tensor = model.forward(batch, return_loss=True)
                return loss
        """

        ...

    @abstractmethod
    def configure_optimizer(self, parameters: Iterator[torch.nn.Parameter]) -> Optimizer:
        """
        Configure an optimizer
        """
        ...

    """ Default methods that can be overwritten """

    def configure_scheduler(self, optimizer: Optimizer, local_gradient_steps: int) -> LRScheduler:
        """
        Configure a LR scheduler.

        Args:
            optimizer (Optimizer): The optimizer
            local_gradient_steps (int): The total number of gradient steps for this GPU.
        """
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=local_gradient_steps,
            power=1.0,
        )

    def configure_logger(self, output_dir: Path, is_main_worker: bool) -> logging.Logger:
        """
        Customize the logger
        """
        return create_logger(
            name="train",
            log_dir=output_dir,
            # all non-main workers are noop loggers
            noop=not is_main_worker,
        )

    def configure_count_parameters(self, model: TModel) -> tuple[int, dict[str, int]]:
        """
        Determine how to count parameters for this model
        """
        return count_params(model)

    def configure_run_id(self) -> str:
        """
        Set a unique identifier for this experiment. Used as postfix for the
        output directory. A run id is never shared: two runs started in the same
        second, or pointed at the same parent directory, still get their own.
        """
        return f"{time():.6f}-{uuid4().hex[:8]}"

    @property
    def output_dir(self) -> Path:
        """
        The directory this run writes its artefacts to.
        """
        assert self._output_dir is not None, "the run has no output directory yet"
        return self._output_dir

    """ Utility functions  """

    def _init_distributed_model(self, base_model: TModel) -> DistributedDataParallel:
        """
        Create a distributed version of the model.
        """

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        # for multi-device modules and CPU modules, device_ids must be None
        device_ids = [self._local_rank] if self._is_cuda else None
        return DistributedDataParallel(model, device_ids=device_ids)

    def _unpack_distributed_model(self, module: TModel | DistributedDataParallel) -> TModel:
        if isinstance(module, DistributedDataParallel):
            return module.module
        return module

    def _dump_output_config(self):
        """
        Dump all config files to disk (only on main worker).
        """
        if not self._is_main_writer:
            return
        # copy the config over into the output format
        output_config = GenericOutputConfig(
            io=self.config.io,
            params=self.config.params,
            train=self.config.train,
            resume=self._resume_metadata,
            summary=self._running_summary_stats,
        )
        dump_yml(self.output_dir / self.options.config_file_name, output_config)

    def _write_csv_loss(
        self,
        kind: ModelMode,
        loss: float,
        epoch: int,
        batch: int,
        cum_batch: int,
        elements_seen: int,
        lr: float,
        avg_grad: float,
        avg_grad_clipped: float,
    ) -> None:
        # no need to check for main worker - the writer has been initialized
        # before so that only the main worker performs io
        row = CSVLossEntry(
            gpu_rank=self._local_rank,
            timestamp=str(datetime.now()),
            kind=kind.value,
            elements_seen=elements_seen,
            epoch=epoch,
            batch=batch,
            cum_batch=cum_batch,
            loss=loss,
            lr=lr,
            avg_grad=avg_grad,
            avg_grad_clipped=avg_grad_clipped,
        )
        self._csv_loss_writer.write_row(row)

    def _write_csv_timemem(
        self, cum_batch: int, num_items, kind: ModelMode, fw_time: float, bw_time: float | None
    ):
        mem_snapshot = cuda_memory_snapshot(self._device)
        row = CSVTimeAndMemSnapshotEntry(
            kind=kind.value,
            num_items=num_items,
            cum_batch=cum_batch,
            fw_time=fw_time,
            bw_time=bw_time,
            allocated=mem_snapshot.allocated,
            allocated_max=mem_snapshot.allocated_max,
            reserved=mem_snapshot.reserved,
            reserved_max=mem_snapshot.reserved_max,
            total=mem_snapshot.total,
        )
        self._csv_timemem_writer.write_row(row)

    def _save_best_models(self) -> tuple[int, int, Path]:
        num_written = 0
        num_overwritten = 0
        best_checkpoint = Path()
        if not self._is_main_writer:
            return num_written, num_overwritten, best_checkpoint

        # save final n best models - best models are iterated first
        for idx, (loss, model_state) in enumerate(self._top_n_models):
            did_overwrite, checkpoint_path = save_model_state(
                self.output_dir,
                f"{self.config.io.name_model}_top{idx + 1}.pth",
                model=model_state,
                loss=loss,
            )
            num_written += 1
            if idx == 0:
                best_checkpoint = checkpoint_path
            if did_overwrite:
                num_overwritten += 1
        return num_written, num_overwritten, best_checkpoint

    def _log_cuda_memory_snapshot(self, cumulative_batch_idx: int | None) -> None:
        if self._is_cuda:
            snapshot = cuda_memory_snapshot(self._device)
            prefix = f"[{cumulative_batch_idx}] " if cumulative_batch_idx else ""
            self._log.debug(f"{prefix}CUDA memory: {snapshot}")

    def _calc_logging_points(self, total_batch_iters: int) -> tuple[set[int], set[int]]:
        """
        Calculate the cumulative micro-batch counts to log the training loss and
        run validation at. The counts number the completed micro-batches of the
        run, so the first micro-batch of a run sits at 1 and a run that has
        already completed `n` micro-batches never revisits a point below `n + 1`.
        """
        log_train_loss_amount = self.config.io.log_train_loss_amount
        if total_batch_iters < log_train_loss_amount:
            self._log.warning(
                f"Less batch iterations ({total_batch_iters}) "
                f"than number of train loss log points ({log_train_loss_amount}). "
                f"Clipping train loss log points to {total_batch_iters}"
            )
            log_train_loss_amount = total_batch_iters

        validate_amount = self.config.io.validate_amount
        if total_batch_iters < validate_amount:
            self._log.warning(
                f"Less batch iterations ({total_batch_iters}) "
                f"than number of validation runs ({validate_amount}). "
                f"Clipping validation points to {total_batch_iters}"
            )
            validate_amount = total_batch_iters
        log_train_loss_idxs = set(
            torch.linspace(1, total_batch_iters, log_train_loss_amount).long().tolist()
        )
        run_valid_interval_idxs = set(
            torch.linspace(1, total_batch_iters, validate_amount).long().tolist()
        )

        return log_train_loss_idxs, run_valid_interval_idxs

    def _confirm_checkpoint_safe_point(self, cum_batch: int) -> None:
        """
        Confirm over the process group that every rank reached the same
        checkpoint-safe point: the same cumulative micro-batch count, on an
        accumulation boundary and with no pending gradient. A checkpoint may
        only be written once all ranks agree.
        """
        accumulation = self.config.train.gradient_accumulate_every
        with required_stage(STAGE_CHECKPOINT_SAVE, world_size=self._world_size) as report:
            report.state = str(cum_batch)
            if cum_batch % accumulation != 0:
                raise ValueError(
                    f"cumulative micro-batch {cum_batch} is not an accumulation boundary "
                    f"(gradient_accumulate_every={accumulation})"
                )
            if not self._gradients_are_cleared():
                raise ValueError(f"cumulative micro-batch {cum_batch} has pending gradients")

    def _gradients_are_cleared(self) -> bool:
        return all(parameter.grad is None for parameter in self._model_dist.parameters())

    def _save_latest_checkpoint(
        self,
        *,
        loss: float,
        next_batch_idx: int,
        next_epoch: int,
        cum_batch: int,
    ) -> None:
        if not self.config.train.latest_checkpoint_enabled:
            return
        self._confirm_checkpoint_safe_point(cum_batch)
        if not self._is_main_writer:
            return

        original_model = self._unpack_distributed_model(self._model_dist)
        save_training_checkpoint_state(
            self.output_dir,
            self.config.train.latest_checkpoint_name,
            model=original_model,
            loss=loss,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            grad_scaler=self._grad_scaler,
            epoch=next_epoch,
            batch=next_batch_idx,
            cum_batch=cum_batch,
        )

        self._log.debug(
            f"Saved latest training state at epoch {next_epoch}, batch {next_batch_idx}"
        )

    def _maybe_save_latest_checkpoint(
        self,
        *,
        loss: float,
        next_batch_idx: int,
        next_epoch: int,
        cum_batch: int,
        completed_optimizer_steps: int,
    ) -> None:
        if not self.config.train.latest_checkpoint_enabled:
            return

        now = time()
        steps_elapsed = completed_optimizer_steps - self._last_latest_checkpoint_step
        seconds_elapsed = now - self._last_latest_checkpoint_time
        due_by_steps = steps_elapsed >= self.config.train.latest_checkpoint_interval_steps
        due_by_time = (
            self.config.train.latest_checkpoint_interval_seconds is not None
            and seconds_elapsed >= self.config.train.latest_checkpoint_interval_seconds
        )
        if not due_by_steps and not due_by_time:
            return

        self._save_latest_checkpoint(
            loss=loss,
            next_batch_idx=next_batch_idx,
            next_epoch=next_epoch,
            cum_batch=cum_batch,
        )
        self._last_latest_checkpoint_step = completed_optimizer_steps
        self._last_latest_checkpoint_time = now

    def _save_training_state(self, batch_i: int, epoch: int):
        if not self._is_main_writer:
            return
        num_written, num_overwritten, _ = self._save_best_models()
        self._log.debug(
            f"Saved {num_written} best model(s) (overwrote {num_overwritten})",
        )

        self._dump_output_config()
        self._log.debug(f"Saved training state at epoch {epoch}, batch {batch_i}")

    def avg_gradient_value(self) -> float:
        gradients = [
            p.grad.mean().item() for p in self._model_dist.parameters() if p.grad is not None
        ]
        return sum(gradients) / len(gradients)

    """ Training and evaluation """

    def _evaluate(
        self,
        model: torch.nn.Module,
        loader: DataLoader[TBatch],
        items_seen_so_far: int,
        cumulative_batch_idx: int,
    ) -> float:
        """
        Evaluate any model on any dataset.
        """
        model.eval()
        loss = 0.0
        time_taken = 0.0
        target_iters = (
            min(self.config.train.max_eval_steps, len(loader))
            if self.config.train.max_eval_steps
            else len(loader)
        )
        for it, batch in enumerate(
            tqdm(
                loader,
                total=target_iters,
                desc="Evaluating",
                leave=False,
                disable=not self.options.display_progress,
                mininterval=self.options.valid_prog_min_interval_seconds,
            )
        ):
            if it == target_iters:
                break
            with torch.autocast(
                device_type=self._device_type,
                dtype=self.options.amp_dtype,
            ):
                with torch.inference_mode():
                    start_eval = time()
                    loss_tensor = self.model_forward(
                        cast(TModel, model),
                        batch=batch,
                        device=self._device,
                    )
                    eval_time = time() - start_eval
                    loss += float(loss_tensor.item())
                    # for the eval dataloader, we don't drop the last batch,
                    # hence, the last batch might have a lower batch size.
                    # therefore, count manually to report accurate times per
                    # element
                    time_taken += eval_time

        if self.options.track_first_fw_bw_exec_times:
            self._write_csv_timemem(
                cum_batch=cumulative_batch_idx,
                kind=ModelMode.VALID,
                fw_time=time_taken,
                bw_time=None,
                num_items=items_seen_so_far,
            )
        return loss / target_iters

    def train(
        self,
        train_dataset: DistributedDataset[TBatch],
        valid_dataset: DistributedDataset[TBatch],
    ) -> TModel | None:
        self._running_summary_stats.training_start = datetime.now().isoformat()
        best_model = self._train(train_dataset, valid_dataset)

        self._running_summary_stats.training_end = datetime.now().isoformat()
        self._log_cuda_memory_snapshot(-99)
        self._dump_output_config()
        return best_model

    def _get_dataloader(
        self,
        dataset: DistributedDataset[TBatch],
        data_loader_kwargs: dict[str, Any],
        **additional_data_loader_kwargs: dict[str, Any],
    ) -> DataLoader:
        """Generic data loader instantiation.

        Args:
            dataset: a distributed dataset object.
            data_loader_kwargs: additional arguments for the data loader.

        Returns:
            a data loader.
        """
        return DataLoader(dataset, **{**data_loader_kwargs, **additional_data_loader_kwargs})

    def get_train_dataloader(
        self, dataset: DistributedDataset[TBatch], **additional_data_loader_kwargs: dict[str, Any]
    ) -> DataLoader:
        """Train data loader instantiation.

        Args:
            dataset: a distributed dataset object.
            data_loader_kwargs: additional arguments for the data loader.

        Returns:
            the train data loader.
        """
        return self._get_dataloader(
            dataset=dataset,
            data_loader_kwargs=dict(
                batch_size=self.config.train.batch_size,
                pin_memory=True,
                shuffle=self.config.train.shuffle_train,  # False by default
                # drop the last batch so all batches have the same num of elements
                drop_last=True,
                # no need for a distributed sampler, the dataset is already distributed
                sampler=None,
            ),
            **additional_data_loader_kwargs,
        )

    def get_valid_dataloader(
        self, dataset: DistributedDataset[TBatch], **additional_data_loader_kwargs: dict[str, Any]
    ) -> DataLoader:
        """Validation data loader instantiation.

        Args:
            dataset: a distributed dataset object.
            data_loader_kwargs: additional arguments for the data loader.

        Returns:
            the validation data loader.
        """
        return self._get_dataloader(
            dataset=dataset,
            data_loader_kwargs=dict(
                pin_memory=True,
                shuffle=self.config.train.shuffle_eval,  # False by default
                drop_last=False,
                batch_size=self.config.train.batch_size,
            ),
            **additional_data_loader_kwargs,
        )

    def get_test_dataloader(
        self, dataset: DistributedDataset[TBatch], **additional_data_loader_kwargs: dict[str, Any]
    ) -> DataLoader:
        """Test data loader instantiation.

        Args:
            dataset: a distributed dataset object.
            data_loader_kwargs: additional arguments for the data loader.

        Returns:
            the test data loader.
        """
        return self._get_dataloader(
            dataset=dataset,
            data_loader_kwargs=dict(
                pin_memory=True,
                shuffle=False,
                batch_size=self.config.train.batch_size,
            ),
            **additional_data_loader_kwargs,
        )

    def _train(
        self,
        train_dataset: DistributedDataset[TBatch],
        valid_dataset: DistributedDataset[TBatch],
    ) -> TModel:
        """
        Train a model on a training dataset and occasionally run it on the
        validation set.
        """

        if self._is_cuda:
            torch.cuda.empty_cache()

        train_conf = self.config.train

        # instantiate train and validation data loaders, currently
        # no additional arguments forwarded to instantiation.
        train_loader = self.get_train_dataloader(train_dataset)
        valid_loader = self.get_valid_dataloader(valid_dataset)

        # calculate the number of data elements this worker should train on
        # based on the number of (global) target elements and number of workers
        # (cpus/gpus) available. because we use a distributed sampler, the local
        # test data loader only sees 1/world_size of the training data already
        global_target_elements = train_conf.target_elements
        local_target_elements = global_target_elements // self._world_size

        # by setting drop_last=True in the train loader, we make sure all batches have
        # the same number of elements
        if self.config.train.target_elements_strategy == "batch":
            elements_per_batch = self.config.train.batch_size
        else:
            elements_per_batch = self.config.train.batch_size * self.config.params.input_seq_len

        # because global target elements is a lower bound - we always want to
        # train on at least this number of elements - we may train on one more
        # batch (due to batch sizes and sequence lengths). in order to reach the
        # lower bound, the actual number of elements trained per worker may be
        # slightly higher.
        local_batch_iters = self.local_batch_iters()
        expected_local_elements = elements_per_batch * local_batch_iters
        expected_global_elements = expected_local_elements * self._world_size
        delta = expected_global_elements - global_target_elements

        self._log.debug(f"Global target elements: {global_target_elements}")
        self._log.debug(
            f"Local target elements: {local_target_elements} ({self._world_size} workers)"
        )
        self._log.debug(f"Elements per batch: {elements_per_batch} (bs: {train_conf.batch_size}) ")
        self._log.debug(
            f"Expected global target elements (w.r.t batch size): {expected_global_elements}"
        )
        self._log.debug(
            f"Expected local target elements (w.r.t batch size): {expected_local_elements}"
        )

        self._log.debug(f"Target element delta (global): {delta} elements")
        self._log.info(f"Running {local_batch_iters} batch iterations")

        epoch: int = 0
        epoch_batch_idx: int = 0
        # the cursor counts the micro-batches this rank has completed, and is the
        # coordinate the run logs, validates and saves at. it is read from the
        # checkpoint and never recomputed from a loader length.
        cum_batch: int = 0
        if self._resume_cursor is not None:
            self._log.debug("Resuming training, offsetting start epoch and batch index")
            epoch = self._resume_cursor.epoch
            epoch_batch_idx = self._resume_cursor.batch
            cum_batch = self._resume_cursor.cum_batch
            train_dataset.offset_to(epoch)
        else:
            self._log.info("Starting training from scratch")
        self._log.debug(f"Starting from epoch {epoch}")
        self._log.debug(f"Starting from batch {epoch_batch_idx}")
        self._log.debug(f"Starting from cumulative micro-batch {cum_batch}")
        self._log_cuda_memory_snapshot(None)

        start_cum_batch = cum_batch
        if start_cum_batch > local_batch_iters:
            self._log.warning(
                f"Resume position {start_cum_batch} exceeds target "
                f"iterations {local_batch_iters}; no further training will run"
            )
        remaining_batch_iters = max(local_batch_iters - start_cum_batch, 0)
        self._log.info(f"Remaining batch iterations: {remaining_batch_iters}")
        log_train_idxs, run_valid_idxs = self._calc_logging_points(local_batch_iters)
        log_train_idxs = {idx for idx in log_train_idxs if idx > start_cum_batch}
        run_valid_idxs = {idx for idx in run_valid_idxs if idx > start_cum_batch}

        def before_new_epoch(epoch: int) -> None:
            self._log.info(f"Initializing epoch {epoch}")
            train_dataset.offset_to(epoch)

        # total elements seen during trainings
        elements_seen_total = elements_per_batch * start_cum_batch
        curr_avg_grad: float = -1
        curr_avg_grad_clipped: float = -1
        completed_optimizer_steps = start_cum_batch // train_conf.gradient_accumulate_every
        self._last_latest_checkpoint_step = completed_optimizer_steps
        self._last_latest_checkpoint_time = time()
        last_successful_optimizer_step: tuple[int, int, int, float] | None = None
        for iteration in tqdm(
            epoch_cycler(
                train_loader,
                before_new_epoch=before_new_epoch,
                start_epoch=epoch,
                start_batch=epoch_batch_idx,
                max_iters=remaining_batch_iters,
            ),
            desc="Training",
            mininterval=self.options.train_prog_min_interval_seconds,
            disable=not self.options.display_progress,
        ):
            batch: TBatch
            next_epoch: int
            next_batch_idx: int

            epoch, epoch_batch_idx, batch = iteration.epoch, iteration.batch, iteration.item
            next_epoch, next_batch_idx = iteration.next_epoch, iteration.next_batch

            self._model_dist.train()

            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
            with torch.autocast(
                device_type=self._device_type,
                dtype=self.options.amp_dtype,
            ):
                start_fw_measure = time()
                train_loss = self.model_forward(
                    # warning - we cast so that the arguments type hints for
                    # TModel.forward() are preserved, however, because the model
                    # has been wrapped with DistributedDataParallel, other
                    # methods might not be available. Use only model.forward()
                    cast(TModel, self._model_dist),
                    batch=batch,
                    device=self._device,
                )
                fw_exec_time = time() - start_fw_measure
                train_loss_as_flt = float(train_loss.item())
                if math.isnan(train_loss_as_flt):
                    shape = batch.shape if isinstance(batch, torch.Tensor) else batch[0].shape
                    self._log.error(
                        f"Invalid loss at batch {epoch_batch_idx}, epoch {epoch}. Train loss (raw): {train_loss}, batch: {shape}"
                    )

                # scale the gradient
                train_loss = train_loss / train_conf.gradient_accumulate_every
            elements_seen_total += elements_per_batch

            scaled_loss = self._grad_scaler.scale(train_loss)

            start_bw_measure = time()
            scaled_loss.backward()
            bw_exec_time = time() - start_bw_measure

            # this micro-batch is complete: the cursor moves before anything
            # reads it, so every point below is decided on a completed count
            cum_batch += 1
            log_prefix = f"{[cum_batch]}"

            # if enabled, report forward/backward pass execution times for the
            # first track_first_fw_bw_exec_times iterations as well as cuda
            # memory usage. after a few iterations, we can usually be sure
            # memory will not further increase assuming there are no memory
            # leaks. skip the first forward/backward pass, which takes much more
            # time due to the construction of the computation graph, optimizer
            # warmup, etc.

            if self.options.track_first_fw_bw_exec_times:
                self.options.track_first_fw_bw_exec_times -= 1
                self._write_csv_timemem(
                    cum_batch=cum_batch,
                    kind=ModelMode.TRAIN,
                    fw_time=fw_exec_time,
                    bw_time=bw_exec_time,
                    # batch size is constant during training
                    num_items=self.config.train.batch_size,
                )

            if cum_batch in log_train_idxs:
                self._log.info(f"{log_prefix} Training loss: {train_loss_as_flt}")
                self._write_csv_loss(
                    ModelMode.TRAIN,
                    loss=train_loss_as_flt,
                    epoch=epoch,
                    batch=epoch_batch_idx,
                    cum_batch=cum_batch,
                    elements_seen=elements_seen_total,
                    lr=self._scheduler.get_last_lr()[0],
                    avg_grad=curr_avg_grad,
                    avg_grad_clipped=curr_avg_grad_clipped,
                )

            # accumulate the gradient with clipping:
            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            if cum_batch % train_conf.gradient_accumulate_every == 0:
                # restore the scaled gradient for clipping
                self._grad_scaler.unscale_(self._optimizer)

                curr_avg_grad = self.avg_gradient_value()

                if (max_clip := self.config.train.gradient_clipping) is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model_dist.parameters(),
                        max_clip,
                    )

                curr_avg_grad_clipped = self.avg_gradient_value()

                self._grad_scaler.step(self._optimizer)
                scale = self._grad_scaler.get_scale()
                self._grad_scaler.update()

                # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/7
                skip_lr_sched = scale > self._grad_scaler.get_scale()
                if self._scheduler and not skip_lr_sched:
                    self._scheduler.step()
                self._optimizer.zero_grad(set_to_none=True)
                if not skip_lr_sched:
                    completed_optimizer_steps += 1
                    # the cursor already points at the next micro-batch to run
                    last_successful_optimizer_step = (
                        next_epoch,
                        next_batch_idx,
                        cum_batch,
                        train_loss_as_flt,
                    )
                    self._maybe_save_latest_checkpoint(
                        loss=train_loss_as_flt,
                        next_batch_idx=next_batch_idx,
                        next_epoch=next_epoch,
                        cum_batch=cum_batch,
                        completed_optimizer_steps=completed_optimizer_steps,
                    )

            # before evaluation, we do not perform a gradient update. hence, the
            # "elements_seen_total" we log below might be slightly off.
            # specifically, this number might be larger than the true number.
            # this is because the validation might be performed while gradients
            # are still being accumulated, and thus have the model has not
            # learned from the elements yet. on a large scale, this hardly
            # matters
            if not self.options.skip_validation and cum_batch in run_valid_idxs:
                valid_loss = self._evaluate(
                    self._model_dist,
                    valid_loader,
                    items_seen_so_far=elements_seen_total,
                    cumulative_batch_idx=cum_batch,
                )
                self._log.info(f"{log_prefix} Validation loss: {valid_loss}")
                self._write_csv_loss(
                    ModelMode.VALID,
                    loss=valid_loss,
                    epoch=epoch,
                    batch=epoch_batch_idx,
                    cum_batch=cum_batch,
                    elements_seen=elements_seen_total,
                    lr=-1,
                    avg_grad=-1,
                    avg_grad_clipped=-1,
                )

                # after validating, save the state, maybe it's really good!
                # before i/o, use a barrier to make sure training states are in
                # sync (as seen in https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html )
                dist.barrier()
                original_model = self._unpack_distributed_model(self._model_dist)
                self._top_n_models.add(
                    (
                        valid_loss,
                        original_model.state_dict(),
                    )
                )
                self._save_training_state(next_batch_idx, next_epoch)

        else:
            # we have seen exactly local_batch_iters batches
            elements_match = (
                elements_seen_total == expected_local_elements
                or start_cum_batch >= local_batch_iters
            )
            if not elements_match:
                self._log.fatal(
                    f"Mismatch between expected and actual elements seen: {expected_local_elements}, {elements_seen_total}"
                )
            if completed_optimizer_steps == 0 and start_cum_batch < local_batch_iters:
                # a run that ends without a single optimizer step has trained
                # nothing; only a resume cursor that already reached the target
                # is a legitimate zero-step run
                self._log.fatal(
                    "Finished training without a single optimizer step: "
                    f"{local_batch_iters} batch iterations, gradient accumulation of "
                    f"{train_conf.gradient_accumulate_every}"
                )
                raise RuntimeError(
                    "the run finished without a single optimizer step: "
                    f"batch_iterations={local_batch_iters}, "
                    f"gradient_accumulate_every={train_conf.gradient_accumulate_every}"
                )
            self._log.info("Finished training")
            self._log.info(f"Stats (local): Elements seen: {elements_seen_total}")
            if last_successful_optimizer_step:
                next_epoch, next_batch_idx, save_cum_batch, loss = last_successful_optimizer_step
                self._save_latest_checkpoint(
                    loss=loss,
                    next_batch_idx=next_batch_idx,
                    next_epoch=next_epoch,
                    cum_batch=save_cum_batch,
                )

        best_model = self._unpack_distributed_model(self._model_dist)

        if self._is_main_writer and self.config.io.num_models_to_save > 0:
            # the writer returns the model candidate it holds; the other ranks
            # return their latest model, which only becomes the tested one once
            # the chosen state is distributed to them
            ((least_loss, best_state),) = self._top_n_models.get_top(1)
            best_model.load_state_dict(best_state)
            self._log.info(f"Returning model with least loss ({least_loss})")

        self._log_cuda_memory_snapshot(None)

        return best_model

    def test(
        self,
        test_dataset: DistributedDataset[TBatch],
        model: torch.nn.Module,
    ) -> None:
        """
        Evaluate a model on the test set. Every rank takes part: the model has
        been distributed beforehand, and the writers among the ranks record the
        result.
        """
        # instantiate test data loader, currently
        # no additional arguments forwarded to instantiation.
        test_loader = self.get_test_dataloader(test_dataset)
        self._log.info("Started testing")
        model.eval()
        test_loss = self._evaluate(model, test_loader, -1, -1)
        self._log.info(f"Test loss: {test_loss}")
        self._write_csv_loss(
            ModelMode.TEST,
            loss=test_loss,
            elements_seen=-1,
            epoch=-1,
            batch=-1,
            cum_batch=-1,
            lr=-1,
            avg_grad=-1,
            avg_grad_clipped=-1,
        )
        self._log.info("Finished testing")
