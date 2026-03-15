"""
Custom callbacks for the SFTTrainer.

Includes:
    - WandbMetricsCallback: logs extra metrics to W&B (GPU memory, LR, etc.)
    - EarlyStoppingOnPlateau: stops training if eval loss plateaus
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class WandbMetricsCallback(TrainerCallback):
    """Log additional system metrics to Weights & Biases.

    Tracks:
        - Peak GPU memory allocated
        - Current GPU memory allocated
        - Current learning rate
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if logs is None:
            return

        # GPU memory metrics
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logs["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated(device) / 1e9, 2
            )
            logs["gpu_memory_peak_gb"] = round(
                torch.cuda.max_memory_allocated(device) / 1e9, 2
            )
            logs["gpu_memory_reserved_gb"] = round(
                torch.cuda.memory_reserved(device) / 1e9, 2
            )


class EarlyStoppingOnPlateau(TrainerCallback):
    """Stop training early if eval loss does not improve for N evaluations.

    Parameters
    ----------
    patience : int
        Number of evaluation checks with no improvement after which
        training will be stopped.
    min_delta : float
        Minimum change in the monitored metric to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.wait = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ):
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if self.best_loss is None or eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.wait = 0
            logger.info(
                f"[EarlyStopping] New best eval_loss: {eval_loss:.4f} | patience reset"
            )
        else:
            self.wait += 1
            logger.info(
                f"[EarlyStopping] No improvement. "
                f"eval_loss: {eval_loss:.4f} (best: {self.best_loss:.4f}) | "
                f"patience: {self.wait}/{self.patience}"
            )
            if self.wait >= self.patience:
                logger.warning(
                    f"[EarlyStopping] Stopping training — no improvement for "
                    f"{self.patience} evaluations."
                )
                control.should_training_stop = True


class LogModelInfoCallback(TrainerCallback):
    """Log model parameter counts at the start of training."""

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if model is None:
            return

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = 100 * trainable_params / total_params if total_params > 0 else 0

        logger.info(
            f"\n{'='*60}\n"
            f"  Model Parameter Summary\n"
            f"{'='*60}\n"
            f"  Total parameters:     {total_params:>15,}\n"
            f"  Trainable parameters: {trainable_params:>15,}\n"
            f"  Trainable %:          {pct:>14.2f}%\n"
            f"{'='*60}"
        )

        # Also log to W&B if available
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.summary["total_params"] = total_params
                wandb.run.summary["trainable_params"] = trainable_params
                wandb.run.summary["trainable_pct"] = round(pct, 2)
        except ImportError:
            pass
