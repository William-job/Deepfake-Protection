"""
训练脚本说明：
  - train.py（使用本 Trainer 类）为规范训练脚本，支持完整训练流程（checkpoint、日志、验证等）
  - full_quick_train.py 为快速原型脚本，用于快速实验和原型验证，功能精简
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.utils.logger import log_metrics
from src.utils.metrics import compute_all_metrics
from src.training.losses import CARLoss
from src.training.curriculum import ProgressiveMixedCurriculum
from src.models.gating import TemperatureScheduler
from src.quantization.qat import QATRegularizer


class Trainer:
    def __init__(self, model, config, logger, writer, device="cuda", use_amp=True):
        self.model = model.to(device)
        self.config = config
        self.logger = logger
        self.writer = writer
        self.device = device
        self.use_amp = use_amp and device != "cpu"

        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda', init_scale=128.0)
            self.logger.info("AMP (FP16) mixed precision training enabled")

        self.criterion = CARLoss(
            aux_loss_weight=config.training.aux_loss_weight,
            load_balance_weight=config.training.load_balance_weight,
            difficulty_loss_weight=getattr(config.training, "difficulty_loss_weight", 0.0),
            min_expert_weight=getattr(config.training, "min_expert_weight", 0.0),
            min_expert_threshold=getattr(config.training, "min_expert_threshold", 0.05),
            label_smoothing=getattr(config.training, "label_smoothing", 0.0),
        )

        stem_params = []
        other_params = []
        freeze_stem = getattr(config.training, "freeze_stem", False)
        for name, param in model.named_parameters():
            if "stem" in name:
                if freeze_stem:
                    param.requires_grad = False
                    continue
                stem_params.append(param)
            else:
                other_params.append(param)

        if freeze_stem:
            logger.info(f"Stem frozen ({sum(1 for n,p in model.named_parameters() if 'stem' in n)} layers), "
                        f"only fine-tuning expert/gating/difficulty heads")
            param_groups = [{"params": other_params, "lr": config.training.lr}]
        else:
            param_groups = [
                {"params": stem_params, "lr": config.training.lr_backbone},
                {"params": other_params, "lr": config.training.lr},
            ]

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=config.training.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs - config.training.warmup_epochs,
        )

        self.temp_scheduler = TemperatureScheduler(
            start_temp=config.training.temperature_start,
            end_temp=config.training.temperature_end,
            total_epochs=config.training.temperature_anneal_epochs,
        )
        self.curriculum = ProgressiveMixedCurriculum(total_epochs=config.training.epochs)

        self.qat = QATRegularizer(
            model,
            min_bit_width=getattr(config.training, "qat_min_bit", 6),
            max_bit_width=getattr(config.training, "qat_max_bit", 8),
            noise_scale=getattr(config.training, "qat_noise_scale", 0.02),
            stem_protect=True,
        )
        self._current_stage = None
        self._ema_decay = getattr(config.training, "ema_decay", 0.999)
        self._ema_shadow = {}
        self._ema_enabled = getattr(config.training, "ema_enabled", True)

        self.current_epoch = 0
        self.best_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        self._train_auc_history = []
        self._val_auc_history = []
        self._train_loss_history = []
        self._val_loss_history = []
        self._overfit_warned = False

        # Session tracking
        self._session_id = 0
        self._session_start_time = None

        # NaN detection
        self._nan_consecutive_count = 0
        self._nan_max_consecutive = 10

        # OOM auto-recovery
        self._oom_original_batch_size = None
        self._oom_degraded = False

        # Memory monitoring
        self._memory_history = []  # list of (max_allocated, max_reserved) per epoch

        # Epoch timing for intermediate checkpoint
        self._epoch_times = []  # list of epoch durations in seconds

    def _init_ema(self):
        if not self._ema_enabled:
            return
        self._ema_shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._ema_shadow[name] = param.data.clone().detach()

    def _update_ema(self):
        if not self._ema_enabled:
            return
        for name, param in self.model.named_parameters():
            if name in self._ema_shadow:
                self._ema_shadow[name] = (
                    self._ema_decay * self._ema_shadow[name]
                    + (1.0 - self._ema_decay) * param.data.detach()
                )

    def _swap_ema_weights(self):
        if not self._ema_enabled or not self._ema_shadow:
            return {}
        backup = {}
        for name, param in self.model.named_parameters():
            if name in self._ema_shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self._ema_shadow[name])
        return backup

    def _restore_weights(self, backup):
        for name, param in self.model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_qat_loss = 0.0
        all_preds = []
        all_labels = []

        stage = self._current_stage or {}
        qat_enabled = stage.get("qat_enabled", False)
        qat_weight = stage.get("qat_loss_weight", 0.0)

        # Task 10: gradient accumulation
        accum_steps = getattr(self.config.training, "gradient_accumulation_steps", 1)
        if accum_steps < 1:
            accum_steps = 1

        # Task 12: statistics for w_mean and gate_logits
        all_w_means = []  # per-expert mean weights
        all_gate_logits = []  # gate logits

        # Task 7: intermediate checkpoint
        estimated_epoch_time_min = None
        if len(self._epoch_times) > 0:
            avg_epoch_time = sum(self._epoch_times) / len(self._epoch_times)
            estimated_epoch_time_min = avg_epoch_time / 60.0
        self._step_save_enabled = estimated_epoch_time_min is not None and estimated_epoch_time_min > 30

        total_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            retry_batch = True
            while retry_batch:
                retry_batch = False
                try:
                    frames = batch["frames"].to(self.device)
                    labels = batch["label"].to(self.device)

                    if accum_steps > 1 and batch_idx % accum_steps == 0:
                        self.optimizer.zero_grad()
                    elif accum_steps == 1:
                        self.optimizer.zero_grad()

                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(frames)
                            loss, loss_dict = self.criterion(outputs, labels)

                        qat_loss = torch.tensor(0.0, device=self.device)
                        if qat_enabled and qat_weight > 0:
                            qat_loss_raw, _ = self.qat.compute_regularization_loss(
                                difficulty=outputs.get("difficulty")
                            )
                            loss = loss + qat_weight * qat_loss_raw
                            qat_loss = qat_loss_raw.detach()

                        if accum_steps > 1:
                            loss = loss / accum_steps

                        self.scaler.scale(loss).backward()
                    else:
                        outputs = self.model(frames)
                        loss, loss_dict = self.criterion(outputs, labels)

                        qat_loss = torch.tensor(0.0, device=self.device)
                        if qat_enabled and qat_weight > 0:
                            qat_loss_raw, _ = self.qat.compute_regularization_loss(
                                difficulty=outputs.get("difficulty")
                            )
                            loss = loss + qat_weight * qat_loss_raw
                            qat_loss = qat_loss_raw.detach()

                        if accum_steps > 1:
                            loss = loss / accum_steps

                        loss.backward()

                    # Task 8: NaN detection on loss
                    loss_val = loss.item()
                    is_nan_loss = torch.isnan(torch.tensor(loss_val)) or torch.isinf(torch.tensor(loss_val))
                    if is_nan_loss:
                        self._nan_consecutive_count += 1
                        self.logger.warning(
                            f"NaN/Inf loss detected at epoch {self.current_epoch}, "
                            f"batch {batch_idx} (consecutive: {self._nan_consecutive_count}). "
                            f"Skipping batch."
                        )
                        if self._nan_consecutive_count >= self._nan_max_consecutive:
                            self.logger.error(
                                f"Terminating training: {self._nan_max_consecutive} consecutive "
                                f"NaN/Inf batches detected."
                            )
                            raise RuntimeError(
                                f"Training terminated: {self._nan_max_consecutive} consecutive NaN/Inf batches."
                            )
                        self.optimizer.zero_grad()
                        continue

                    # Gradient clipping and optimizer step (accumulation aware)
                    if (batch_idx + 1) % accum_steps == 0 or batch_idx == len(train_loader) - 1:
                        if self.use_amp:
                            # Unscale gradients first (this also detects NaN/Inf in unscaled grads)
                            self.scaler.unscale_(self.optimizer)
                            # Task 8: NaN detection on gradients (after unscaling)
                            grad_nan = False
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        grad_nan = True
                                        break
                            if grad_nan:
                                self._nan_consecutive_count += 1
                                self.logger.warning(
                                    f"NaN/Inf gradient detected at epoch {self.current_epoch}, "
                                    f"batch {batch_idx} (consecutive: {self._nan_consecutive_count}). "
                                    f"Skipping batch."
                                )
                                if self._nan_consecutive_count >= self._nan_max_consecutive:
                                    self.logger.error(
                                        f"Terminating training: {self._nan_max_consecutive} consecutive "
                                        f"NaN/Inf gradient batches detected."
                                    )
                                    raise RuntimeError(
                                        f"Training terminated: {self._nan_max_consecutive} consecutive NaN/Inf gradient batches."
                                    )
                                self.scaler.update()  # Let scaler adapt to NaN
                                self.optimizer.zero_grad()
                                continue
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.training.grad_clip
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.training.grad_clip
                            )
                            # Task 8: NaN detection on gradients (FP32)
                            grad_nan = False
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        grad_nan = True
                                        break
                            if grad_nan:
                                self._nan_consecutive_count += 1
                                self.logger.warning(
                                    f"NaN/Inf gradient detected at epoch {self.current_epoch}, "
                                    f"batch {batch_idx} (consecutive: {self._nan_consecutive_count}). "
                                    f"Skipping batch."
                                )
                                if self._nan_consecutive_count >= self._nan_max_consecutive:
                                    self.logger.error(
                                        f"Terminating training: {self._nan_max_consecutive} consecutive "
                                        f"NaN/Inf gradient batches detected."
                                    )
                                    raise RuntimeError(
                                        f"Training terminated: {self._nan_max_consecutive} consecutive NaN/Inf gradient batches."
                                    )
                                self.optimizer.zero_grad()
                                continue
                            self.optimizer.step()

                    # Both loss and gradients are clean — reset counter
                    self._nan_consecutive_count = 0
                    self._update_ema()

                except torch.cuda.OutOfMemoryError as e:
                    # Task 9: OOM auto-recovery
                    current_bs = self.config.data.batch_size
                    if self._oom_original_batch_size is None:
                        self._oom_original_batch_size = current_bs
                    new_bs = max(2, current_bs // 2)
                    if new_bs < current_bs:
                        self._oom_degraded = True
                        self.config.data.batch_size = new_bs
                        self.logger.warning(
                            f"CUDA OOM at epoch {self.current_epoch}, batch {batch_idx}. "
                            f"Reducing batch_size from {current_bs} to {new_bs}. "
                            f"Rebuilding DataLoader..."
                        )
                        # Rebuild DataLoader with new batch size
                        from src.data.dataloader import create_dataloader
                        train_loader = create_dataloader(self.config, split="train")
                        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
                        pbar.update(batch_idx)  # Skip to current position
                        # Clean up GPU memory
                        torch.cuda.empty_cache()
                        retry_batch = True
                        continue
                    else:
                        self.logger.error(
                            f"CUDA OOM but batch_size already at minimum (2). "
                            f"Cannot reduce further. Raising error."
                        )
                        raise

            # Process successful batch
            total_loss += loss.item()
            total_qat_loss += qat_loss.item()

            logits = outputs["logits"]
            if logits.size(1) > 1:
                preds = torch.sigmoid(logits[:, 1])
            else:
                preds = torch.sigmoid(logits.squeeze(-1))
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            # Task 12: collect w_mean and gate_logits
            w = outputs.get("w")
            if w is not None:
                all_w_means.append(w.detach().mean(dim=0).cpu().numpy())  # [4]
            gl = outputs.get("gate_logits")
            if gl is not None:
                all_gate_logits.append(gl.detach().cpu().numpy())

            if batch_idx % self.config.logging.log_interval == 0:
                extra = {"loss": f"{loss.item():.4f}"}
                if "loss_difficulty" in loss_dict and loss_dict.get("loss_difficulty", 0) > 0:
                    extra["d_loss"] = f"{loss_dict['loss_difficulty']:.4f}"
                if loss_dict.get("loss_anti_collapse", 0) > 0:
                    extra["ac"] = f"{loss_dict['loss_anti_collapse']:.4f}"
                if qat_loss.item() > 0:
                    extra["qat"] = f"{qat_loss.item():.4f}"
                if self._oom_degraded:
                    extra["bs"] = str(self.config.data.batch_size)
                pbar.set_postfix(extra)
                # Also log to file for progress monitoring
                self.logger.info(
                    f"Epoch {self.current_epoch} [Train] {batch_idx}/{total_batches} "
                    f"({batch_idx * 100 // total_batches}%) - loss: {loss.item():.4f}"
                )

            # Task 7: intermediate checkpoint every 500 steps
            if self._step_save_enabled and batch_idx > 0 and batch_idx % 500 == 0:
                step_ckpt_dir = self.config.logging.checkpoint_dir
                os.makedirs(step_ckpt_dir, exist_ok=True)
                step_ckpt_path = os.path.join(
                    step_ckpt_dir,
                    f"checkpoint_epoch_{self.current_epoch}_step_{batch_idx}.pt"
                )
                step_ckpt = {
                    "epoch": self.current_epoch,
                    "step": batch_idx,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict() if self.use_amp else None,
                }
                torch.save(step_ckpt, step_ckpt_path)
                self.logger.info(
                    f"Intermediate checkpoint saved: epoch {self.current_epoch}, step {batch_idx}"
                )

        avg_loss = total_loss / len(train_loader)
        metrics = compute_all_metrics(np.array(all_preds), np.array(all_labels))

        log_metrics(self.writer, {"loss": avg_loss, **metrics}, self.current_epoch, "train")

        extra_info = ""
        if "loss_difficulty" in loss_dict and loss_dict.get("loss_difficulty", 0) > 0:
            extra_info += f", DiffLoss: {loss_dict['loss_difficulty']:.4f}"
        if loss_dict.get("loss_anti_collapse", 0) > 0:
            extra_info += f", AntiCol: {loss_dict['loss_anti_collapse']:.4f}"
        if total_qat_loss > 0 and qat_enabled:
            avg_qat = total_qat_loss / len(train_loader)
            extra_info += f", QATLoss: {avg_qat:.4f}"
            self.writer.add_scalar("train/qat_loss", avg_qat, self.current_epoch)

        # Task 12: log w_mean and gate_logits statistics
        if all_w_means:
            w_means_arr = np.stack(all_w_means, axis=0)  # [num_batches, 4]
            for i, name in enumerate(self.model.expert_names):
                w_mean = w_means_arr[:, i].mean()
                w_std = w_means_arr[:, i].std()
                self.logger.info(
                    f"  Expert '{name}' w_mean: mean={w_mean:.4f}, std={w_std:.4f}"
                )
                self.writer.add_scalar(f"train/w_mean_{name}", w_mean, self.current_epoch)
                self.writer.add_scalar(f"train/w_std_{name}", w_std, self.current_epoch)
        if all_gate_logits:
            gate_logits_arr = np.concatenate(all_gate_logits, axis=0)  # [total_batches*B, 4]
            for i, name in enumerate(self.model.expert_names):
                gl_mean = gate_logits_arr[:, i].mean()
                gl_std = gate_logits_arr[:, i].std()
                self.logger.info(
                    f"  Expert '{name}' gate_logits: mean={gl_mean:.4f}, std={gl_std:.4f}"
                )
                self.writer.add_scalar(f"train/gate_logits_mean_{name}", gl_mean, self.current_epoch)
                self.writer.add_scalar(f"train/gate_logits_std_{name}", gl_std, self.current_epoch)

        self.logger.info(
            f"Epoch {self.current_epoch} Train - Loss: {avg_loss:.4f}{extra_info}, "
            f"Acc: {metrics['accuracy']:.4f}(th={metrics['threshold']:.3f}), "
            f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}"
        )

        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_difficulty = []
        total_d_loss = 0.0

        total_batches = len(val_loader)
        log_interval = max(1, total_batches // 10)  # Log every 10% progress
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        valid_loss_count = 0
        for batch_idx, batch in enumerate(pbar):
            frames = batch["frames"].to(self.device)
            labels = batch["label"].to(self.device)

            try:
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(frames)
                        loss, loss_dict = self.criterion(outputs, labels)
                else:
                    outputs = self.model(frames)
                    loss, loss_dict = self.criterion(outputs, labels)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.logger.warning(
                    f"OOM during validation at epoch {self.current_epoch}, "
                    f"batch {batch_idx}. Skipping batch."
                )
                continue

            # NaN/Inf loss 跳过：保留 preds 用于 AUC，但不计入 avg_loss
            loss_val = loss.item()
            is_nan_loss = torch.isnan(torch.tensor(loss_val)) or torch.isinf(torch.tensor(loss_val))
            if is_nan_loss:
                self.logger.warning(
                    f"NaN/Inf loss during validation at epoch {self.current_epoch}, "
                    f"batch {batch_idx} (loss={loss_val}). Skipping loss accumulation."
                )
            else:
                total_loss += loss_val
                valid_loss_count += 1
                if "loss_difficulty" in loss_dict:
                    total_d_loss += loss_dict["loss_difficulty"]

            logits = outputs["logits"]
            if logits.size(1) > 1:
                preds = torch.sigmoid(logits[:, 1])
            else:
                preds = torch.sigmoid(logits.squeeze(-1))
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            diff = outputs.get("difficulty")
            if diff is not None:
                all_difficulty.extend(diff.cpu().numpy().flatten().tolist())

            extra = {"loss": f"{loss_val:.4f}"}
            if "loss_difficulty" in loss_dict:
                extra["d_loss"] = f"{loss_dict['loss_difficulty']:.4f}"
            pbar.set_postfix(extra)

            # Progress logging: log every 10% of validation
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                progress_pct = (batch_idx + 1) * 100 // total_batches
                self.logger.info(
                    f"Epoch {self.current_epoch} [Val] {batch_idx + 1}/{total_batches} "
                    f"({progress_pct}%) - loss: {loss_val:.4f}"
                )

        avg_loss = total_loss / max(valid_loss_count, 1)
        metrics = compute_all_metrics(np.array(all_preds), np.array(all_labels))

        log_metrics(self.writer, {"loss": avg_loss, **metrics}, self.current_epoch, "val")

        extra_info = ""
        if total_d_loss > 0:
            extra_info = f", DiffLoss: {total_d_loss / max(valid_loss_count, 1):.4f}"
        if all_difficulty:
            d_arr = np.array(all_difficulty)
            extra_info += f", d_mean: {d_arr.mean():.3f}"
            self.writer.add_scalar("val/difficulty_mean", d_arr.mean(), self.current_epoch)
            self.writer.add_scalar("val/difficulty_std", d_arr.std(), self.current_epoch)
        self.logger.info(
            f"Epoch {self.current_epoch} Val   - Loss: {avg_loss:.4f}{extra_info}, "
            f"Acc: {metrics['accuracy']:.4f}(th={metrics['threshold']:.3f}), "
            f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}"
        )

        return avg_loss, metrics

    def save_checkpoint(self, metrics, is_best=False):
        checkpoint_dir = self.config.logging.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_auc": self.best_auc,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "history": {
                "train_auc": self._train_auc_history,
                "val_auc": self._val_auc_history,
                "train_loss": self._train_loss_history,
                "val_loss": self._val_loss_history,
            },
        }
        if self.use_amp:
            checkpoint["scaler"] = self.scaler.state_dict()
        if self._ema_enabled and self._ema_shadow:
            checkpoint["ema_shadow"] = self._ema_shadow

        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{self.current_epoch}.pt")
        torch.save(checkpoint, path)

        # Atomic write for checkpoint_latest.pt
        latest_tmp = os.path.join(checkpoint_dir, "checkpoint_latest.pt.tmp")
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_tmp)
        os.replace(latest_tmp, latest_path)

        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            if self._ema_enabled and self._ema_shadow:
                ema_state = self.model.state_dict()
                backup = self._swap_ema_weights()
                ema_path = os.path.join(checkpoint_dir, "best_model_ema.pt")
                torch.save({**checkpoint, "model_state_dict": self.model.state_dict()}, ema_path)
                self._restore_weights(backup)
                self.logger.info(f"EMA best model saved at epoch {self.current_epoch}")
            self.logger.info(f"Best model saved at epoch {self.current_epoch} (AUC: {metrics['auc']:.4f})")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing:
            self.logger.info(f"  New params (random init): {len(missing)} keys")
        if unexpected:
            self.logger.info(f"  Deprecated params: {len(unexpected)} keys")

        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            self.logger.info(f"  Optimizer state partial load (new params will use default): {e}")

        try:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            self.logger.info("  Scheduler state skipped (architecture change)")

        if self.use_amp and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # Load training history
        if "history" in checkpoint:
            hist = checkpoint["history"]
            self._train_auc_history = hist.get("train_auc", self._train_auc_history)
            self._val_auc_history = hist.get("val_auc", self._val_auc_history)
            self._train_loss_history = hist.get("train_loss", self._train_loss_history)
            self._val_loss_history = hist.get("val_loss", self._val_loss_history)

        # Task 13: fix epoch logic - checkpoint["epoch"] is already completed, start from next
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_auc = checkpoint.get("best_auc", 0.0)
        saved_best_epoch = checkpoint.get("best_epoch", 0)
        if (checkpoint["epoch"] + 1) - saved_best_epoch > 3:
            self.best_auc = 0.0
            self.best_epoch = 0
            self.patience_counter = 0
            self.logger.info(f"  best_auc reset (saved best at epoch {saved_best_epoch}, resuming at {self.current_epoch})")

        # Task 6: checkpoint integrity validation - forward dummy input
        if self.device == "cuda" or torch.cuda.is_available():
            try:
                self.model.eval()  # eval mode to avoid batch norm batch_size > 1 requirement
                with torch.no_grad():
                    # Dummy input: [batch=1, frames=8, channels=3, height=224, width=224]
                    dummy = torch.randn(1, 8, 3, 224, 224, device=self.device)
                    _ = self.model(dummy)
                self.model.train()  # restore training mode
                self.logger.info("  Checkpoint integrity validated: forward pass OK")
            except Exception as e:
                self.logger.error(f"  Checkpoint validation FAILED: {str(e)}")
                raise RuntimeError(f"Checkpoint integrity check failed: {e}")

        # Task 5: print resumption summary
        completed_epochs = checkpoint["epoch"] + 1  # since epoch 0-indexed completed
        remaining_epochs = self.config.training.epochs - completed_epochs
        best_val_auc = self.best_auc
        # Estimate remaining time
        if len(self._epoch_times) > 0:
            avg_epoch_time = sum(self._epoch_times) / len(self._epoch_times)
            est_remaining_sec = avg_epoch_time * remaining_epochs
            est_remaining_min = est_remaining_sec / 60
            est_remaining_hour = est_remaining_min / 60
            if est_remaining_hour >= 1:
                time_str = f"{est_remaining_hour:.1f} hours"
            else:
                time_str = f"{est_remaining_min:.1f} minutes"
        else:
            time_str = "N/A (no previous timing)"

        self.logger.info("=" * 60)
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"  Completed epochs: {completed_epochs}")
        self.logger.info(f"  Best validation AUC: {best_val_auc:.4f}")
        self.logger.info(f"  Remaining epochs: {remaining_epochs}")
        self.logger.info(f"  Estimated remaining time: {time_str}")
        self.logger.info("=" * 60)

    def _check_overfit(self, window=3):
        if len(self._val_auc_history) < window + 1:
            return

        t_auc = self._train_auc_history[-window:]
        v_auc = self._val_auc_history[-window:]
        t_loss = self._train_loss_history[-window:]
        v_loss = self._val_loss_history[-window:]

        train_auc_delta = t_auc[-1] - t_auc[0]
        val_auc_delta = v_auc[-1] - v_auc[0]

        flags = []

        auc_gap = t_auc[-1] - v_auc[-1]
        if auc_gap > 0.015:
            flags.append(f"Train AUC exceeds Val AUC ({auc_gap:+.4f})")

        if val_auc_delta < -0.01:
            flags.append(f"Val AUC declining ({val_auc_delta:+.4f} over {window} epochs)")

        if train_auc_delta > 0.015 and val_auc_delta < train_auc_delta * 0.3:
            flags.append(f"Train AUC rising faster than Val ({train_auc_delta:+.4f} vs {val_auc_delta:+.4f})")

        if len(v_loss) >= 3:
            x = list(range(len(v_loss)))
            trend = sum((x[i] - x[0]) * (v_loss[i] - v_loss[0]) for i in range(len(x))) / max(sum((xi - x[0]) ** 2 for xi in x), 1)
            if trend > 0.01:
                flags.append(f"Val loss rising ({trend:+.4f}/epoch)")

        loss_ratio = v_loss[-1] / max(t_loss[-1], 0.001)
        if loss_ratio > 2.5:
            flags.append(f"Val loss {loss_ratio:.1f}x train loss")

        if flags and not self._overfit_warned:
            self._overfit_warned = True
            self.logger.warning("=" * 50)
            self.logger.warning("⚠ OVERFITTING DETECTED at epoch %d", self.current_epoch)
            for flag in flags:
                self.logger.warning("  • %s", flag)
            self.logger.warning("=" * 50)

        return bool(flags)

    def train(self, train_loader, val_loader, resume_from=None):
        # Task 4: session logging - start
        self._session_id += 1
        self._session_start_time = time.time()
        session_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._session_start_time))
        self.logger.info(f"=== Session {self._session_id} 开始 === {session_start_str}")

        # Task 15: save config snapshot to log directory
        log_dir = self.config.logging.log_dir
        os.makedirs(log_dir, exist_ok=True)
        try:
            import yaml
            config_snapshot_path = os.path.join(log_dir, "config_snapshot.yaml")
            with open(config_snapshot_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config.to_dict(), f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"Config snapshot saved to {config_snapshot_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save config snapshot: {e}")

        # Task 16: create experiment_declaration.txt template if not exists
        exp_decl_path = os.path.join(log_dir, "experiment_declaration.txt")
        if not os.path.exists(exp_decl_path):
            template = (
                "=== Experiment Declaration ===\n"
                "Hypothesis:\n"
                "  [Describe the hypothesis being tested]\n\n"
                "IV (Independent Variable):\n"
                "  [List the independent variables]\n\n"
                "DV (Dependent Variable):\n"
                "  [List the dependent variables]\n\n"
                "CV (Control Variable):\n"
                "  [List the control variables]\n\n"
                "Expected:\n"
                "  [Describe expected outcomes]\n\n"
                "=== Fill in the above sections before training ===\n"
            )
            with open(exp_decl_path, "w", encoding="utf-8") as f:
                f.write(template)
            self.logger.info(f"Experiment declaration template created at {exp_decl_path}. Please fill in before training.")

        if resume_from:
            self.load_checkpoint(resume_from)

        self._init_ema()

        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Task 11: memory monitoring at epoch start
            if torch.cuda.is_available():
                max_allocated = torch.cuda.max_memory_allocated()
                max_reserved = torch.cuda.max_memory_reserved()
                self._memory_history.append((max_allocated, max_reserved))
                self.logger.info(
                    f"Epoch {epoch} - GPU Memory: allocated={max_allocated / 1024 ** 2:.1f}MB, "
                    f"reserved={max_reserved / 1024 ** 2:.1f}MB"
                )
                # Check for memory leak: 3 consecutive epochs with > 100MB growth
                if len(self._memory_history) >= 3:
                    recent = self._memory_history[-3:]
                    growth_allocated = (recent[-1][0] - recent[0][0]) / (1024 ** 2)
                    growth_reserved = (recent[-1][1] - recent[0][1]) / (1024 ** 2)
                    epochs_span = len(recent) - 1
                    if epochs_span > 0:
                        per_epoch_growth_alloc = growth_allocated / epochs_span
                        per_epoch_growth_resv = growth_reserved / epochs_span
                        if per_epoch_growth_alloc > 100 or per_epoch_growth_resv > 100:
                            self.logger.warning(
                                "⚠ POTENTIAL MEMORY LEAK: "
                                f"allocated memory growing {per_epoch_growth_alloc:.1f}MB/epoch, "
                                f"reserved memory growing {per_epoch_growth_resv:.1f}MB/epoch "
                                f"over last {epochs_span} epochs"
                            )

            temp = self.temp_scheduler.step(self.model.gating, epoch)
            stage = self.curriculum.get_stage(epoch)
            self._current_stage = stage
            # 课程切换时检测 top_k 变化，重置门控参数打破死状态。
            # top_k=1 阶段门控梯度为 0，参数保持初始值不变；切到 top_k=2 时若不重置，
            # 死状态会被带入新阶段，导致未激活专家永远无法进入激活集。
            prev_top_k = self.model.gating.top_k
            self.model.gating.top_k = min(self.config.model.top_k, stage["target_top_k"])
            if self.model.gating.top_k != prev_top_k:
                self.model.gating.reset_parameters()
                # 刷新门控参数的 EMA shadow，避免重置后 EMA 仍持有旧值导致不同步
                if self._ema_enabled and self._ema_shadow:
                    for name, param in self.model.gating.named_parameters(prefix="gating"):
                        full_name = name if name.startswith("gating") else f"gating.{name}"
                        if full_name in self._ema_shadow:
                            self._ema_shadow[full_name] = param.data.clone().detach()
                self.logger.info(f"  Curriculum switch: top_k {prev_top_k} -> {self.model.gating.top_k}, gating reset")
            self.criterion.difficulty_loss_weight = getattr(self.config.training, "difficulty_loss_weight", 0.0) * stage["difficulty_loss_weight"]
            self.criterion.min_expert_weight = getattr(self.config.training, "min_expert_weight", 0.0) * stage.get("anti_collapse_weight", 0)
            qat_tag = f", qat={stage['qat_loss_weight']:.3f}" if stage["qat_enabled"] else ""
            ac_tag = f", ac={self.criterion.min_expert_weight:.3f}" if self.criterion.min_expert_weight > 0 else ""
            self.logger.info(f"Epoch {epoch}: stage = {stage['name']}, temperature = {temp:.4f}, top_k = {self.model.gating.top_k}{qat_tag}{ac_tag}")

            train_loss, train_metrics = self.train_epoch(train_loader)

            # Track epoch time
            epoch_duration = time.time() - epoch_start_time
            self._epoch_times.append(epoch_duration)

            self._train_auc_history.append(train_metrics["auc"])
            self._train_loss_history.append(train_loss)

            if epoch >= self.config.training.warmup_epochs:
                self.scheduler.step()

            if epoch % self.config.logging.eval_interval == 0:
                val_loss, val_metrics = self.validate(val_loader)

                self._val_auc_history.append(val_metrics["auc"])
                self._val_loss_history.append(val_loss)

                self._check_overfit()

                if val_metrics["auc"] > self.best_auc:
                    self.best_auc = val_metrics["auc"]
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.save_checkpoint(val_metrics, is_best=True)
                else:
                    self.patience_counter += 1

            if epoch % self.config.logging.save_interval == 0:
                self.save_checkpoint(val_metrics if "val_metrics" in dir() else train_metrics)

            if self.patience_counter >= self.config.training.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.logger.info(f"Training completed. Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch}")

        # Task 4: session logging - end
        session_end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.logger.info(f"=== Session {self._session_id} 结束 === {session_end_str}")
