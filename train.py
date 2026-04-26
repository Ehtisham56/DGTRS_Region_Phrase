from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import TrainConfig, parse_config_from_args
from dataset_loader import create_dataloaders
from eval import evaluate_recall
from loss.contrastive_loss import (
    CLIPContrastiveLoss,
    DGCLApproxLoss,
    RegionPhraseLoss,
    dgcl_alpha_schedule,
)
from model.dgtrs_clip import DGTRSCLIP
from model.dgtrs_longclip import DGTRSLongCLIP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.model_family == "longclip_approx":
        return DGTRSLongCLIP(
            longclip_checkpoint=cfg.longclip_checkpoint,
            longclip_base_model=cfg.longclip_base_model,
            freeze_longclip_visual=cfg.freeze_longclip_visual,
            freeze_longclip_text=cfg.freeze_longclip_text,
        )

    return DGTRSCLIP(
        projection_dim=cfg.projection_dim,
        text_model_name=cfg.text_model_name,
        pretrained_image_encoder=cfg.pretrained_image_encoder,
        freeze_image_encoder=cfg.freeze_image_encoder,
        freeze_text_encoder=cfg.freeze_text_encoder,
        text_gradient_checkpointing=cfg.text_gradient_checkpointing,
    )


def build_optimizer(cfg: TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_name = cfg.optimizer_name.lower()
    if optimizer_name == "auto":
        optimizer_name = "sgd" if cfg.model_family == "longclip_approx" else "adamw"

    if optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            dampening=cfg.dampening,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )

    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    raise ValueError(f"Unknown optimizer_name: {cfg.optimizer_name}")


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: torch.nn.Module,
    local_criterion: torch.nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    accumulation_steps: int,
    use_amp: bool,
    grad_clip_norm: float,
    lambda_local: float = 0.3,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss        = 0.0   # total loss
    running_global_loss = 0.0
    running_local_loss  = 0.0
    running_long_loss   = 0.0
    running_short_loss  = 0.0
    running_alpha       = 0.0
    num_batches = len(loader)
    amp_enabled = use_amp and device.type == "cuda"

    alpha = dgcl_alpha_schedule(
        epoch_index=epoch - 1,
        total_epochs=cfg.epochs,
        warmup_ratio=cfg.dgcl_warmup_ratio,
        decay_ratio=cfg.dgcl_decay_ratio,
        alpha_min=cfg.dgcl_alpha_min,
        refine_perturb=cfg.dgcl_refine_perturb,
    )

    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            if cfg.model_family == "longclip_approx":
                long_input_ids = batch["long_input_ids"].to(device, non_blocking=True)
                short_input_ids = batch["short_input_ids"].to(device, non_blocking=True)
                phrase_input_ids = batch["phrase_input_ids"].to(device, non_blocking=True)

                outputs = model(
                    images=images,
                    long_input_ids=long_input_ids,
                    short_input_ids=short_input_ids if cfg.use_dgcl else None,
                    phrase_input_ids=phrase_input_ids,
                )

                if cfg.use_dgcl and "logits_per_image_short" in outputs:
                    loss_dict = criterion(
                        logits_per_image_long=outputs["logits_per_image"],
                        logits_per_image_short=outputs["logits_per_image_short"],
                        alpha=alpha,
                    )
                    running_long_loss += float(loss_dict["loss_long"].item())
                    running_short_loss += float(loss_dict["loss_short"].item())
                    running_alpha += alpha
                else:
                    loss_dict = criterion(outputs["logits_per_image"])

                loss_global = loss_dict["loss"]

                # --- Local loss (region–phrase) ---
                if "patch_emb" in outputs and "phrase_emb" in outputs:
                    patch_emb_local  = outputs["patch_emb"][:, :64, :]   # (B, ≤64, D)
                    phrase_emb_local = outputs["phrase_emb"][:, :8,  :]  # (B, ≤8,  D)

                    local_dict  = local_criterion(patch_emb_local, phrase_emb_local)
                    loss_local  = local_dict["loss"]
                    running_local_loss += float(loss_local.item())
                else:
                    loss_local = torch.tensor(0.0, device=device)

                running_global_loss += float(loss_global.item())

                # --- Combine ---
                loss = loss_global + lambda_local * loss_local

            else:
                # ---- Baseline path (pure global alignment) ----
                input_ids      = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss_dict   = criterion(outputs["logits_per_image"])
                loss = loss_dict["loss"]

        running_loss += loss.item()

        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()

        should_step = (step + 1) % accumulation_steps == 0 or (step + 1) == num_batches
        if should_step:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        del outputs, loss_dict, loss, scaled_loss, images
        # Free local-alignment tensors to avoid accumulation across steps
        if cfg.model_family == "longclip_approx":
            del loss_global, loss_local

    metrics = {
        "loss": running_loss / max(1, num_batches),
    }
    
    if cfg.model_family == "longclip_approx":
        metrics["loss_global"] = running_global_loss / max(1, num_batches)
        metrics["loss_local"]  = running_local_loss  / max(1, num_batches)
        if cfg.use_dgcl:
            metrics["loss_long"]  = running_long_loss  / max(1, num_batches)
            metrics["loss_short"] = running_short_loss / max(1, num_batches)
            metrics["alpha"]      = running_alpha      / max(1, num_batches)

    return metrics


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    metrics: dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config.to_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def load_checkpoint_for_resume(
    resume_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    last_epoch = int(checkpoint.get("epoch", 0))
    last_metrics = checkpoint.get("metrics", {})
    return last_epoch, last_metrics


def _load_checkpoint_metadata(checkpoint_path: Path) -> tuple[int, dict[str, float]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics", {})
    return epoch, metrics


def _write_best_summary(
    run_dir: Path, epoch: int, metrics: dict[str, float], train_loss: float
) -> None:
    summary_path = run_dir / "best_summary.json"
    payload = {
        "epoch": epoch,
        "train_loss": train_loss,
        "metrics": metrics,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    cfg = parse_config_from_args()

    if cfg.model_family == "longclip_approx":
        if cfg.tokenization_mode != "longclip":
            print("Overriding tokenization_mode to 'longclip' for longclip_approx.")
            cfg.tokenization_mode = "longclip"
        if cfg.normalization_mode != "clip":
            print("Overriding normalization_mode to 'clip' for longclip_approx.")
            cfg.normalization_mode = "clip"
    elif cfg.model_family == "baseline":
        if cfg.tokenization_mode != "distilbert":
            print("Overriding tokenization_mode to 'distilbert' for baseline.")
            cfg.tokenization_mode = "distilbert"
        if cfg.normalization_mode != "imagenet":
            print("Overriding normalization_mode to 'imagenet' for baseline.")
            cfg.normalization_mode = "imagenet"

    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader, _ = create_dataloaders(
        train_csv=cfg.train_csv,
        val_csv=cfg.val_csv,
        image_root=cfg.image_root,
        image_fallback_roots=cfg.image_fallback_roots,
        tokenizer_name=cfg.text_model_name,
        tokenization_mode=cfg.tokenization_mode,
        normalization_mode=cfg.normalization_mode,
        image_size=cfg.image_size,
        max_text_length=cfg.max_text_length,
        long_context_length=cfg.long_context_length,
        short_truncate_length=cfg.short_truncate_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        persistent_workers=cfg.persistent_workers,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    model = build_model(cfg).to(device)

    if cfg.model_family == "longclip_approx" and cfg.use_dgcl:
        criterion: torch.nn.Module = DGCLApproxLoss()
    else:
        criterion = CLIPContrastiveLoss()

    # Local region–phrase loss (only active for baseline model family)
    local_criterion: torch.nn.Module = RegionPhraseLoss()

    optimizer = build_optimizer(cfg, model)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    start_epoch = 1
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        run_dir = resume_path.parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.output_dir) / f"{cfg.run_name}_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    best_metric = -math.inf
    monitor_metric = cfg.monitor_metric
    best_epoch = 0
    best_epoch_loss = math.inf
    best_epoch_metrics: dict[str, float] = {}

    if cfg.resume_from:
        resumed_epoch, resumed_metrics = load_checkpoint_for_resume(
            resume_path=Path(cfg.resume_from),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = resumed_epoch + 1

        # Best-so-far bookkeeping: prefer a lightweight summary, then fall back
        # to best_model.pt metadata (if present). If neither exists, fall back
        # to the resumed checkpoint's metrics.
        run_best_summary = run_dir / "best_summary.json"
        if run_best_summary.exists():
            try:
                payload = json.loads(run_best_summary.read_text(encoding="utf-8"))
                best_epoch = int(payload.get("epoch", 0))
                best_epoch_loss = float(payload.get("train_loss", math.inf))
                best_epoch_metrics = payload.get("metrics", {}) or {}
                best_metric = float(best_epoch_metrics.get(monitor_metric, -math.inf))
            except (OSError, ValueError, TypeError):
                best_epoch = 0
                best_epoch_loss = math.inf
                best_epoch_metrics = {}

        if best_epoch == 0 and (run_dir / "best_model.pt").exists():
            best_epoch, best_epoch_metrics = _load_checkpoint_metadata(
                run_dir / "best_model.pt"
            )
            best_epoch_loss = math.inf
            best_metric = float(best_epoch_metrics.get(monitor_metric, -math.inf))

        if best_epoch == 0:
            best_epoch = resumed_epoch
            best_epoch_metrics = resumed_metrics
            best_epoch_loss = math.inf
            best_metric = float(resumed_metrics.get(monitor_metric, -math.inf))

    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")
    if cfg.resume_from:
        print(f"Resuming from: {cfg.resume_from}")
        print(f"Resume start epoch: {start_epoch}")

    if start_epoch > cfg.epochs:
        print(
            "Resume checkpoint is already at/after target epochs. "
            "Increase --epochs to continue training."
        )
        print(f"Checkpoints saved to: {run_dir}")
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            local_criterion=local_criterion,
            device=device,
            cfg=cfg,
            epoch=epoch,
            accumulation_steps=cfg.accumulation_steps,
            use_amp=cfg.use_amp,
            grad_clip_norm=cfg.grad_clip_norm,
        )

        scheduler.step()

        val_metrics = evaluate_recall(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=cfg.use_amp,
            model_family=cfg.model_family,
        )

        metric_value = val_metrics.get(monitor_metric, val_metrics["recall_at_1"])

        log_line = (
            f"Epoch [{epoch}/{cfg.epochs}] "
            f"Total Loss: {train_metrics['loss']:.4f} "
            f"Val R@1: {val_metrics['recall_at_1']:.4f} "
            f"Val R@5: {val_metrics['recall_at_5']:.4f}"
        )
        # Baseline: show global / local breakdown
        if cfg.model_family not in ("longclip_approx",) and "loss_global" in train_metrics:
            log_line += (
                f" Global: {train_metrics['loss_global']:.4f}"
                f" Local: {train_metrics['loss_local']:.4f}"
            )
        if cfg.model_family == "longclip_approx" and cfg.use_dgcl:
            log_line += (
                f" Long/Short: {train_metrics['loss_long']:.4f}/"
                f"{train_metrics['loss_short']:.4f}"
                f" Alpha: {train_metrics['alpha']:.4f}"
            )
        print(log_line)

        is_best = float(metric_value) > best_metric
        if is_best:
            best_metric = float(metric_value)
            best_epoch = epoch
            best_epoch_loss = float(train_metrics.get("loss", math.inf))
            best_epoch_metrics = dict(val_metrics)
            _write_best_summary(
                run_dir=run_dir,
                epoch=best_epoch,
                metrics=best_epoch_metrics,
                train_loss=best_epoch_loss,
            )

        if best_epoch > 0:
            best_r1 = float(best_epoch_metrics.get("recall_at_1", float("nan")))
            best_r5 = float(best_epoch_metrics.get("recall_at_5", float("nan")))
            best_loss_str = (
                f"{best_epoch_loss:.4f}" if math.isfinite(best_epoch_loss) else "N/A"
            )
            print(
                "Best so far: "
                f"Epoch {best_epoch} "
                f"Train Loss: {best_loss_str} "
                f"Val R@1: {best_r1:.4f} "
                f"Val R@5: {best_r5:.4f}"
            )

        if cfg.save_every_epoch:
            save_checkpoint(
                checkpoint_path=run_dir / f"epoch_{epoch}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=cfg,
                metrics=val_metrics,
            )

        if is_best:
            save_checkpoint(
                checkpoint_path=run_dir / "best_model.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=cfg,
                metrics=val_metrics,
            )

        save_checkpoint(
            checkpoint_path=run_dir / "last_model.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=cfg,
            metrics=val_metrics,
        )

    print("Training completed.")
    if best_epoch > 0:
        print(f"Best epoch: {best_epoch}")
    print(f"Best {monitor_metric}: {best_metric:.4f}")
    print(f"Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()
