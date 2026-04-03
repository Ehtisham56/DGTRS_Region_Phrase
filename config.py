from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    # Paths
    train_csv: str = "dataset/train/ret2.csv"
    val_csv: str = ""
    image_root: str = "images"
    image_fallback_roots: list[str] = field(default_factory=lambda: ["img"])
    output_dir: str = "checkpoints"
    run_name: str = "dgtrs_longclip_single_gpu"
    resume_from: str = ""

    # Mode
    model_family: str = "longclip_approx"  # longclip_approx | baseline
    tokenization_mode: str = "longclip"  # longclip | distilbert
    normalization_mode: str = "clip"  # clip | imagenet

    # Runtime
    seed: int = 42
    device: str = "auto"  # auto | cuda | cpu
    use_amp: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    # Data
    image_size: int = 224
    max_text_length: int = 64
    long_context_length: int = 248
    short_truncate_length: int = 20
    val_split: float = 0.1

    # Model
    projection_dim: int = 256
    text_model_name: str = "distilbert-base-uncased"
    pretrained_image_encoder: bool = True
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = False
    text_gradient_checkpointing: bool = True
    longclip_checkpoint: str = ""
    longclip_base_model: str = "ViT-B/16"
    freeze_longclip_visual: bool = False
    freeze_longclip_text: bool = False

    # Optimization
    epochs: int = 3
    batch_size: int = 4
    accumulation_steps: int = 4
    optimizer_name: str = "auto"  # auto | sgd | adamw
    lr: float = 5e-4
    momentum: float = 0.9
    dampening: float = 0.1
    nesterov: bool = False
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0

    # DGCL approximation for single-caption data
    use_dgcl: bool = True
    dgcl_alpha_min: float = 0.2
    dgcl_warmup_ratio: float = 0.2
    dgcl_decay_ratio: float = 0.6
    dgcl_refine_perturb: float = 0.1

    # Logging/checkpointing
    save_every_epoch: bool = False
    monitor_metric: str = "recall_at_1"  # recall_at_1 | recall_at_5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2)


class StoreBoolAction(argparse.Action):
    """Parses explicit booleans for CLI flags (true/false/1/0)."""

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, bool):
            result = values
        else:
            lowered = str(values).strip().lower()
            if lowered in {"1", "true", "yes", "y"}:
                result = True
            elif lowered in {"0", "false", "no", "n"}:
                result = False
            else:
                raise argparse.ArgumentTypeError(
                    f"Invalid boolean value '{values}' for {option_string}."
                )
        setattr(namespace, self.dest, result)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DGTRS-CLIP style model")

    parser.add_argument("--train_csv", type=str, default=TrainConfig.train_csv)
    parser.add_argument("--val_csv", type=str, default=TrainConfig.val_csv)
    parser.add_argument("--image_root", type=str, default=TrainConfig.image_root)
    parser.add_argument(
        "--image_fallback_roots",
        nargs="*",
        default=TrainConfig().image_fallback_roots,
    )
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--run_name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--resume_from", type=str, default=TrainConfig.resume_from)

    parser.add_argument("--model_family", type=str, default=TrainConfig.model_family)
    parser.add_argument(
        "--tokenization_mode", type=str, default=TrainConfig.tokenization_mode
    )
    parser.add_argument(
        "--normalization_mode", type=str, default=TrainConfig.normalization_mode
    )

    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--use_amp", action=StoreBoolAction, default=TrainConfig.use_amp)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument(
        "--pin_memory", action=StoreBoolAction, default=TrainConfig.pin_memory
    )
    parser.add_argument(
        "--persistent_workers",
        action=StoreBoolAction,
        default=TrainConfig.persistent_workers,
    )

    parser.add_argument("--image_size", type=int, default=TrainConfig.image_size)
    parser.add_argument(
        "--max_text_length", type=int, default=TrainConfig.max_text_length
    )
    parser.add_argument(
        "--long_context_length", type=int, default=TrainConfig.long_context_length
    )
    parser.add_argument(
        "--short_truncate_length", type=int, default=TrainConfig.short_truncate_length
    )
    parser.add_argument("--val_split", type=float, default=TrainConfig.val_split)

    parser.add_argument("--projection_dim", type=int, default=TrainConfig.projection_dim)
    parser.add_argument("--text_model_name", type=str, default=TrainConfig.text_model_name)
    parser.add_argument(
        "--pretrained_image_encoder",
        action=StoreBoolAction,
        default=TrainConfig.pretrained_image_encoder,
    )
    parser.add_argument(
        "--freeze_image_encoder",
        action=StoreBoolAction,
        default=TrainConfig.freeze_image_encoder,
    )
    parser.add_argument(
        "--freeze_text_encoder",
        action=StoreBoolAction,
        default=TrainConfig.freeze_text_encoder,
    )
    parser.add_argument(
        "--text_gradient_checkpointing",
        action=StoreBoolAction,
        default=TrainConfig.text_gradient_checkpointing,
    )
    parser.add_argument(
        "--longclip_checkpoint", type=str, default=TrainConfig.longclip_checkpoint
    )
    parser.add_argument(
        "--longclip_base_model", type=str, default=TrainConfig.longclip_base_model
    )
    parser.add_argument(
        "--freeze_longclip_visual",
        action=StoreBoolAction,
        default=TrainConfig.freeze_longclip_visual,
    )
    parser.add_argument(
        "--freeze_longclip_text",
        action=StoreBoolAction,
        default=TrainConfig.freeze_longclip_text,
    )

    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument(
        "--accumulation_steps", type=int, default=TrainConfig.accumulation_steps
    )
    parser.add_argument("--optimizer_name", type=str, default=TrainConfig.optimizer_name)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--momentum", type=float, default=TrainConfig.momentum)
    parser.add_argument("--dampening", type=float, default=TrainConfig.dampening)
    parser.add_argument("--nesterov", action=StoreBoolAction, default=TrainConfig.nesterov)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--min_lr", type=float, default=TrainConfig.min_lr)
    parser.add_argument("--grad_clip_norm", type=float, default=TrainConfig.grad_clip_norm)

    parser.add_argument("--use_dgcl", action=StoreBoolAction, default=TrainConfig.use_dgcl)
    parser.add_argument(
        "--dgcl_alpha_min", type=float, default=TrainConfig.dgcl_alpha_min
    )
    parser.add_argument(
        "--dgcl_warmup_ratio", type=float, default=TrainConfig.dgcl_warmup_ratio
    )
    parser.add_argument(
        "--dgcl_decay_ratio", type=float, default=TrainConfig.dgcl_decay_ratio
    )
    parser.add_argument(
        "--dgcl_refine_perturb", type=float, default=TrainConfig.dgcl_refine_perturb
    )

    parser.add_argument(
        "--save_every_epoch",
        action=StoreBoolAction,
        default=TrainConfig.save_every_epoch,
    )
    parser.add_argument("--monitor_metric", type=str, default=TrainConfig.monitor_metric)

    return parser


def parse_config_from_args() -> TrainConfig:
    parser = build_arg_parser()
    args = parser.parse_args()
    return TrainConfig(**vars(args))
