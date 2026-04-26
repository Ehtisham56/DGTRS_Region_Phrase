from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset_loader import create_dataloaders
from model.dgtrs_clip import DGTRSCLIP
from model.dgtrs_longclip import DGTRSLongCLIP


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


@torch.no_grad()
def compute_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    model_family: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    image_embeddings = []
    text_embeddings = []

    amp_enabled = use_amp and device.type == "cuda"

    for batch in tqdm(loader, desc="Computing embeddings", unit="batch"):
        images = batch["images"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            if model_family == "longclip_approx":
                long_input_ids = batch["long_input_ids"].to(device, non_blocking=True)
                outputs = model(images=images, long_input_ids=long_input_ids)
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        image_embeddings.append(outputs["image_embeddings"].float().cpu())
        text_embeddings.append(outputs["text_embeddings"].float().cpu())

        del outputs, images

    return torch.cat(image_embeddings, dim=0), torch.cat(text_embeddings, dim=0)


def _recall_at_k(similarity: torch.Tensor, k: int) -> float:
    topk_indices = similarity.topk(k=k, dim=1).indices
    targets = torch.arange(similarity.size(0)).unsqueeze(1)
    hits = (topk_indices == targets).any(dim=1).float()
    return hits.mean().item()


def evaluate_recall(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    model_family: str,
) -> dict[str, float]:
    image_embeddings, text_embeddings = compute_embeddings(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
        model_family=model_family,
    )

    # Embeddings are already L2-normalized. Dot product equals cosine similarity.
    similarity_i2t = image_embeddings @ text_embeddings.t()
    similarity_t2i = similarity_i2t.t()

    max_k = min(5, similarity_i2t.size(1))
    max_k_10 = min(10, similarity_i2t.size(1))
    i2t_r1 = _recall_at_k(similarity_i2t, k=1)
    i2t_r5 = _recall_at_k(similarity_i2t, k=max_k)
    i2t_r10 = _recall_at_k(similarity_i2t, k=max_k_10)
    t2i_r1 = _recall_at_k(similarity_t2i, k=1)
    t2i_r5 = _recall_at_k(similarity_t2i, k=max_k)
    t2i_r10 = _recall_at_k(similarity_t2i, k=max_k_10)

    return {
        "image_to_text_recall_at_1": i2t_r1,
        "image_to_text_recall_at_5": i2t_r5,
        "image_to_text_recall_at_10": i2t_r10,
        "text_to_image_recall_at_1": t2i_r1,
        "text_to_image_recall_at_5": t2i_r5,
        "text_to_image_recall_at_10": t2i_r10,
        "recall_at_1": 0.5 * (i2t_r1 + t2i_r1),
        "recall_at_5": 0.5 * (i2t_r5 + t2i_r5),
        "recall_at_10": 0.5 * (i2t_r10 + t2i_r10),
    }


def _build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DGTRS-CLIP model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_csv", type=str, default=TrainConfig.train_csv)
    parser.add_argument("--val_csv", type=str, default=TrainConfig.val_csv)
    parser.add_argument("--image_root", type=str, default=TrainConfig.image_root)
    parser.add_argument(
        "--image_fallback_roots",
        nargs="*",
        default=TrainConfig().image_fallback_roots,
    )
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--model_family", type=str, default=TrainConfig.model_family)
    parser.add_argument(
        "--tokenization_mode", type=str, default=TrainConfig.tokenization_mode
    )
    parser.add_argument(
        "--normalization_mode", type=str, default=TrainConfig.normalization_mode
    )
    parser.add_argument("--image_size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--max_text_length", type=int, default=TrainConfig.max_text_length)
    parser.add_argument(
        "--long_context_length", type=int, default=TrainConfig.long_context_length
    )
    parser.add_argument(
        "--short_truncate_length", type=int, default=TrainConfig.short_truncate_length
    )
    parser.add_argument("--text_model_name", type=str, default=TrainConfig.text_model_name)
    parser.add_argument("--projection_dim", type=int, default=TrainConfig.projection_dim)
    parser.add_argument(
        "--longclip_checkpoint", type=str, default=TrainConfig.longclip_checkpoint
    )
    parser.add_argument(
        "--longclip_base_model", type=str, default=TrainConfig.longclip_base_model
    )
    parser.add_argument(
        "--freeze_longclip_visual",
        action="store_true",
        default=TrainConfig.freeze_longclip_visual,
    )
    parser.add_argument(
        "--freeze_longclip_text",
        action="store_true",
        default=TrainConfig.freeze_longclip_text,
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--val_split", type=float, default=TrainConfig.val_split)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    return parser


def main() -> None:
    parser = _build_eval_parser()
    args = parser.parse_args()

    if args.model_family == "longclip_approx":
        if args.tokenization_mode != "longclip":
            print("Overriding tokenization_mode to 'longclip' for longclip_approx.")
            args.tokenization_mode = "longclip"
        if args.normalization_mode != "clip":
            print("Overriding normalization_mode to 'clip' for longclip_approx.")
            args.normalization_mode = "clip"
    elif args.model_family == "baseline":
        if args.tokenization_mode != "distilbert":
            print("Overriding tokenization_mode to 'distilbert' for baseline.")
            args.tokenization_mode = "distilbert"
        if args.normalization_mode != "imagenet":
            print("Overriding normalization_mode to 'imagenet' for baseline.")
            args.normalization_mode = "imagenet"

    device = _resolve_device(args.device)

    _, val_loader, _ = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_root=args.image_root,
        image_fallback_roots=args.image_fallback_roots,
        tokenizer_name=args.text_model_name,
        tokenization_mode=args.tokenization_mode,
        normalization_mode=args.normalization_mode,
        image_size=args.image_size,
        max_text_length=args.max_text_length,
        long_context_length=args.long_context_length,
        short_truncate_length=args.short_truncate_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=False,
        val_split=args.val_split,
        seed=args.seed,
    )

    if args.model_family == "longclip_approx":
        model = DGTRSLongCLIP(
            longclip_checkpoint=args.longclip_checkpoint,
            longclip_base_model=args.longclip_base_model,
            freeze_longclip_visual=args.freeze_longclip_visual,
            freeze_longclip_text=args.freeze_longclip_text,
        ).to(device)
    else:
        model = DGTRSCLIP(
            projection_dim=args.projection_dim,
            text_model_name=args.text_model_name,
        ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_recall(
        model=model,
        loader=val_loader,
        device=device,
        use_amp=args.use_amp,
        model_family=args.model_family,
    )

    print("Validation metrics:")
    print(f"Recall@1: {metrics['recall_at_1']:.4f}")
    print(f"Recall@5: {metrics['recall_at_5']:.4f}")
    print(f"Recall@10: {metrics['recall_at_10']:.4f}")
    print(
        "I2T R@1/R@5/R@10: "
        f"{metrics['image_to_text_recall_at_1']:.4f}/"
        f"{metrics['image_to_text_recall_at_5']:.4f}/"
        f"{metrics['image_to_text_recall_at_10']:.4f}"
    )
    print(
        "T2I R@1/R@5/R@10: "
        f"{metrics['text_to_image_recall_at_1']:.4f}/"
        f"{metrics['text_to_image_recall_at_5']:.4f}/"
        f"{metrics['text_to_image_recall_at_10']:.4f}"
    )


if __name__ == "__main__":
    main()
