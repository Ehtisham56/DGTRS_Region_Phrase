from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DistilBertTokenizerFast

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.Resampling.BICUBIC


@dataclass
class Sample:
    filename: str
    caption: str


def build_image_transform(
    image_size: int,
    is_train: bool,
    normalization_mode: str = "imagenet",
) -> transforms.Compose:
    mode = normalization_mode.lower()
    if mode == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    elif mode == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.9, 1.0), interpolation=BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_samples(csv_file: str | Path) -> list[Sample]:
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    samples: list[Sample] = []
    with csv_file.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            filename = row.get("filename", "").strip()
            caption = row.get("title", "").strip()
            if not filename or not caption:
                continue
            samples.append(Sample(filename=filename, caption=caption))

    if not samples:
        raise RuntimeError(f"No valid samples found in {csv_file}")
    return samples


def split_samples(
    samples: list[Sample], val_split: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be in range (0, 1)")

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split_idx = int(len(indices) * (1.0 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Phrase extraction  (Step 2 of Region–Phrase alignment)
# ---------------------------------------------------------------------------

def extract_phrases(text: str, min_len: int = 4) -> list[str]:
    """Split a caption on commas and return non-trivial phrase fragments.

    Example:
        "airport runway, taxiway, terminal building" ->
        ["airport runway", "taxiway", "terminal building"]

    Phrases shorter than *min_len* characters are dropped to avoid
    passing trivial/empty tokens to the text encoder.
    """
    return [p.strip() for p in text.split(",") if len(p.strip()) >= min_len]


class DGTRSDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_root: str | Path,
        image_fallback_roots: list[str] | None,
        transform: transforms.Compose,
    ) -> None:
        self.samples = samples
        self.image_root = Path(image_root)
        self.image_fallback_roots = [Path(p) for p in (image_fallback_roots or [])]
        self.transform = transform

        # Resolve image paths once to avoid repeated disk checks per sample.
        self.resolved_image_paths = [self._resolve_image_path(s.filename) for s in samples]

    def _resolve_image_path(self, filename: str) -> Path:
        primary = self.image_root / filename
        if primary.exists():
            return primary

        for root in self.image_fallback_roots:
            candidate = root / filename
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Image '{filename}' not found in {self.image_root} or fallback roots "
            f"{self.image_fallback_roots}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = self.resolved_image_paths[idx]

        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image_tensor = self.transform(image)

        # Extract comma-separated phrases from the caption for local alignment.
        phrases = extract_phrases(sample.caption)

        return {
            "image":    image_tensor,
            "caption":  sample.caption,
            "filename": sample.filename,
            "phrases":  phrases,   # list[str], variable length per sample
        }


class CLIPCollator:
    def __init__(
        self,
        tokenizer: DistilBertTokenizerFast | None,
        max_text_length: int,
        tokenization_mode: str,
        long_context_length: int,
        short_truncate_length: int,
        # --- Local alignment settings ---
        max_phrases: int = 8,
        phrase_max_length: int = 32,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.tokenization_mode = tokenization_mode.lower()
        self.long_context_length = long_context_length
        self.short_truncate_length = short_truncate_length
        # Maximum number of phrases per sample (excess phrases are truncated)
        self.max_phrases = max_phrases
        self.phrase_max_length = phrase_max_length

        self.longclip_tokenizer = None
        if self.tokenization_mode == "longclip":
            from model.longclip import tokenizer as longclip_tokenizer

            self.longclip_tokenizer = longclip_tokenizer

    def _tokenize_phrases(
        self, batch_phrases: list[list[str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and pad phrases to a fixed (B, P, L) tensor pair.

        Args:
            batch_phrases: B sublists, each holding variable-length strings.

        Returns:
            phrase_input_ids      — (B, P, L)  long tensor
            phrase_attention_mask — (B, P, L)  long tensor
        """
        assert self.longclip_tokenizer is not None, "LongCLIP tokenizer required for phrases"

        B = len(batch_phrases)
        P = self.max_phrases
        L = self.phrase_max_length

        # Pre-allocate with padding (token_id=0). For LongCLIP, there's no attention_mask returned by default.
        # LongCLIP tokenizer expects a list of strings and returns a tensor.
        all_ids = torch.zeros(B, P, L, dtype=torch.long)

        for i, phrases in enumerate(batch_phrases):
            phrases = phrases[:P]   # truncate to max_phrases
            if not phrases:
                continue            # leave all-zero rows for this sample

            enc = self.longclip_tokenizer(
                phrases,
                context_length=L,
                truncate_length=L,
                truncate=True,
            ).to(dtype=torch.long)
            
            n = len(phrases)
            all_ids[i, :n] = enc

        return all_ids

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images    = torch.stack([item["image"] for item in batch], dim=0)
        captions  = [item["caption"]  for item in batch]
        filenames = [item["filename"] for item in batch]
        phrases   = [item["phrases"]  for item in batch]   # list of list[str]

        output = {
            "images":    images,
            "captions":  captions,
            "filenames": filenames,
            "phrases":   phrases,   # raw strings kept for debugging
        }

        if self.tokenization_mode == "distilbert":
            if self.tokenizer is None:
                raise RuntimeError("DistilBERT tokenizer was not initialized.")

            # --- Global caption tokens (unchanged from baseline) ---
            text = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            output["input_ids"]      = text["input_ids"]
            output["attention_mask"] = text["attention_mask"]

            return output

        if self.tokenization_mode == "longclip":
            if self.longclip_tokenizer is None:
                raise RuntimeError("Long-CLIP tokenizer was not initialized.")

            long_tokens = self.longclip_tokenizer(
                captions,
                context_length=self.long_context_length,
                truncate_length=self.long_context_length,
                truncate=True,
            ).to(dtype=torch.long)
            short_tokens = self.longclip_tokenizer(
                captions,
                context_length=self.long_context_length,
                truncate_length=self.short_truncate_length,
                truncate=True,
            ).to(dtype=torch.long)
            output["long_input_ids"]  = long_tokens
            output["short_input_ids"] = short_tokens
            
            # --- Phrase tokens for local region–phrase alignment ---
            phrase_ids = self._tokenize_phrases(phrases)
            output["phrase_input_ids"] = phrase_ids  # (B, P, L)
            
            return output

        raise ValueError(f"Unknown tokenization_mode: {self.tokenization_mode}")


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    image_root: str,
    image_fallback_roots: list[str],
    tokenizer_name: str,
    tokenization_mode: str,
    normalization_mode: str,
    image_size: int,
    max_text_length: int,
    long_context_length: int,
    short_truncate_length: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    val_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DistilBertTokenizerFast | None]:
    tokenizer = None
    if tokenization_mode.lower() == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)

    if val_csv:
        train_samples = load_samples(train_csv)
        val_samples = load_samples(val_csv)
    else:
        samples = load_samples(train_csv)
        train_samples, val_samples = split_samples(samples, val_split=val_split, seed=seed)

    train_dataset = DGTRSDataset(
        samples=train_samples,
        image_root=image_root,
        image_fallback_roots=image_fallback_roots,
        transform=build_image_transform(
            image_size=image_size,
            is_train=True,
            normalization_mode=normalization_mode,
        ),
    )
    val_dataset = DGTRSDataset(
        samples=val_samples,
        image_root=image_root,
        image_fallback_roots=image_fallback_roots,
        transform=build_image_transform(
            image_size=image_size,
            is_train=False,
            normalization_mode=normalization_mode,
        ),
    )

    collator = CLIPCollator(
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        tokenization_mode=tokenization_mode,
        long_context_length=long_context_length,
        short_truncate_length=short_truncate_length,
    )

    use_persistent_workers = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
        collate_fn=collator,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
        collate_fn=collator,
        drop_last=False,
    )

    return train_loader, val_loader, tokenizer
