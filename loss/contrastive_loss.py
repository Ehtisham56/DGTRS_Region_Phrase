from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _symmetric_clip_ce(logits_per_image: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size = logits_per_image.size(0)
    targets = torch.arange(batch_size, device=logits_per_image.device)

    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_image.t(), targets)
    loss = 0.5 * (loss_i2t + loss_t2i)

    return {
        "loss": loss,
        "loss_i2t": loss_i2t,
        "loss_t2i": loss_t2i,
    }


def dgcl_alpha_schedule(
    epoch_index: int,
    total_epochs: int,
    warmup_ratio: float,
    decay_ratio: float,
    alpha_min: float,
    refine_perturb: float,
) -> float:
    if total_epochs <= 1:
        return 1.0

    warmup_ratio = max(0.0, min(1.0, warmup_ratio))
    decay_ratio = max(0.0, min(1.0 - warmup_ratio, decay_ratio))
    alpha_min = max(0.0, min(1.0, alpha_min))
    refine_perturb = max(0.0, min(1.0, refine_perturb))

    progress = epoch_index / max(1, total_epochs - 1)
    stage1_end = warmup_ratio
    stage2_end = warmup_ratio + decay_ratio

    if progress <= stage1_end:
        return 1.0

    if progress <= stage2_end:
        stage_progress = (progress - stage1_end) / max(1e-8, decay_ratio)
        return alpha_min + 0.5 * (1.0 - alpha_min) * (1.0 + math.cos(math.pi * stage_progress))

    refine_progress = (progress - stage2_end) / max(1e-8, 1.0 - stage2_end)
    perturb = refine_perturb * (1.0 - refine_progress) * 0.5 * (
        1.0 + math.cos(math.pi * refine_progress)
    )
    return max(alpha_min, min(1.0, alpha_min + perturb))


class CLIPContrastiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits_per_image: torch.Tensor) -> dict[str, torch.Tensor]:
        return _symmetric_clip_ce(logits_per_image)


class RegionPhraseLoss(nn.Module):
    """Local contrastive loss for region–phrase alignment.
    
    Implements EXACTLY:
    - sim = patch_emb @ phrase_emb^T  -> (B, N, P)
    - max_sim, _ = sim.max(dim=1)     -> (B, P)
    - loss = cross_entropy(max_sim, arange(B))
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        patch_emb:  torch.Tensor,   # (B, N, D)
        phrase_emb: torch.Tensor,   # (B, P, D)
    ) -> dict[str, torch.Tensor]:
        B, N, D = patch_emb.shape
        
        # STEP 5: Patch-Phrase Similarity
        # sim = patch_emb @ phrase_emb^T
        # Shape: (B, N, P)
        sim = torch.matmul(patch_emb, phrase_emb.transpose(1, 2))

        # STEP 6: Region Selection
        # max_sim: (B, P)
        max_sim, _ = sim.max(dim=1)

        # STEP 7: Local Loss
        # labels: torch.arange(B)
        labels = torch.arange(B, device=max_sim.device)

        loss_local = F.cross_entropy(max_sim, labels)

        return {
            "loss": loss_local,
            "loss_i2p": loss_local,
            "loss_p2i": torch.tensor(0.0, device=max_sim.device)
        }


class DGCLApproxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits_per_image_long: torch.Tensor,
        logits_per_image_short: torch.Tensor,
        alpha: float,
    ) -> dict[str, torch.Tensor]:
        alpha = float(max(0.0, min(1.0, alpha)))
        long_loss = _symmetric_clip_ce(logits_per_image_long)
        short_loss = _symmetric_clip_ce(logits_per_image_short)

        total_loss = alpha * long_loss["loss"] + (1.0 - alpha) * short_loss["loss"]

        return {
            "loss": total_loss,
            "loss_long": long_loss["loss"],
            "loss_short": short_loss["loss"],
            "loss_i2t_long": long_loss["loss_i2t"],
            "loss_t2i_long": long_loss["loss_t2i"],
            "loss_i2t_short": short_loss["loss_i2t"],
            "loss_t2i_short": short_loss["loss_t2i"],
            "alpha": torch.tensor(alpha, device=total_loss.device),
        }
