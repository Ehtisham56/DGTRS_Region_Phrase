from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import longclip


class DGTRSLongCLIP(nn.Module):
    def __init__(
        self,
        longclip_checkpoint: str = "",
        longclip_base_model: str = "ViT-B/16",
        freeze_longclip_visual: bool = False,
        freeze_longclip_text: bool = False,
    ) -> None:
        super().__init__()

        checkpoint_path = Path(longclip_checkpoint) if longclip_checkpoint else None
        if checkpoint_path is not None:
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Long-CLIP checkpoint not found: {checkpoint_path}"
                )
            backbone, _ = longclip.load(str(checkpoint_path), device="cpu")
            self.init_source = f"checkpoint:{checkpoint_path}"
        else:
            try:
                backbone, _ = longclip.load_from_clip(
                    longclip_base_model,
                    device="cpu",
                    jit=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize Long-CLIP from CLIP weights. "
                    "Provide --longclip_checkpoint to a local DGTRS-CLIP/Long-CLIP state dict."
                ) from exc
            self.init_source = f"clip:{longclip_base_model}"

        self.backbone = backbone

        if freeze_longclip_visual:
            for param in self.backbone.visual.parameters():
                param.requires_grad = False

        if freeze_longclip_text:
            text_modules = [
                self.backbone.transformer,
                self.backbone.token_embedding,
                self.backbone.ln_final,
            ]
            for module in text_modules:
                for param in module.parameters():
                    param.requires_grad = False
            self.backbone.text_projection.requires_grad = False
            self.backbone.positional_embedding.requires_grad = False
            if hasattr(self.backbone, "positional_embedding_res"):
                self.backbone.positional_embedding_res.requires_grad = False

        # --- Region-phrase alignment ---
        if hasattr(self.backbone.visual, 'transformer'):
            patch_dim = self.backbone.visual.width
        else:
            patch_dim = self.backbone.visual.attnpool.c_proj.in_features
            
        embed_dim = self.backbone.text_projection.shape[1]
        self.patch_proj = nn.Linear(patch_dim, embed_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone.encode_image(images)
        return F.normalize(image_features, dim=-1)

    def encode_image_with_patches(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.backbone.visual
        if hasattr(visual, 'transformer'):
            # ViT logic
            x = visual.conv1(images.type(visual.conv1.weight.dtype))
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)
            
            global_feat = visual.ln_post(x[:, 0, :])
            if visual.proj is not None:
                global_feat = global_feat @ visual.proj
                
            patches = x[:, 1:, :]  # (B, N, C)
        else:
            # ResNet logic
            def stem(x):
                x = visual.relu1(visual.bn1(visual.conv1(x)))
                x = visual.relu2(visual.bn2(visual.conv2(x)))
                x = visual.relu3(visual.bn3(visual.conv3(x)))
                x = visual.avgpool(x)
                return x

            x = images.type(visual.conv1.weight.dtype)
            x = stem(x)
            x = visual.layer1(x)
            x = visual.layer2(x)
            x = visual.layer3(x)
            feat_map = visual.layer4(x)  # (B, C, H, W)
            global_feat = visual.attnpool(feat_map)
            
            B, C, H, W = feat_map.shape
            patches = feat_map.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

        global_emb = F.normalize(global_feat, dim=-1)
        patch_emb = self.patch_proj(patches)
        patch_emb = F.normalize(patch_emb, dim=-1)
        return global_emb, patch_emb

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        text_features = self.backbone.encode_text(token_ids)
        return F.normalize(text_features, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        long_input_ids: torch.Tensor,
        short_input_ids: torch.Tensor | None = None,
        phrase_input_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if phrase_input_ids is not None:
            image_embeddings, patch_emb = self.encode_image_with_patches(images)
        else:
            image_embeddings = self.encode_image(images)
            patch_emb = None
            
        text_embeddings_long = self.encode_text(long_input_ids)

        logit_scale = self.backbone.logit_scale.exp().clamp(max=100.0)
        logits_per_image_long = logit_scale * image_embeddings @ text_embeddings_long.t()

        outputs = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings_long,
            "text_embeddings_long": text_embeddings_long,
            "logit_scale": logit_scale,
            "logits_per_image": logits_per_image_long,
            "logits_per_text": logits_per_image_long.t(),
        }
        
        if patch_emb is not None:
            outputs["patch_emb"] = patch_emb

        if short_input_ids is not None:
            text_embeddings_short = self.encode_text(short_input_ids)
            logits_per_image_short = logit_scale * image_embeddings @ text_embeddings_short.t()
            outputs["text_embeddings_short"] = text_embeddings_short
            outputs["logits_per_image_short"] = logits_per_image_short
            outputs["logits_per_text_short"] = logits_per_image_short.t()

        # --- Region-phrase local alignment ---
        if phrase_input_ids is not None:
            B, P, L = phrase_input_ids.shape
            flat_ids = phrase_input_ids.view(B * P, L)
            
            flat_feat = self.encode_text(flat_ids)
            phrase_emb = flat_feat.view(B, P, -1)  # (B, P, D)
            
            outputs["phrase_emb"] = phrase_emb

        return outputs
