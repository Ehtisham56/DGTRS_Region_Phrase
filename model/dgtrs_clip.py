from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .projection_head import ProjectionHead
from .text_encoder import TextEncoder


class DGTRSCLIP(nn.Module):
    def __init__(
        self,
        projection_dim: int = 256,
        text_model_name: str = "distilbert-base-uncased",
        pretrained_image_encoder: bool = True,
        freeze_image_encoder: bool = False,
        freeze_text_encoder: bool = False,
        text_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(
            pretrained=pretrained_image_encoder,
            freeze=freeze_image_encoder,
        )
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            freeze=freeze_text_encoder,
            gradient_checkpointing=text_gradient_checkpointing,
        )

        self.image_projection = ProjectionHead(
            input_dim=self.image_encoder.output_dim,
            projection_dim=projection_dim,
        )
        self.text_projection = ProjectionHead(
            input_dim=self.text_encoder.output_dim,
            projection_dim=projection_dim,
        )

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images)
        image_embeddings = self.image_projection(image_features)
        return F.normalize(image_embeddings, dim=-1)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self.text_projection(text_features)
        return F.normalize(text_embeddings, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
        logits_per_text  = logits_per_image.t()

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings":  text_embeddings,
            "logits_per_image": logits_per_image,
            "logits_per_text":  logits_per_text,
            "logit_scale":      logit_scale,
        }
