from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DistilBertModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        freeze: bool = False,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(model_name)
        self.output_dim = self.backbone.config.hidden_size

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooled output; CLS-equivalent is first token embedding.
        cls_embedding = outputs.last_hidden_state[:, 0]
        return cls_embedding
