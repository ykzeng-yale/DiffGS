from typing import Tuple
import torch
import torch.nn as nn

try:
    from pointcept.models.backbones.ptv3 import PTV3Backbone  # type: ignore
except Exception:
    # allows docs generation or environments without pointcept
    PTV3Backbone = None  # type: ignore

class PTv3Encoder(nn.Module):
    """Thin wrapper adapting Point Transformer V3 backbone to DiffGS."""
    requires_query = False

    def __init__(self, out_channels: int = 256, **kwargs) -> None:
        super().__init__()
        if PTV3Backbone is None:
            raise ImportError("pointcept is required for PTv3Encoder")
        self.backbone = PTV3Backbone(width=out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, 3, N].
        Returns:
            Tuple of encoded features [B, C, N] and None for compatibility.
        """
        feats = self.backbone(x)  # [B, N, C]
        feats = feats.permute(0, 2, 1).contiguous()  # -> [B, C, N]
        return feats, None
