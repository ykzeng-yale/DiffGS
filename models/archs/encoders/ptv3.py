"""PTv3 encoder wrapper."""

import warnings
import torch.nn as nn


class PTv3Encoder(nn.Module):
    """Wrapper for the PTv3 encoder from the Pointcept project.

    This module requires the :mod:`pointcept` package. The constructor tries to
    import the Point Transformer V3 implementation and will raise an informative
    error if the package is missing. The interface is intentionally simple and
    mirrors the minimal functionality used by :class:`GsModel`.
    """

    def __init__(self, c_dim=512, dim=3, **kwargs):
        super().__init__()
        try:
            from pointcept.models.backbones.pt_v3 import PointTransformerV3
        except Exception as exc:  # pragma: no cover - library might be missing
            raise ImportError(
                "PTv3Encoder requires the `pointcept` package. "
                "Please install it from https://github.com/Pointcept/Pointcept"
            ) from exc

        self.model = PointTransformerV3(in_channels=dim, embed_dim=c_dim, **kwargs)

    def forward(self, points, query=None):
        """Forward pass of the encoder.

        Parameters
        ----------
        points : Tensor
            Input point cloud of shape ``(B, N, dim)``.
        query : Tensor, optional
            Query points. Currently ignored as PTv3 operates directly on the
            input ``points``.
        """
        return self.model(points)

    # The following methods are provided for API compatibility with
    # :class:`ConvPointnet`. They fall back to using the query points
    # directly and emit a warning so existing pipelines keep working.

    def forward_with_plane_features(self, plane_features, query):  # noqa: D401
        warnings.warn(
            "PTv3Encoder ignores plane features; using query points directly",
            stacklevel=2,
        )
        return self.forward(query)

    def forward_with_pc_features(self, c, p, query):  # noqa: D401
        warnings.warn(
            "PTv3Encoder does not use precomputed point features; using query points",
            stacklevel=2,
        )
        return self.forward(query)

    def get_point_cloud_features(self, p):  # noqa: D401
        """Return features for the input points."""
        return self.model(p)

    def get_plane_features(self, p):  # noqa: D401
        """Plane features are unavailable for PTv3."""
        raise NotImplementedError(
            "PTv3Encoder does not provide plane features; consider using ConvPointnet"
        )
