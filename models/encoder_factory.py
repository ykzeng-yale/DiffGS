from typing import Any, Dict

from models.archs.encoders.conv_pointnet import ConvPointnet
from models.archs.encoders.dgcnn import DGCNN
from models.ptv3_encoder import PTv3Encoder

ENCODER_REGISTRY: Dict[str, Any] = {
    "conv_pointnet": ConvPointnet,
    "dgcnn": DGCNN,
    "ptv3": PTv3Encoder,
}


def get_encoder(name: str, **kwargs):
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type: {name}")
    cls = ENCODER_REGISTRY[name]
    # filter kwargs unsupported by class
    from inspect import signature
    sig = signature(cls.__init__)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**filtered)
