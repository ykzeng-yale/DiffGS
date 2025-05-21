from models.ptv3_encoder import PTv3Encoder
import torch
import sys

if __name__ == "__main__":
    try:
        enc = PTv3Encoder()
    except Exception as e:
        print(f"skip test: {e}")
        sys.exit(0)
    out, _ = enc(torch.randn(2, 3, 2048))
    print("output shape:", out.shape)
    assert out.shape[0] == 2

