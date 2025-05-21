from models.ptv3_encoder import PTv3Encoder
import torch

if __name__ == "__main__":
    enc = PTv3Encoder()
    print(enc(torch.randn(2, 3, 2048))[0].shape)
