import torch
from PIL import Image
import numpy as np
from vqvae_string_codec_256 import (
    VQVAE,
    encode_image_to_string,
    decode_string_to_image
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "./checkpoints/demo/vqvae_latest.pt"   # <-- Your trained model path


def load_model():
    model = VQVAE(
        in_channels=3,
        hidden=256,
        z_dim=128,
        num_embeddings=4096
    ).to(DEVICE)

    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def encode_my_image(img_path):
    model = load_model()
    s = encode_image_to_string(model, img_path, device=DEVICE)
    print("\nEncoded string:\n", s)
    return s


def decode_my_string(s, out_path="decoded.png"):
    model = load_model()
    img = decode_string_to_image(model, s, device=DEVICE)
    img.save(out_path)
    print("\nDecoded image saved to:", out_path)


if __name__ == "__main__":
    # CHANGE THIS
    test_image = "peakpx.png"

    # 1) Encode image → string
    encoded = encode_my_image(test_image)

    # 2) Decode string → reconstructed image
    decode_my_string(encoded, "reconstructed.png")
