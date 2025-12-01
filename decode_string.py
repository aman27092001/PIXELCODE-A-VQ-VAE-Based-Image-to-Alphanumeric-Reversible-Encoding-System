# decode_string.py
import torch
from torchvision.utils import save_image

from vqvae_string_codec_256 import (
    VQVAE,
    load_checkpoint,
    decode_string_to_image,
    IMAGE_SIZE, Z_DIM, HIDDEN, NUM_EMBEDDINGS
)

def load_model(ckpt_path, device):
    model = VQVAE(
        in_channels=3,
        hidden=HIDDEN,
        z_dim=Z_DIM,
        num_embeddings=NUM_EMBEDDINGS,
    ).to(device)
    load_checkpoint(model, ckpt_path, map_location=device)
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = "./checkpoints/custom/vqvae_epoch120.pth"            # epoch 30
    encoded_string = input("Paste encoded string:\n\n")

    print("\nLoading model…")
    model = load_model(ckpt, device)

    print("Decoding…")
    img = decode_string_to_image(
        model,
        encoded_string,
        device=device,
        gzip_compress=True,
        expect_checksum=True
    )

    img_vis = (img + 1) * 0.5

    save_path = "./decoded_image/decoded.png"
    save_image(img_vis, save_path)

    print("\nDecoded image saved as:", save_path)
