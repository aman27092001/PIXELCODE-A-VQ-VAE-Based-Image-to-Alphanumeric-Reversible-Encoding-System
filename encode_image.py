# encode_image.py
import torch
import torchvision.transforms as T
from PIL import Image

from vqvae_string_codec_256 import (
    VQVAE,
    load_checkpoint,
    encode_image_to_string,
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

    ckpt = "./checkpoints/custom/vqvae_epoch120.pth"             # epoch 30 last value
    img_path = "./test_image/Test00008.png"

    print("Loading model…")
    model = load_model(ckpt, device)

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)

    print("Encoding…")
    code = encode_image_to_string(
        model,
        img_tensor,
        device=device,
        gzip_compress=True,
        add_checksum=True
    )

    print("\nEncoded string length:", len(code))
    print(code)
