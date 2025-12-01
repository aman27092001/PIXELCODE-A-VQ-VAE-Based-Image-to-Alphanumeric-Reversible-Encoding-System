# train_custom.py
import torch
from vqvae_string_codec_256 import train

if __name__ == "__main__":
    # IMPORTANT FOR WINDOWS: DO NOT REMOVE THIS GUARD
    model = train(
        num_epochs=120,                 # last value = 30
        dataset_root="./images",     # folder with subfolders
        image_size=256,
        batch_size=16,                # safe for small datasets  last value  = 4
        num_embeddings=2048,
        z_dim=256,                     # last value = 128
        hidden=384,                    # last value = 256
        num_workers=0,               # REQUIRED on Windows + small dataset
        out_dir="./checkpoints/custom"
    )

    print("Training complete. Model saved in ./checkpoints/custom/")
