"""
vqvae_string_codec.py  - Upgraded VQ-VAE pipeline for image -> alphanumeric string -> image

Features:
- 16x16 latent grid for 128x128 inputs
- config: num_embeddings=512, z_dim=128, hidden=256
- EMA vector quantizer
- deterministic encoding -> gzip -> base62 -> checksum
- robust decode (handles gzip/non-gzip, verifies checksum)
- training loop, demo helper, checkpoints, sample saving
- Windows-safe demo entrypoint suggested in run_demo.py

Usage:
- Put this file and run run_demo.py (see companion run_demo.py snippet).
- Adjust hyperparams in train(...) call or at top constants.

Author: Generated for your project
"""
import os
import math
import gzip
import hashlib
from io import BytesIO
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

# -----------------------------
# Configurable defaults
# -----------------------------
DEFAULT_IMAGE_SIZE = 128
DEFAULT_Z_DIM = 128        # embedding dimension per spatial cell
DEFAULT_HIDDEN = 256       # channel count for conv features
DEFAULT_NUM_EMBEDDINGS = 512
DEFAULT_COMMITMENT = 0.25
DEFAULT_BATCH = 48
DEFAULT_EPOCHS = 30
DEFAULT_NUM_WORKERS = 2    # safe on Windows; increase on Linux

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE = len(ALPHABET)  # 62

# -----------------------------
# Utility: base62 conversions & checksum
# -----------------------------
def int_to_base62(n: int) -> str:
    if n == 0:
        return ALPHABET[0]
    s = []
    while n > 0:
        n, r = divmod(n, BASE)
        s.append(ALPHABET[r])
    return ''.join(reversed(s))

def base62_to_int(s: str) -> int:
    n = 0
    for ch in s:
        n = n * BASE + ALPHABET.index(ch)
    return n

def bits_to_base62(bitstring: str) -> str:
    pad_len = (8 - (len(bitstring) % 8)) % 8
    bitstring_padded = bitstring + ('0' * pad_len)
    b = int(bitstring_padded, 2) if bitstring_padded != '' else 0
    s = int_to_base62(b)
    pad_char = ALPHABET[pad_len]
    return pad_char + s

def base62_to_bits(s: str) -> str:
    if len(s) == 0:
        return ''
    pad_char = s[0]
    pad_len = ALPHABET.index(pad_char)
    num_part = s[1:]
    if num_part == '':
        b = 0
    else:
        b = base62_to_int(num_part)
    bitstring_padded = bin(b)[2:]
    total_bits = ((len(bitstring_padded) + 7) // 8) * 8
    bitstring_padded = bitstring_padded.zfill(total_bits)
    if pad_len > 0:
        return bitstring_padded[:-pad_len]
    else:
        return bitstring_padded

def add_checksum_to_string(s: str) -> str:
    h = hashlib.sha1(s.encode('utf-8')).digest()
    v = int.from_bytes(h[:4], 'big') >> 8  # 24 bits
    cs = int_to_base62(v)
    cs = cs.rjust(4, ALPHABET[0])
    return s + cs

def verify_and_strip_checksum(s: str):
    if len(s) < 4:
        return None, False
    core = s[:-4]
    cs = s[-4:]
    expected = add_checksum_to_string(core)[-4:]
    return core, (cs == expected)

# -----------------------------
# VQ-VAE model components
# -----------------------------
class Encoder(nn.Module):
    """
    Encoder downsamples input 128x128 -> 16x16 (three stride-2 convs).
    Output: B x z_dim x 16 x 16
    """
    def __init__(self, in_channels=3, hidden=DEFAULT_HIDDEN, z_dim=DEFAULT_Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden//2, 4, 2, 1),   # 128 -> 64
            nn.ReLU(True),
            nn.Conv2d(hidden//2, hidden, 4, 2, 1),        # 64 -> 32
            nn.ReLU(True),
            nn.Conv2d(hidden, z_dim, 4, 2, 1),            # 32 -> 16
            # output: B x z_dim x 16 x 16
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    """
    Decoder mirrors encoder: 16x16 -> 128x128
    """
    def __init__(self, out_channels=3, hidden=DEFAULT_HIDDEN, z_dim=DEFAULT_Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden, 4, 2, 1),   # 16 -> 32
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden, hidden//2, 4, 2, 1), # 32 -> 64
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden//2, out_channels, 4, 2, 1), # 64 -> 128
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

class VectorQuantizerEMA(nn.Module):
    """
    EMA variant VQ (stable codebook updates).
    Stores embedding as buffer: shape (embedding_dim, num_embeddings)
    """
    def __init__(self, num_embeddings=DEFAULT_NUM_EMBEDDINGS, embedding_dim=DEFAULT_Z_DIM,
                 commitment_cost=DEFAULT_COMMITMENT, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, z):
        # z: B x D x H x W -> flatten to N x D
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # B x H x W x D
        flat_z = z_perm.view(-1, self.embedding_dim)  # N x D

        emb_t = self.embedding.t()  # K x D
        distances = (flat_z.pow(2).sum(1, keepdim=True)
                     - 2 * flat_z @ emb_t.t()
                     + emb_t.pow(2).sum(1).unsqueeze(0))  # N x K

        encoding_indices = torch.argmin(distances, dim=1)  # N
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)  # N x K

        quantized = torch.matmul(encodings, self.embedding.t())  # N x D
        quantized = quantized.view(z_perm.shape)  # B x H x W x D
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # B x D x H x W

        if self.training:
            enc_sum = encodings.sum(0)  # K
            embed_sum = flat_z.t() @ encodings  # D x K

            self.cluster_size.data.mul_(self.decay).add_(enc_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum().clamp(min=1e-5)
            cluster_size = ((self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps)) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embed_normalized)

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized_out = z + (quantized - z).detach()
        encoding_indices = encoding_indices.view(z_perm.shape[0], z_perm.shape[1], z_perm.shape[2])  # B x H x W

        return quantized_out, loss, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden=DEFAULT_HIDDEN, z_dim=DEFAULT_Z_DIM,
                 num_embeddings=DEFAULT_NUM_EMBEDDINGS, commitment_cost=DEFAULT_COMMITMENT):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden=hidden, z_dim=z_dim)
        self.vq = VectorQuantizerEMA(num_embeddings=num_embeddings, embedding_dim=z_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(out_channels=in_channels, hidden=hidden, z_dim=z_dim)

    def forward(self, x):
        z_e = self.encoder(x)  # B x D x H x W (D=z_dim, H=16)
        z_q, vq_loss, indices = self.vq(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices

# -----------------------------
# Dataloaders, training, saving
# -----------------------------
def get_dataloaders(dataset='CIFAR10', image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH, num_workers=DEFAULT_NUM_WORKERS):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),  # to [-1,1]
    ])
    if dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        # Placeholder: user can replace with ImageFolder or custom dataset
        trainset = torchvision.datasets.ImageFolder(root=dataset, transform=transform)
        testset = None
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if testset is not None else None
    return train_loader, test_loader

def save_checkpoint(model, opt, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'opt_state': opt.state_dict(), 'step': step}, path)

def load_checkpoint(model, path, map_location=None):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck['model_state'])
    return ck.get('step', 0)

def train(num_epochs=DEFAULT_EPOCHS,
          dataset='CIFAR10',
          image_size=DEFAULT_IMAGE_SIZE,
          batch_size=DEFAULT_BATCH,
          lr=3e-4,
          device='cuda' if torch.cuda.is_available() else 'cpu',
          num_embeddings=DEFAULT_NUM_EMBEDDINGS,
          z_dim=DEFAULT_Z_DIM,
          hidden=DEFAULT_HIDDEN,
          out_dir='./checkpoints',
          num_workers=DEFAULT_NUM_WORKERS):
    model = VQVAE(in_channels=3, hidden=hidden, z_dim=z_dim, num_embeddings=num_embeddings).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = get_dataloaders(dataset=dataset, image_size=image_size, batch_size=batch_size, num_workers=num_workers)

    l1_loss = nn.L1Loss()
    global_step = 0
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            recon, vq_loss, indices = model(imgs)
            recon_loss = l1_loss(recon, imgs)
            loss = recon_loss + vq_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            global_step += 1
            if global_step % 500 == 0:
                with torch.no_grad():
                    sample = torch.cat([imgs[:8], (recon[:8]).clamp(-1,1)], dim=0)
                    sample = (sample + 1) * 0.5  # to [0,1]
                    os.makedirs('samples', exist_ok=True)
                    save_image(sample, f'samples/sample_{global_step}.png', nrow=8)
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1e-9)})
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        ckpt_path = os.path.join(out_dir, f'vqvae_epoch{epoch+1}.pth')
        save_checkpoint(model, opt, global_step, ckpt_path)
    return model

# -----------------------------
# Encoding / decoding helpers
# -----------------------------
def indices_to_bitstring(indices: np.ndarray, K: int) -> str:
    b = math.ceil(math.log2(K))
    bits = ''.join(format(int(x), f'0{b}b') for x in indices.flatten().tolist())
    return bits

def bitstring_to_indices(bits: str, H: int, W: int, K: int) -> np.ndarray:
    b = math.ceil(math.log2(K))
    assert len(bits) == H * W * b, f"bitstring length {len(bits)} != expected {H*W*b}"
    idxs = [int(bits[i:i + b], 2) for i in range(0, len(bits), b)]
    arr = np.array(idxs, dtype=np.int64).reshape(H, W)
    return arr

def encode_image_to_string(model: VQVAE, image_tensor: torch.Tensor, device='cuda' if torch.cuda.is_available() else 'cpu',
                           gzip_compress=True, add_checksum=True):
    """
    image_tensor: torch tensor (C,H,W) with values in [-1,1]
    """
    model.eval()
    with torch.no_grad():
        img = image_tensor.unsqueeze(0).to(device)
        z = model.encoder(img)  # 1 x D x H x W (H=16)
        _, _, indices = model.vq(z)  # 1 x H x W
        inds = indices[0].cpu().numpy().astype(np.int64)
        H, W = inds.shape
        K = model.vq.num_embeddings
        bitstring = indices_to_bitstring(inds, K)

        if gzip_compress:
            # pack bitstring -> bytes -> gzip -> base62
            byte_len = (len(bitstring) + 7) // 8
            if byte_len > 0:
                b_int = int(bitstring, 2)
                bytearr = b_int.to_bytes(byte_len, 'big')
            else:
                bytearr = b''
            gz = gzip.compress(bytearr)
            gz_int = int.from_bytes(gz, 'big') if len(gz) > 0 else 0
            s = int_to_base62(gz_int)
            out = 'G' + s
        else:
            out = bits_to_base62(bitstring)

        if add_checksum:
            out = add_checksum_to_string(out)
        return out

def decode_string_to_image(model: VQVAE, s: str, device='cuda' if torch.cuda.is_available() else 'cpu',
                           gzip_compress=True, expect_checksum=True):
    model.eval()
    if expect_checksum:
        core, ok = verify_and_strip_checksum(s)
        if not ok:
            raise ValueError("Checksum failed or string corrupted")
        s = core

    if gzip_compress:
        if not s.startswith('G'):
            raise ValueError("Expected gzip-prefixed string ('G')")
        num_part = s[1:]
        gz_int = base62_to_int(num_part) if num_part != '' else 0
        byte_len = (gz_int.bit_length() + 7) // 8
        gz_bytes = gz_int.to_bytes(byte_len, 'big') if byte_len > 0 else b''
        bytearr = gzip.decompress(gz_bytes) if len(gz_bytes) > 0 else b''
        if len(bytearr) == 0:
            bitstring = ''
        else:
            b_int = int.from_bytes(bytearr, 'big')
            bitstring = bin(b_int)[2:].zfill(len(bytearr) * 8)
    else:
        bitstring = base62_to_bits(s)

    # infer H,W from model.encoder output shape using dummy input
    dummy = torch.zeros(1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE).to(device)
    with torch.no_grad():
        zshape = model.encoder(dummy).shape  # 1 x D x H x W
    H, W = zshape[2], zshape[3]
    K = model.vq.num_embeddings
    b_per_idx = math.ceil(math.log2(K))
    total_bits = H * W * b_per_idx
    if len(bitstring) < total_bits:
        bitstring = bitstring.zfill(total_bits)
    elif len(bitstring) > total_bits:
        bitstring = bitstring[-total_bits:]
    inds = bitstring_to_indices(bitstring, H, W, K)

    # map indices -> embeddings (embedding: D x K)
    emb = model.vq.embedding  # D x K (torch tensor as buffer)
    emb_t = emb.t().contiguous()  # K x D
    flat = emb_t[inds.flatten()]  # (H*W, D) as torch tensor
    # ensure we have torch tensor on correct device
    if isinstance(flat, np.ndarray):
        z = torch.tensor(flat, dtype=torch.float32, device=device)
    else:
        z = flat.to(device).float()
    # reshape to (1, D, H, W)
    z = z.view(1, H, W, -1).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        recon = model.decoder(z)
        recon = recon.clamp(-1, 1)
    return recon.cpu().squeeze(0)

# -----------------------------
# Demo / quick-run helper
# -----------------------------
def demo_train_and_encode():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    # training parameters (tunable)
    model = train(num_epochs=3, dataset='CIFAR10', image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH,
                  device=device, num_embeddings=DEFAULT_NUM_EMBEDDINGS, z_dim=DEFAULT_Z_DIM, hidden=DEFAULT_HIDDEN,
                  out_dir='./checkpoints/demo', num_workers=DEFAULT_NUM_WORKERS)
    # save final ckpt
    opt_dummy = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(model, opt_dummy, 0, './checkpoints/demo_final.pth')

    # load a test image and encode/decode
    transform = T.Compose([T.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    img, _ = testset[0]
    s = encode_image_to_string(model, img, device=device, gzip_compress=True, add_checksum=True)
    print("Encoded string (len={}): {}".format(len(s), s))
    recon = decode_string_to_image(model, s, device=device, gzip_compress=True, expect_checksum=True)
    # save visuals
    orig = (img + 1) * 0.5
    rec_vis = (recon + 1) * 0.5
    os.makedirs('demos', exist_ok=True)
    save_image(orig, 'demos/orig.png')
    save_image(rec_vis, 'demos/recon.png')
    print("Saved demos/orig.png and demos/recon.png")

# If executed directly, don't auto-run training (prevents spawn issues). Use run_demo.py to run demo_train_and_encode().
if __name__ == "__main__":
    print("Module loaded. Use demo_train_and_encode() from run_demo.py to execute a demo safely on Windows.")
