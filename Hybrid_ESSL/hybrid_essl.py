# hybrid_ssl_train_runs.py
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import RandomCrop3d, NSDataset
from transformations import NSTransforms
import torchvision.models.resnet as resnet

# ============================================================
# Config
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CROP_T, CROP_X, CROP_Y = 16, 128, 128
BATCH_SIZE = 16
EPOCHS = 60
LR = 3e-4

Z_TOTAL = 512
Z_INV = 256
Z_EQ = 256
assert Z_INV + Z_EQ == Z_TOTAL

LAMBDA_EQ = 0.7
DATA_ROOT = "/mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs/datasets"

OUT_DIR = "results_hybrid"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Models
# ============================================================
class Encoder(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.backbone = resnet.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            5 * T, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class Projector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z):
        return self.net(z)

# ============================================================
# Losses
# ============================================================
def off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg(z1, z2, sim=25, std=25, cov=1):
    z1 = z1 - z1.mean(0)
    z2 = z2 - z2.mean(0)

    sim_loss = F.mse_loss(z1, z2)
    std_loss = torch.mean(F.relu(1 - z1.std(0))) + torch.mean(F.relu(1 - z2.std(0)))

    cov1 = (z1.T @ z1) / (z1.shape[0] - 1)
    cov2 = (z2.T @ z2) / (z2.shape[0] - 1)

    cov_loss = off_diagonal(cov1).pow(2).sum() + off_diagonal(cov2).pow(2).sum()
    return sim * sim_loss + std * std_loss + cov * cov_loss


def care_loss(z1, z2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    G1 = z1 @ z1.T
    G2 = z2 @ z2.T

    mask = ~torch.eye(z1.size(0), dtype=torch.bool, device=z1.device)
    return ((G1[mask] - G2[mask]) ** 2).mean()

# ============================================================
# Transform samplers
# ============================================================
def sample_g_inv():
    g = torch.zeros(9)
    g[1] = np.random.uniform(-1.0, 1.0)  # x translation
    g[2] = np.random.uniform(-1.0, 1.0)  # y translation
    return g.float()

def sample_g_eq():
    g = torch.zeros(9)
    k = np.random.choice([3, 4, 5, 6])
    g[k] = np.random.uniform(-0.1, 0.1)
    return g.float()

# ============================================================
# Dataset
# ============================================================
class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        self.ds = NSDataset(
            data_root=DATA_ROOT,
            mode=mode,
            crop_size=(CROP_T, CROP_X, CROP_Y),
            transforms_strength=[0.0] * 9,
            steps=2,
            order=2,
            size=26624,
        )
        self.crop = RandomCrop3d((CROP_T, CROP_X, CROP_Y))
        self.group = NSTransforms()

    def __len__(self):
        return len(self.ds)

    def _apply(self, g, sample):
        t, x, y, vx, vy = sample
        t2, x2, y2, vx2, vy2 = self.group.apply(g, t, x, y, vx, vy)
        return self.crop(
            torch.stack((t2, x2, y2, vx2, vy2))
        ).flatten(0, 1)

    def __getitem__(self, idx):
        # x1 is already [5*T, H, W]
        x1, _, _ = self.ds[idx]

        # reshape correctly
        sample = x1.view(5, CROP_T, CROP_X, CROP_Y)

        x_inv1 = self._apply(sample_g_inv(), sample)
        x_inv2 = self._apply(sample_g_inv(), sample)

        x_eq1  = self._apply(sample_g_eq(), sample)
        x_eq2  = self._apply(sample_g_eq(), sample)

        return x_inv1, x_inv2, x_eq1, x_eq2



# ============================================================
# Training
# ============================================================
def train():
    encoder = Encoder(CROP_T).to(DEVICE)
    projector = Projector(Z_INV).to(DEVICE)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()), lr=LR
    )

    loader = DataLoader(
        HybridDataset("train"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    for epoch in range(EPOCHS):
        encoder.train()
        losses = []

        for xinv1, xinv2, xeq1, xeq2 in tqdm(loader, desc=f"Epoch {epoch}"):
            xinv1, xinv2 = xinv1.to(DEVICE), xinv2.to(DEVICE)
            xeq1, xeq2 = xeq1.to(DEVICE), xeq2.to(DEVICE)

            z = encoder(torch.cat([xinv1, xinv2, xeq1, xeq2], dim=0))
            z1, z2, zeq1, zeq2 = z.chunk(4)

            loss_inv = vicreg(
                projector(z1[:, :Z_INV]),
                projector(z2[:, :Z_INV]),
            )

            loss_eq = care_loss(zeq1[:, Z_INV:], zeq2[:, Z_INV:])
            loss = loss_inv + LAMBDA_EQ * loss_eq

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"[Epoch {epoch}] loss={np.mean(losses):.4f}")

    torch.save({"encoder": encoder.state_dict()}, f"{OUT_DIR}/hybrid_encoder.pt")


if __name__ == "__main__":
    train()
