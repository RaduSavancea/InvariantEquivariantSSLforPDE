import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import RandomCrop3d, get_eval_loader_ns
from transformations import NSTransforms
import torchvision.models.resnet as resnet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CROP_T = 16
CROP_X = 128
CROP_Y = 128

BATCH_SIZE = 16
EPOCHS_SSL = 60
EPOCHS_REG = 40

LR_SSL = 3e-4
LR_REG = 1e-4

STRENGTHS = [
    0.0,
    1.0,   # x transl
    1.0,   # y tansl
    0.1,   # scale
    0.1,   # rotat
]

DATA_ROOT = "/mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs/datasets"



#encoder resnet bacbone maybe a vit backbone with tokens physiscs awre is better

class Encoder(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.net = resnet.resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(
            5 * T, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.net.fc = nn.Identity()

    def forward(self, x):
        return self.net(x)


class Projector(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z):
        return self.net(z)

#sasme as paper
class RegressionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Linear(dim, 1)

    def forward(self, z):
        return self.net(z).squeeze(-1)


# same as lie paper
def off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg(z1, z2, sim=25, std=25, cov=1):
    z1 = z1 - z1.mean(0)
    z2 = z2 - z2.mean(0)

    sim_loss = F.mse_loss(z1, z2)

    std_loss = torch.mean(F.relu(1 - z1.std(0))) + \
               torch.mean(F.relu(1 - z2.std(0)))

    cov1 = (z1.T @ z1) / (z1.shape[0] - 1)
    cov2 = (z2.T @ z2) / (z2.shape[0] - 1)

    cov_loss = off_diagonal(cov1).pow(2).sum() + \
               off_diagonal(cov2).pow(2).sum()

    return sim * sim_loss + std * std_loss + cov * cov_loss

#Look at care loss same almost as paper
def care_equivariance_loss(z1, z2):
 
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    G1 = z1 @ z1.T
    G2 = z2 @ z2.T

    bs = z1.size(0)
    mask = ~torch.eye(bs, dtype=torch.bool, device=z1.device)

    return ((G1[mask] - G2[mask]) ** 2).mean()

#
class ESSLDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, mode="train"):
        from utils import NSDataset

        self.ds = NSDataset(
            data_root=data_root,
            mode=mode,
            crop_size=(CROP_T, CROP_X, CROP_Y),
            transforms_strength=STRENGTHS,
            steps=2,
            order=2,
            size=26624,
        )

        self.crop = RandomCrop3d((CROP_T, CROP_X, CROP_Y))
        self.group = NSTransforms()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # invariant views
        x1, x2, _ = self.ds[idx]

        sample = self.ds[idx][0].view(5, CROP_T, CROP_X, CROP_Y)
        x, y, t, vx, vy = sample

        # sample two augmentations 
        def sample_g():
            return torch.tensor([
                0.0,
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1),
            ], dtype=torch.float32)

        g1, g2 = sample_g(), sample_g()

        t1, x1p, y1p, vx1, vy1 = self.group.apply(g1, t, x, y, vx, vy)
        t2, x2p, y2p, vx2, vy2 = self.group.apply(g2, t, x, y, vx, vy)

        x_eq1 = self.crop(torch.stack((x1p, y1p, t1, vx1, vy1))).flatten(0, 1)
        x_eq2 = self.crop(torch.stack((x2p, y2p, t2, vx2, vy2))).flatten(0, 1)

        return x1, x2, x_eq1, x_eq2



def train_essl():
    encoder = Encoder(CROP_T).to(DEVICE)
    projector = Projector().to(DEVICE)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=LR_SSL,
    )

    loader = DataLoader(
        ESSLDataset(DATA_ROOT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    for epoch in range(EPOCHS_SSL):
        losses = []

        for x1, x2, xeq1, xeq2 in tqdm(loader, desc=f"CARE Epoch {epoch}"):
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            xeq1, xeq2 = xeq1.to(DEVICE), xeq2.to(DEVICE)

            # invariant loss
            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))
            loss_inv = vicreg(z1, z2)

            # equivariant CARE loss
            zeq1 = encoder(xeq1)
            zeq2 = encoder(xeq2)
            loss_eq = care_equivariance_loss(zeq1, zeq2)

            loss = loss_inv + 0.8 * loss_eq

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"[CARE] Epoch {epoch} | Loss {np.mean(losses):.4f}")

    torch.save({"encoder": encoder.state_dict()}, "care_encoder_big_equiv.pt")
    return encoder


def train_and_eval_regression(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    reg = RegressionHead().to(DEVICE)
    opt = torch.optim.Adam(reg.parameters(), lr=LR_REG)

    train_loader = get_eval_loader_ns(
        DATA_ROOT, batch_size=128, mode="val",
        crop_size=(CROP_T, CROP_X, CROP_Y), num_workers=8
    )

    for epoch in range(EPOCHS_REG):
        losses = []
        for x, b in train_loader:
            x, b = x.to(DEVICE), b.to(DEVICE)
            with torch.no_grad():
                z = encoder(x)
            loss = F.mse_loss(reg(z), b)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"[REG] Epoch {epoch} | MSE {np.mean(losses):.6f}")



if __name__ == "__main__":
    encoder = train_essl()
    train_and_eval_regression(encoder)
