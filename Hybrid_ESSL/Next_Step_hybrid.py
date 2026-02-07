import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torchvision.models.resnet as resnet

from utils import get_loader_ns, get_eval_loader_ns


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CROP_T, CROP_X, CROP_Y = 16, 128, 128
BATCH_SIZE = 16
EPOCHS_TS = 40
EPOCHS_REG = 40
LR_TS = 1e-4
LR_REG = 1e-4

Z_TOTAL = 512
Z_INV = 256

DATA_ROOT = "/mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs/datasets"

OUT_DIR = Path("results")
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

CKPT_PATH = "hybrid_encoder.pt"


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

def unpack_spacetime(x, T):
    B, _, H, W = x.shape
    return x.view(B, T, 5, H, W)


def save_gt_pred_error(gt, pred, idx):
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    err = np.linalg.norm(pred - gt, axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(gt[0], cmap="viridis")
    axs[0].set_title("GT vx")

    axs[1].imshow(pred[0], cmap="viridis")
    axs[1].set_title("Pred vx")

    axs[2].imshow(err, cmap="inferno")
    axs[2].set_title("Error")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"timestep_{idx:03d}.png")
    plt.close()


def load_encoder():
    encoder = Encoder(CROP_T).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


class ConditionedUNet(nn.Module):
    def __init__(self, z_dim=512):
        super().__init__()
        self.z_proj = nn.Linear(z_dim, 64)
        self.net = nn.Sequential(
            nn.Conv2d(2 + 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),
        )

    def forward(self, u_t, z):
        B, _, H, W = u_t.shape
        zc = self.z_proj(z)[:, :, None, None].expand(-1, -1, H, W)
        return self.net(torch.cat([u_t, zc], dim=1))


class RegressionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)


def train_timestep(encoder):
    solver = ConditionedUNet(Z_TOTAL).to(DEVICE)
    opt = torch.optim.Adam(solver.parameters(), lr=LR_TS)

    loader = get_loader_ns(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        mode="train",
        crop_size=(CROP_T, CROP_X, CROP_Y),
        strengths=[0.0] * 9,
        steps=0,
        order=0,
        num_workers=8,
        dataset_size=26624,
    )

    for epoch in range(EPOCHS_TS):
        losses = []
        solver.train()

        for x, _, _ in tqdm(loader, desc=f"[TS] Epoch {epoch}"):
            x = x.to(DEVICE)
            xt = unpack_spacetime(x, CROP_T)

            u_t  = xt[:, -2, 3:5]
            u_gt = xt[:, -1, 3:5]

            with torch.no_grad():
                z = encoder(x)

            du = solver(u_t, z)
            u_pred = u_t + du

            loss = F.mse_loss(u_pred, u_gt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"[TS] Epoch {epoch} | Train MSE = {np.mean(losses):.6f}")

    torch.save(solver.state_dict(), OUT_DIR / "solver.pt")
    return solver


def evaluate_timestep(encoder, solver):
    solver.eval()

    loader = get_eval_loader_ns(
        data_root=DATA_ROOT,
        batch_size=1,
        mode="test",
        crop_size=(CROP_T, CROP_X, CROP_Y),
        num_workers=4,
    )

    mse_list = []

    with torch.no_grad():
        for idx, (x, _) in enumerate(tqdm(loader, desc="Evaluating TS")):
            x = x.to(DEVICE)
            xt = unpack_spacetime(x, CROP_T)

            u_t  = xt[:, -2, 3:5]
            u_gt = xt[:, -1, 3:5]

            z = encoder(x)
            du = solver(u_t, z)
            u_pred = u_t + du

            mse = F.mse_loss(u_pred, u_gt)
            mse_list.append(mse.item())

            if idx < 20:
                save_gt_pred_error(u_gt[0], u_pred[0], idx)

    print(f"\nOne-step TEST MSE: {np.mean(mse_list):.6f}")


def train_regression(encoder):
    reg = RegressionHead(Z_INV).to(DEVICE)
    opt = torch.optim.Adam(reg.parameters(), lr=LR_REG)

    loader = get_eval_loader_ns(
        data_root=DATA_ROOT,
        batch_size=128,
        mode="val",
        crop_size=(CROP_T, CROP_X, CROP_Y),
        num_workers=8,
    )

    for epoch in range(EPOCHS_REG):
        losses = []

        for x, b in loader:
            x, b = x.to(DEVICE), b.to(DEVICE)

            with torch.no_grad():
                z = encoder(x)
                z_inv = z[:, :Z_INV]

            pred = reg(z_inv)
            loss = F.mse_loss(pred, b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"[REG] Epoch {epoch} | Val MSE = {np.mean(losses):.6f}")

    torch.save(reg.state_dict(), OUT_DIR / "regressor.pt")
    return reg


def evaluate_regression(encoder, reg, split="test"):
    encoder.eval()
    reg.eval()

    loader = get_eval_loader_ns(
        data_root=DATA_ROOT,
        batch_size=128,
        mode=split,
        crop_size=(CROP_T, CROP_X, CROP_Y),
        num_workers=8,
    )

    mse_list = []

    with torch.no_grad():
        for x, b in tqdm(loader, desc=f"Evaluating REG ({split})"):
            x, b = x.to(DEVICE), b.to(DEVICE)

            z = encoder(x)
            z_inv = z[:, :Z_INV]

            pred = reg(z_inv)
            mse = F.mse_loss(pred, b)
            mse_list.append(mse.item())

    mean_mse = np.mean(mse_list)
    print(f"\n[REG] {split.upper()} MSE : {mean_mse:.6f}")
    print(f"[REG] {split.upper()} RMSE: {np.sqrt(mean_mse):.6f}")


if __name__ == "__main__":
    encoder = load_encoder()

    print("\nTraining one-step predictor")
    solver = train_timestep(encoder)

    print("\nEvaluating one-step predictor")
    evaluate_timestep(encoder, solver)

    print("\nTraining regression head")
    reg = train_regression(encoder)


    print("\nEvaluating regression TEST")
    evaluate_regression(encoder, reg, split="test")
