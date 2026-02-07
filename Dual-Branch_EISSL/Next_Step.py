import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from utils import get_eval_loader_ns
from pde_inv_equiv import Encoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "/mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs/datasets"

CROP_T = 16
CROP_X = 128
CROP_Y = 128

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

OUT_DIR = Path("one_step_results")
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)



class UNetOneStep(nn.Module):
    def __init__(self, z_dim=512):
        super().__init__()

        self.enc1 = nn.Sequential(
        nn.Conv2d(2 + z_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
        )

        self.out = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, u_t, z):

        B, _, H, W = u_t.shape
        z = z[:, :, None, None].expand(-1, -1, H, W)
        x = torch.cat([u_t, z], dim=1)

        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h = self.dec1(h2)

        return self.out(h)


def save_plot(u_gt, u_pred, idx):
    """
    u_gt, u_pred: [2, H, W]
    """
    vx_gt = u_gt[0].cpu().numpy()
    vx_pr = u_pred[0].cpu().numpy()
    err = np.linalg.norm(u_gt.cpu().numpy() - u_pred.cpu().numpy(), axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(vx_gt, cmap="viridis")
    axs[0].set_title("GT $v_x(t+1)$")

    axs[1].imshow(vx_pr, cmap="viridis")
    axs[1].set_title("Pred $v_x(t+1)$")

    axs[2].imshow(err, cmap="inferno")
    axs[2].set_title("Error |v|")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"one_step_{idx:03d}.png")
    plt.close()



encoder = Encoder(CROP_T).to(DEVICE)
ckpt = torch.load("care_encoder_big_equiv.pt", map_location=DEVICE)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False



solver = UNetOneStep(z_dim=512).to(DEVICE)
opt = torch.optim.Adam(solver.parameters(), lr=LR)



train_loader = get_eval_loader_ns(
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    mode="train",
    crop_size=(CROP_T, CROP_X, CROP_Y),
    num_workers=8,
)




for epoch in range(EPOCHS):
    losses = []

    for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x = x.to(DEVICE)

        B, _, H, W = x.shape
        xt = x.view(B, CROP_T, 5, H, W)

        u_t = xt[:, -2, 3:5]     # veloc at t
        u_tp1 = xt[:, -1, 3:5]   # velo at t+1

        with torch.no_grad():
            z = encoder(x)

        du = solver(u_t, z)
        u_pred = u_t + du

        loss = F.mse_loss(u_pred, u_tp1)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    print(f"[Epoch {epoch}] Train MSE: {np.mean(losses):.6f}")



test_loader = get_eval_loader_ns(
    data_root=DATA_ROOT,
    batch_size=1,
    mode="test",
    crop_size=(CROP_T, CROP_X, CROP_Y),
    num_workers=4,
)

solver.eval()
mse_list = []

print("\none-step prediction")

with torch.no_grad():
    for idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.to(DEVICE)

        B, _, H, W = x.shape
        xt = x.view(B, CROP_T, 5, H, W)

        u_t = xt[:, -2, 3:5]
        u_tp1 = xt[:, -1, 3:5]

        z = encoder(x)
        du = solver(u_t, z)
        u_pred = u_t + du

        mse = F.mse_loss(u_pred, u_tp1)
        mse_list.append(mse.item())

        if idx < 5:
            save_plot(u_tp1[0], u_pred[0], idx)

print(f"One-step TEST MSE: {np.mean(mse_list):.6f}")
print(f"Plots saved to: {PLOT_DIR}")
