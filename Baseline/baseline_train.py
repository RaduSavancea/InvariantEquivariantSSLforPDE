#Baseline Variant; Follows the Paper 


import numpy as np
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


from utils import (
    get_loader_ns,
    get_eval_loader_ns,
    off_diagonal,
    RangeSigmoid,
    log_stats,
    log,
)


# CARE equivariant loss; using gram matrix 

def get_diagonal_mask(bs, device):
    mask = torch.ones((bs, bs), dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)
    return mask


def care_equivariance_loss(z1, z2):
    """
    Enforces existence of rho(g) such that:
    f(gx) = rho(g) f(x)
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    bs = z1.shape[0]
    G1 = z1 @ z1.T
    G2 = z2 @ z2.T

    mask = get_diagonal_mask(bs, z1.device)
    return ((G1[mask] - G2[mask]) ** 2).mean()


# Projector (same as Meta)

def Projector(embedding, dimensions):
    mlp_spec = f"{embedding}-{dimensions}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


# VICReg + CARE equivariant 


class VICRegEquivariant(nn.Module):
    def __init__(
        self,
        sim_coeff,
        cov_coeff,
        std_coeff,
        equiv_coeff,
        inv_coeff,
        batch_size,
        mlp,
        n_time_steps,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_features = int(mlp.split("-")[-1])

        import torchvision.models.resnet as resnet
        self.backbone = resnet.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()

        # 5 channels per timestep: (x, y, t, vx, vy)
        self.backbone.conv1 = nn.Conv2d(
            n_time_steps * 5,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias,
        )

        with torch.no_grad():
            _, out_dim = self.backbone(
                torch.zeros(1, n_time_steps * 5, 224, 224)
            ).shape

        self.projector = Projector(out_dim, mlp)

        self.sim_coeff = sim_coeff      
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff
        self.equiv_coeff = equiv_coeff
        self.inv_coeff = inv_coeff

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        # CARE equivariance loss
        equiv_loss = care_equivariance_loss(z1, z2)

        # Optional weak invariance (locality)
        inv_loss = F.mse_loss(z1, z2)

        # VICReg variance
        z1c = z1 - z1.mean(dim=0)
        z2c = z2 - z2.mean(dim=0)

        std1 = torch.sqrt(z1c.var(dim=0) + 1e-4)
        std2 = torch.sqrt(z2c.var(dim=0) + 1e-4)

        std_loss = (
            torch.mean(F.relu(1 - std1)) +
            torch.mean(F.relu(1 - std2))
        ) / 2

        # VICReg 
        cov1 = (z1c.T @ z1c) / (self.batch_size - 1)
        cov2 = (z2c.T @ z2c) / (self.batch_size - 1)

        cov_loss = (
            off_diagonal(cov1).pow(2).sum() +
            off_diagonal(cov2).pow(2).sum()
        ) / self.num_features

        #here can the equiv part be deactivated but I also tried just for probing to add it, rest is vic
        loss = (
            self.equiv_coeff * equiv_loss
            + self.inv_coeff * inv_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss


# Linear regression head like in papeper

def RegressionHead(embedding, dimensions):
    layers = [
        nn.Linear(embedding, dimensions),
        RangeSigmoid(max=0.6, min=0.1)
    ]
    return nn.Sequential(*layers)


# Training + eval


def train(args):
    device = "cuda"

    train_loader = get_loader_ns(
        data_root=args.data_root,
        batch_size=args.batch_size,
        mode="train",
        crop_size=(args.crop_t, args.crop_x, args.crop_y),
        strengths=args.strengths,
        steps=args.trotter_steps,
        order=args.trotter_order,
        num_workers=args.num_workers,
        dataset_size=args.dataset_size,
    )

    eval_train_loader = get_eval_loader_ns(
        data_root=args.data_root,
        batch_size=args.batch_size_eval,
        mode="val",
        crop_size=(args.crop_t, args.crop_x, args.crop_y),
        num_workers=args.num_workers,
    )

    eval_test_loader = get_eval_loader_ns(
        data_root=args.data_root,
        batch_size=args.batch_size_eval,
        mode="test",
        crop_size=(args.crop_t, args.crop_x, args.crop_y),
        num_workers=args.num_workers,
    )


    model = VICRegEquivariant(
        sim_coeff=0.0,
        cov_coeff=1.0,
        std_coeff=25.0,
        equiv_coeff=1.0,
        inv_coeff=0.05,
        batch_size=args.batch_size,
        mlp=args.mlp,
        n_time_steps=args.crop_t,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ssl_lr)

    writer = SummaryWriter(log_dir=args.logging_folder)
    start_time = time.time()


    # SSL training

    for epoch in range(args.epochs):
        model.train()
        for x1, x2, _ in tqdm(train_loader):
            x1 = x1.cuda()
            x2 = x2.cuda()

            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch}] SSL loss: {loss.item():.4f}")

#model save

    ckpt_dir = Path(args.logging_folder) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "projector": model.projector.state_dict(),
            "args": vars(args),
        },
        ckpt_dir / "encoder_ssl.pt",
    )

    print(f"Saved SSL encoder to {ckpt_dir / 'encoder_base.pt'}")


    # Linear probe (buoyancy)
    
    model.eval()
    regressor = RegressionHead(512, 1).cuda()
    opt = torch.optim.Adam(regressor.parameters(), lr=args.eval_lr)

    for i in range(30):
        for x, b in eval_train_loader:
            x = x.cuda()
            b = b.cuda()

            with torch.no_grad():
                z = model.backbone(x)

            pred = regressor(z).flatten()
            loss = F.mse_loss(pred, b)

            opt.zero_grad()
            loss.backward()
            opt.step()

    print("Linear probe training complete.")

    torch.save(
        {
            "regressor": regressor.state_dict(),
        },
        ckpt_dir / "linear_probe.pt",
    )

    print(f"Saved linear probe to {ckpt_dir / 'linear_probe.pt'}")





class Args:
    data_root = "/mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs/datasets"
    logging_folder = "./runs/equiv_ns"
    mlp = "512-512-512"

    crop_t = 16
    crop_x = 128
    crop_y = 128

    batch_size = 64
    batch_size_eval = 128
    epochs = 100

    ssl_lr = 3e-4
    eval_lr = 7e-4


    num_workers = 8
    dataset_size = 26624

    trotter_steps = 2
    trotter_order = 2

    strengths = [0.0, 1.0, 1.0, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01]

if __name__ == "__main__":
    os.makedirs("./runs/equiv_ns", exist_ok=True)
    train(Args())
