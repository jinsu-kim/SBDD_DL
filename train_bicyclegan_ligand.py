import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from networks_gan import G_Unet_add_all3D, E_3DNLayers, D_N3Dv1LayersMulti


@dataclass
class TrainConfig:
    protein_path: str
    ligand_path: str
    protein_rev_path: str
    ligand_rev_path: str
    output_dir: str = "results"
    epochs: int = 100
    batch_size: int = 1
    lr: float = 2e-4
    min_lr: float = 1e-5
    beta1: float = 0.5
    begin_anneal: int = 50
    lambda_kl: float = 0.01
    lambda_l1: float = 10.0
    lambda_gan: float = 1.0
    lambda_gan2: float = 1.0
    lambda_z: float = 0.5
    nz: int = 8
    gan_mode: str = "lsgan"
    device: str = "cuda:0"
    num_workers: int = 0
    save_every: int = 25


class GANLoss(nn.Module):
    def __init__(self,
                 gan_mode: str,
                 target_real_label: float = 1.0,
                 target_fake_label: float = 0.0) -> None:

        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError(f"GAN mode [{gan_mode}] is not implemented")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def forward(self, predictions, target_is_real: bool):
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]

        losses = []
        for prediction in predictions:
            if self.gan_mode in {"lsgan", "vanilla"}:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == "wgangp":
                loss = -prediction.mean() if target_is_real else prediction.mean()
            else:
                raise RuntimeError(f"Unsupported GAN mode: {self.gan_mode}")
            losses.append(loss)

        return sum(losses), losses


class BicycleGAN(nn.Module):
    """Training wrapper for 3D BicycleGAN."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.G = G_Unet_add_all3D(norm_type="instance", nz=cfg.nz).to(self.device)
        self.E = E_3DNLayers(norm_type="instance", vaeLike=True).to(self.device)
        self.D = D_N3Dv1LayersMulti(norm_type="instance", input_nc=12, num_D=2).to(self.device)
        self.D2 = D_N3Dv1LayersMulti(norm_type="instance", input_nc=12, num_D=2).to(self.device)

        self.criterion_gan = GANLoss(cfg.gan_mode).to(self.device)
        self.criterion_l1 = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        self.optimizer_E = torch.optim.Adam(self.E.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        self.optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_E, self.optimizer_D, self.optimizer_D2]

    def set_input(self, p1: torch.Tensor, l1: torch.Tensor, p2: torch.Tensor, l2: torch.Tensor) -> None:
        self.real_P_encoded = p1.to(self.device, non_blocking=True)
        self.real_L_encoded = l1.to(self.device, non_blocking=True)
        self.real_P_random  = p2.to(self.device, non_blocking=True)
        self.real_L_random  = l2.to(self.device, non_blocking=True)

    def sample_z(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.cfg.nz, device=self.device)

    def encode(self, ligand_shape: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.E(ligand_shape)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self) -> None:
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_L_encoded)
        self.z_random = self.sample_z(self.real_P_encoded.size(0))

        self.fake_L_encoded = self.G(self.real_P_encoded, self.z_encoded)
        self.fake_L_random  = self.G(self.real_P_random, self.z_random)

        self.P_fake_encoded = torch.cat([self.real_P_encoded, self.fake_L_encoded], dim=1)
        self.P_fake_random  = torch.cat([self.real_P_random, self.fake_L_random], dim=1)
        self.P_real_encoded = torch.cat([self.real_P_encoded, self.real_L_encoded], dim=1)
        self.P_real_random  = torch.cat([self.real_P_random, self.real_L_random], dim=1)

        self.mu2, _ = self.E(self.fake_L_random)

    def generator_gan_loss(self, fake_pair: torch.Tensor, discriminator: nn.Module, weight: float) -> torch.Tensor:
        if weight <= 0.0:
            return torch.zeros((), device=self.device)
        pred_fake = discriminator(fake_pair)
        loss, _ = self.criterion_gan(pred_fake, True)
        return loss * weight

    def backward_EG(self) -> None:
        self.loss_G_GAN  = self.generator_gan_loss(self.P_fake_encoded, self.D, self.cfg.lambda_gan)
        self.loss_G_GAN2 = self.generator_gan_loss(self.P_fake_random, self.D2, self.cfg.lambda_gan2)
        self.loss_kl     = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * self.cfg.lambda_kl
        self.loss_G_L1   = self.criterion_l1(self.fake_L_encoded, self.real_L_encoded) * self.cfg.lambda_l1

        self.loss_G      = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def backward_D(self, discriminator: nn.Module, real_pair: torch.Tensor, fake_pair: torch.Tensor):
        pred_fake = discriminator(fake_pair.detach())
        pred_real = discriminator(real_pair)

        loss_D_fake, _ = self.criterion_gan(pred_fake, False)
        loss_D_real, _ = self.criterion_gan(pred_real, True)
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, loss_D_fake, loss_D_real

    def backward_G_alone(self) -> None:
        self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_random)) * self.cfg.lambda_z
        self.loss_z_L1.backward()

    def update_G_and_E(self) -> None:
        set_requires_grad([self.D, self.D2], False)

        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_E.zero_grad(set_to_none=True)
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()

        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_E.zero_grad(set_to_none=True)
        self.backward_G_alone()
        self.optimizer_G.step()

    def update_D(self) -> None:
        set_requires_grad([self.D, self.D2], True)

        self.optimizer_D.zero_grad(set_to_none=True)
        self.loss_D, self.loss_D_fake, self.loss_D_real = self.backward_D(
            self.D, self.P_real_encoded, self.P_fake_encoded
        )
        self.optimizer_D.step()

        self.optimizer_D2.zero_grad(set_to_none=True)
        self.loss_D2, self.loss_D2_fake, self.loss_D2_real = self.backward_D(
            self.D2, self.P_real_random, self.P_fake_random
        )
        self.optimizer_D2.step()

    def optimize_parameters(self) -> None:
        self.forward()
        self.update_G_and_E()
        self.update_D()

    def adjust_learning_rate(self, epoch: int) -> float:
        if self.cfg.begin_anneal == 0 or self.cfg.begin_anneal == self.cfg.epochs:
            lr = self.cfg.lr
        elif epoch > self.cfg.begin_anneal:
            progress = (epoch - self.cfg.begin_anneal) / max(1, self.cfg.epochs - self.cfg.begin_anneal)
            lr = max(self.cfg.min_lr, self.cfg.lr - (self.cfg.lr - self.cfg.min_lr) * progress)
        else:
            lr = self.cfg.lr

        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        return lr

    def loss_dict(self) -> Dict[str, float]:
        return {
            "G_GAN": tensor_to_float(self.loss_G_GAN),
            "G_GAN2": tensor_to_float(self.loss_G_GAN2),
            "G_L1": tensor_to_float(self.loss_G_L1),
            "KL": tensor_to_float(self.loss_kl),
            "z_L1": tensor_to_float(self.loss_z_L1),
            "D": tensor_to_float(self.loss_D),
            "D2": tensor_to_float(self.loss_D2),
        }


def set_requires_grad(nets: Iterable[nn.Module], requires_grad: bool) -> None:
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def tensor_to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def load_npy_tensor(path: str) -> torch.Tensor:
    arr = np.load(path)
    return torch.from_numpy(arr).float()


def build_dataloader(cfg: TrainConfig) -> DataLoader:
    p1 = load_npy_tensor(cfg.protein_path)
    l1 = load_npy_tensor(cfg.ligand_path)
    p2 = load_npy_tensor(cfg.protein_rev_path)
    l2 = load_npy_tensor(cfg.ligand_rev_path)

    n = min(len(p1), len(l1), len(p2), len(l2))
    if len({len(p1), len(l1), len(p2), len(l2)}) != 1:
        print(f"[Warning] Dataset lengths differ. Using first {n} samples.")
        p1, l1, p2, l2 = p1[:n], l1[:n], p2[:n], l2[:n]

    dataset = TensorDataset(p1, l1, p2, l2)
    pin_memory = torch.cuda.is_available() and cfg.device.startswith("cuda")
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )


def make_run_dir(base_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(model: BicycleGAN, save_dir: Path, epoch: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.G.state_dict(), save_dir / f"BG_G_epoch{epoch}.pt")
    torch.save(model.E.state_dict(), save_dir / f"BG_E_epoch{epoch}.pt")
    torch.save(model.D.state_dict(), save_dir / f"BG_D_epoch{epoch}.pt")
    torch.save(model.D2.state_dict(), save_dir / f"BG_D2_epoch{epoch}.pt")


def plot_losses(history: Dict[str, List[float]], save_path: Path) -> None:
    if not history or not history["G_GAN"]:
        return

    total = [
        sum(values)
        for values in zip(
            history["G_GAN"],
            history["G_GAN2"],
            history["G_L1"],
            history["KL"],
            history["z_L1"],
        )
    ]

    epochs = range(1, len(total) + 1)
    fig = plt.figure(figsize=(8, 6))
    plt.title(f"Total Loss\n{total[0]:.5g} --> {total[-1]:.5g}")
    plt.plot(epochs, history["G_GAN"], label="G_GAN")
    plt.plot(epochs, history["G_GAN2"], label="G_GAN2")
    plt.plot(epochs, history["G_L1"], label="G_L1")
    plt.plot(epochs, history["KL"], label="KL")
    plt.plot(epochs, history["z_L1"], label="z_L1")
    plt.plot(epochs, total, label="Total")
    plt.ylim(0.0, 1.2 * max(total))
    plt.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def train(cfg: TrainConfig) -> None:
    run_dir = make_run_dir(cfg.output_dir)
    save_dir = run_dir / "LiGANN_save"

    loader = build_dataloader(cfg)
    model = BicycleGAN(cfg)

    history: Dict[str, List[float]] = {
        "G_GAN": [],
        "G_GAN2": [],
        "G_L1": [],
        "KL": [],
        "z_L1": [],
        "D": [],
        "D2": [],
    }

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        epoch_start = time.time()

        for p1, l1, p2, l2 in loader:
            model.set_input(p1, l1, p2, l2)
            model.optimize_parameters()

        lr = model.adjust_learning_rate(epoch)
        losses = model.loss_dict()
        for key, value in losses.items():
            history[key].append(value)

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(model, save_dir, epoch)

        elapsed = time.time() - epoch_start
        print("#" * 60)
        print(f"Epoch {epoch}/{cfg.epochs} | lr={lr:.6g} | time={elapsed:.3f}s")
        print(
            " | ".join(
                f"{name}: {value:.5f}"
                for name, value in losses.items()
            )
        )
        print("#" * 60)

    plot_losses(history, run_dir / "loss_curve.png")
    save_checkpoint(model, save_dir, cfg.epochs)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train 3D BicycleGAN for ligand shape generation.")
    parser.add_argument("--protein_path", required=True)
    parser.add_argument("--ligand_path", required=True)
    parser.add_argument("--protein_rev_path", required=True)
    parser.add_argument("--ligand_rev_path", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--begin_anneal", type=int, default=50)
    parser.add_argument("--lambda_kl", type=float, default=0.01)
    parser.add_argument("--lambda_l1", type=float, default=10.0)
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_gan2", type=float, default=1.0)
    parser.add_argument("--lambda_z", type=float, default=0.5)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--gan_mode", default="lsgan", choices=["lsgan", "vanilla", "wgangp"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=25)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
