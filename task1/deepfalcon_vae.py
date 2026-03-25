"""
DeepFALCON GSoC 2026 – Common Task 1
Variational Auto-Encoder for Quark/Gluon Jet Events
Three-channel images: ECAL (125x125), HCAL (125x125), Tracks (125x125)

Fixes vs original:
  - Auto-calculates encoder flat_dim (fixes mat1/mat2 shape crash)
  - Lazy HDF5 loading for large datasets (fixes 24GB MemoryError)
  - RAM loading for small datasets (much faster per-epoch)
  - num_workers auto-detected (fixes Windows multiprocessing crash)
  - pin_memory only when CUDA available
  - Early stopping added
  - Python 3.9 compatible type hints
  - MSE / MAE / PSNR evaluation metrics added
  - tqdm replaced with clean print progress (no ipywidgets needed)

Usage:
  python deepfalcon_vae.py --data path/to/file.hdf5 --epochs 30 --max_samples 5000
"""

# ─────────────────────────────────────────────
# 0. Imports & reproducibility
# ─────────────────────────────────────────────
import os, random, time, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import h5py

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Windows multiprocessing fix — always use 0 workers on Windows
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
PIN_MEMORY  = torch.cuda.is_available()

print(f"Device      : {DEVICE}")
print(f"num_workers : {NUM_WORKERS}")
print(f"pin_memory  : {PIN_MEMORY}")


# ─────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────
class JetImageDataset(Dataset):
    """
    Smart loader — uses RAM when max_samples is small (≤ 20k),
    lazy HDF5 per-sample loading when large to avoid MemoryError.

    Supports both key formats:
      'X_jets' / 'X' / 'jetImage'  for images
      'y'      / 'jetLabel'        for labels
    """

    RAM_THRESHOLD = 20_000   # below this → load into RAM (fast)

    def __init__(self, filepath, max_samples=None):
        super().__init__()
        self.filepath = filepath
        self.lazy = False

        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            print(f"HDF5 keys: {keys}")

            # Auto-detect key names
            self.x_key = ("X_jets" if "X_jets" in keys
                          else ("X" if "X" in keys else "jetImage"))
            self.y_key = "y" if "y" in keys else "jetLabel"

            total = f[self.x_key].shape[0]
            self.n = min(max_samples, total) if max_samples else total
            print(f"Total events in file: {total:,}  |  Using: {self.n:,}")

            # Always load labels — they're tiny
            self.labels = torch.tensor(
                f[self.y_key][:self.n], dtype=torch.long)

            if self.n <= self.RAM_THRESHOLD:
                # ── Load everything into RAM ────────────────────────────
                print("Loading into RAM...")
                X = f[self.x_key][:self.n]
                if X.ndim == 4 and X.shape[-1] == 3:
                    X = X.transpose(0, 3, 1, 2)
                X = X.astype(np.float32)
                X = np.log1p(X)
                for c in range(3):
                    p99 = np.percentile(X[:, c], 99)
                    if p99 > 0:
                        X[:, c] = np.clip(X[:, c] / p99, 0, 1)
                self.data = torch.tensor(X, dtype=torch.float32)
                print(f"Loaded {self.n:,} events into RAM. "
                      f"Shape: {tuple(self.data.shape[1:])}")
            else:
                # ── Lazy loading — compute percentiles on small sample ──
                print("Dataset too large for RAM — using lazy HDF5 loading.")
                self.lazy = True
                sample = f[self.x_key][:min(2000, self.n)]
                if sample.ndim == 4 and sample.shape[-1] == 3:
                    sample = sample.transpose(0, 3, 1, 2)
                sample = np.log1p(sample.astype(np.float32))
                self.p99 = [float(np.percentile(sample[:, c], 99))
                            for c in range(3)]
                print(f"Per-channel 99th percentiles: "
                      f"{[round(p,4) for p in self.p99]}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if not self.lazy:
            return self.data[idx], self.labels[idx]
        # Lazy: open file and read one event
        with h5py.File(self.filepath, "r") as f:
            x = f[self.x_key][idx]
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.transpose(2, 0, 1)
        x = np.log1p(x.astype(np.float32))
        for c in range(3):
            if self.p99[c] > 0:
                x[c] = np.clip(x[c] / self.p99[c], 0, 1)
        return torch.tensor(x, dtype=torch.float32), self.labels[idx]


# ─────────────────────────────────────────────
# 2. VAE Architecture
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, base_ch=32):
        super().__init__()
        B = base_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, B,   4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(B),

            nn.Conv2d(B,   B*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(B*2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(B*2),

            nn.Conv2d(B*2, B*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(B*4),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(B*4),

            nn.Conv2d(B*4, B*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(B*8),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(B*8),

            nn.Conv2d(B*8, B*8, 3, stride=2, padding=1),
            nn.BatchNorm2d(B*8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Auto-calculate flat dim with a dummy forward pass ──────────
        # This is the key fix — instead of hardcoding B*8*4*4 which
        # assumes a specific spatial size, we let PyTorch compute it.
        with torch.no_grad():
            dummy    = torch.zeros(1, in_channels, 125, 125)
            out      = self.encoder(dummy)
            self.spatial_shape = out.shape[2:]        # e.g. (3, 3) or (4, 4)
            self.flat_dim      = int(out.flatten(1).shape[1])
        print(f"Encoder spatial output: {self.spatial_shape}  "
              f"flat_dim: {self.flat_dim}")

        self.fc_mu      = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=256, base_ch=32,
                 spatial_shape=None, flat_dim=None):
        super().__init__()
        B = base_ch
        self.base_ch = B

        # Use spatial_shape passed from encoder (guaranteed to match)
        self.spatial_shape = spatial_shape if spatial_shape else (4, 4)
        self.flat_dim      = flat_dim if flat_dim else B * 8 * 4 * 4

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.flat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            ResBlock(B*8),
            nn.ConvTranspose2d(B*8, B*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(B*8),
            nn.LeakyReLU(0.2, inplace=True),

            ResBlock(B*8),
            nn.ConvTranspose2d(B*8, B*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(B*4),
            nn.LeakyReLU(0.2, inplace=True),

            ResBlock(B*4),
            nn.ConvTranspose2d(B*4, B*2, 4, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(B*2),
            nn.LeakyReLU(0.2, inplace=True),

            ResBlock(B*2),
            nn.ConvTranspose2d(B*2, B,   4, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(B),
            nn.LeakyReLU(0.2, inplace=True),

            ResBlock(B),
            nn.ConvTranspose2d(B, out_channels, 4, stride=2, padding=1,
                               output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h   = self.fc(z).view(-1, self.base_ch * 8, *self.spatial_shape)
        out = self.decoder(h)
        # Bilinear resize to exact 125×125 — handles any off-by-one
        if out.shape[-2:] != (125, 125):
            out = F.interpolate(out, size=(125, 125),
                                mode="bilinear", align_corners=False)
        return out


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, base_ch=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = Encoder(in_channels, latent_dim, base_ch)
        # Pass spatial info from encoder to decoder so they always match
        self.decoder    = Decoder(in_channels, latent_dim, base_ch,
                                  spatial_shape=self.encoder.spatial_shape,
                                  flat_dim=self.encoder.flat_dim)

    def reparameterise(self, mu, log_var):
        if self.training:
            std = (0.5 * log_var).exp()
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterise(mu, log_var)
        return self.decoder(z), mu, log_var

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


# ─────────────────────────────────────────────
# 3. Loss  (β-VAE)
# ─────────────────────────────────────────────
def vae_loss(recon, x, mu, log_var, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl         = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()


# ─────────────────────────────────────────────
# 4. Training loop  (with early stopping)
# ─────────────────────────────────────────────
def train_vae(model, train_loader, val_loader,
              n_epochs=50, lr=1e-3, beta=1.0, beta_warmup=10,
              save_dir="checkpoints", patience=10):
    os.makedirs(save_dir, exist_ok=True)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs, eta_min=1e-5)

    history  = {"train_loss": [], "val_loss": [],
                "train_recon": [], "train_kl": []}
    best_val = float("inf")
    no_improve = 0   # early stopping counter

    for epoch in range(1, n_epochs + 1):
        t0       = time.time()
        beta_eff = min(beta, beta * epoch / beta_warmup)

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_recon = tr_kl = 0
        for xb, _ in train_loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon, mu, lv = model(xb)
            loss, rl, kl  = vae_loss(recon, xb, mu, lv, beta_eff)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss  += loss.item()
            tr_recon += rl
            tr_kl    += kl

        n_tr     = len(train_loader)
        tr_loss /= n_tr; tr_recon /= n_tr; tr_kl /= n_tr

        # ── Validation ─────────────────────────────────────────────────
        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(DEVICE)
                recon, mu, lv = model(xb)
                loss, _, _    = vae_loss(recon, xb, mu, lv, beta_eff)
                vl += loss.item()
        vl /= len(val_loader)

        sched.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl)
        history["train_recon"].append(tr_recon)
        history["train_kl"].append(tr_kl)

        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{n_epochs} | "
              f"train {tr_loss:.5f} (recon {tr_recon:.5f}, kl {tr_kl:.5f}) | "
              f"val {vl:.5f} | β {beta_eff:.3f} | {dt:.1f}s")

        # ── Checkpoint & early stopping ────────────────────────────────
        if vl < best_val:
            best_val   = vl
            no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "vae_best.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\nBest val loss: {best_val:.5f}")
    return history


# ─────────────────────────────────────────────
# 5. Evaluation metrics
# ─────────────────────────────────────────────
def compute_metrics(model, dataset, n_samples=500):
    """Compute MSE, MAE, PSNR on n_samples test events."""
    model.eval()
    n   = min(n_samples, len(dataset))
    xs  = torch.stack([dataset[i][0] for i in range(n)]).to(DEVICE)
    with torch.no_grad():
        recon, _, _ = model(xs)
    o = xs.cpu().numpy()
    r = recon.cpu().numpy()

    mse  = float(np.mean((o - r) ** 2))
    mae  = float(np.mean(np.abs(o - r)))
    psnr = float(10 * np.log10(1.0 / (mse + 1e-8)))

    print(f"\n── Reconstruction Metrics ({n} test events) ──")
    print(f"  MSE  : {mse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  PSNR : {psnr:.2f} dB")
    return {"MSE": mse, "MAE": mae, "PSNR": psnr}


# ─────────────────────────────────────────────
# 6. Visualisation helpers
# ─────────────────────────────────────────────
CHANNEL_NAMES = ["ECAL", "HCAL", "Tracks"]
CHANNEL_CMAPS = ["inferno", "plasma", "viridis"]


def plot_original_vs_recon(model, dataset, n_events=6,
                            out_path="recon_comparison.png",
                            label_filter=None, title=""):
    model.eval()
    samples, labels_list = [], []
    for img, lbl in dataset:
        if label_filter is not None and lbl.item() != label_filter:
            continue
        samples.append(img); labels_list.append(lbl.item())
        if len(samples) == n_events:
            break

    if len(samples) == 0:
        print(f"No samples found for label_filter={label_filter}, skipping.")
        return

    x = torch.stack(samples).to(DEVICE)
    with torch.no_grad():
        recon, _, _ = model(x)
    x     = x.cpu().numpy()
    recon = recon.cpu().numpy()

    label_map = {0: "Gluon", 1: "Quark"}
    n_cols    = 6   # orig+recon × 3 channels
    fig, axes = plt.subplots(n_events, n_cols,
                              figsize=(n_cols * 2.5, n_events * 2.5),
                              facecolor="#0d0d0d")

    for row in range(n_events):
        for ch in range(3):
            orig_ax  = axes[row, ch * 2]
            recon_ax = axes[row, ch * 2 + 1]
            vmax = max(x[row, ch].max(), recon[row, ch].max(), 1e-6)

            orig_ax.imshow(x[row, ch],      cmap=CHANNEL_CMAPS[ch],
                           vmin=0, vmax=vmax, interpolation="nearest")
            recon_ax.imshow(recon[row, ch],  cmap=CHANNEL_CMAPS[ch],
                            vmin=0, vmax=vmax, interpolation="nearest")

            for ax in (orig_ax, recon_ax):
                ax.set_xticks([]); ax.set_yticks([])
                ax.spines[:].set_visible(False)

            if row == 0:
                orig_ax.set_title(f"{CHANNEL_NAMES[ch]}\nOriginal",
                                  color="white", fontsize=9, pad=4)
                recon_ax.set_title(f"{CHANNEL_NAMES[ch]}\nRecon.",
                                   color="white", fontsize=9, pad=4)

        axes[row, 0].set_ylabel(
            f"Event {row+1}\n({label_map.get(labels_list[row], '?')})",
            color="white", fontsize=8, rotation=90, labelpad=4)

    suptitle = title or "Quark/Gluon Jet VAE – Original vs Reconstructed"
    fig.suptitle(suptitle, color="white", fontsize=13, y=1.01)
    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_loss_curves(history, out_path="loss_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#0d0d0d")
    specs = [
        ("Total Loss",        "train_loss",  "val_loss"),
        ("Recon. Loss (MSE)", "train_recon", None),
        ("KL Divergence",     "train_kl",    None),
    ]
    for ax, (title, tk, vk) in zip(axes, specs):
        ax.plot(epochs, history[tk], color="#00e5ff", lw=1.8, label="Train")
        if vk:
            ax.plot(epochs, history[vk], color="#ff6e40",
                    lw=1.8, ls="--", label="Val")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("Epoch", color="#aaaaaa")
        ax.tick_params(colors="white")
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")
        if vk: ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    fig.suptitle("VAE Training Curves", color="white", fontsize=13)
    plt.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_latent_space(model, loader, out_path="latent_space.png",
                      max_batches=20):
    from sklearn.decomposition import PCA
    model.eval()
    mus, lbls = [], []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            if i >= max_batches: break
            mu, _ = model.encoder(xb.to(DEVICE))
            mus.append(mu.cpu().numpy())
            lbls.append(yb.numpy())
    mus  = np.concatenate(mus)
    lbls = np.concatenate(lbls)
    pca  = PCA(n_components=2)
    z2   = pca.fit_transform(mus)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a2e")
    for lbl, color, name in [(0, "#ff6e40", "Gluon"), (1, "#00e5ff", "Quark")]:
        mask = lbls == lbl
        ax.scatter(z2[mask, 0], z2[mask, 1], s=4, alpha=0.5,
                   color=color, label=name, rasterized=True)
    ax.set_title("Latent Space – PCA (2D)", color="white", fontsize=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                  color="#aaaaaa")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                  color="#aaaaaa")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", markerscale=4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_channel_histograms(model, dataset, out_path="pixel_histograms.png",
                             n_samples=500):
    model.eval()
    n  = min(n_samples, len(dataset))
    xs = torch.stack([dataset[i][0] for i in range(n)]).to(DEVICE)
    with torch.no_grad():
        recon, _, _ = model(xs)
    o = xs.cpu().numpy()
    r = recon.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0d0d0d")
    bins = np.linspace(0, 1, 80)
    for ch, ax in enumerate(axes):
        ax.set_facecolor("#1a1a2e")
        ax.hist(o[:, ch].flatten(), bins=bins, density=True,
                color="#00e5ff", alpha=0.6, label="Original",
                histtype="stepfilled")
        ax.hist(r[:, ch].flatten(), bins=bins, density=True,
                color="#ff6e40", alpha=0.6, label="Recon.",
                histtype="stepfilled")
        ax.set_title(CHANNEL_NAMES[ch], color="white", fontsize=11)
        ax.set_xlabel("Pixel value", color="#aaaaaa")
        ax.set_ylabel("Density",     color="#aaaaaa")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    fig.suptitle("Pixel-Value Distributions – Original vs Reconstructed",
                 color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_generated_samples(model, n=6, out_path="generated.png"):
    """Sample z ~ N(0,I) and decode — tests latent space quality."""
    model.eval()
    with torch.no_grad():
        imgs = model.sample(n, DEVICE).cpu().numpy()
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5), facecolor="#0d0d0d")
    for row in range(n):
        for ch in range(3):
            ax = axes[row, ch]
            ax.imshow(imgs[row, ch], cmap=CHANNEL_CMAPS[ch],
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(CHANNEL_NAMES[ch], color="white")
            if ch == 0:  ax.set_ylabel(f"Sample {row+1}",
                                        color="white", fontsize=8)
    fig.suptitle("VAE Generated Samples  z ~ N(0,I)",
                 color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepFALCON VAE – Task 1")
    parser.add_argument("--data",        type=str,   required=True)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch",       type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--latent",      type=int,   default=256)
    parser.add_argument("--beta",        type=float, default=1.0)
    parser.add_argument("--beta_warmup", type=int,   default=10)
    parser.add_argument("--base_ch",     type=int,   default=32)
    parser.add_argument("--max_samples", type=int,   default=None)
    parser.add_argument("--patience",    type=int,   default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--out_dir",     type=str,   default="outputs")
    parser.add_argument("--ckpt_dir",    type=str,   default="checkpoints")
    parser.add_argument("--resume",      type=str,   default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir,  exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    full_ds = JetImageDataset(args.data, max_samples=args.max_samples)
    n_tr = int(0.80 * len(full_ds))
    n_vl = int(0.10 * len(full_ds))
    n_te = len(full_ds) - n_tr - n_vl
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_tr, n_vl, n_te],
        generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train/Val/Test: {n_tr}/{n_vl}/{n_te}")

    # ── Model ────────────────────────────────────────────────────────────
    model    = VAE(in_channels=3, latent_dim=args.latent,
                   base_ch=args.base_ch).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAE parameters: {n_params:,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        print(f"Resumed from {args.resume}")

    # ── Training ─────────────────────────────────────────────────────────
    history = train_vae(
        model, train_loader, val_loader,
        n_epochs=args.epochs, lr=args.lr,
        beta=args.beta, beta_warmup=args.beta_warmup,
        save_dir=args.ckpt_dir, patience=args.patience)

    model.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, "vae_best.pt"),
                   map_location=DEVICE))

    # ── Evaluation metrics ───────────────────────────────────────────────
    metrics = compute_metrics(model, test_ds, n_samples=500)

    # ── Visualisations ───────────────────────────────────────────────────
    plot_loss_curves(history,
                     os.path.join(args.out_dir, "loss_curves.png"))

    plot_original_vs_recon(model, test_ds, n_events=6, label_filter=1,
                            title="Quark Jets – Original vs Reconstructed",
                            out_path=os.path.join(args.out_dir, "recon_quarks.png"))

    plot_original_vs_recon(model, test_ds, n_events=6, label_filter=0,
                            title="Gluon Jets – Original vs Reconstructed",
                            out_path=os.path.join(args.out_dir, "recon_gluons.png"))

    plot_latent_space(model, test_loader,
                      out_path=os.path.join(args.out_dir, "latent_space.png"))

    plot_channel_histograms(model, test_ds,
                             out_path=os.path.join(args.out_dir,
                                                    "pixel_histograms.png"))

    plot_generated_samples(model,
                            out_path=os.path.join(args.out_dir, "generated.png"))

    print(f"\n✓ Done. All outputs saved to: {args.out_dir}")
    print(f"  MSE={metrics['MSE']:.6f}  "
          f"MAE={metrics['MAE']:.6f}  "
          f"PSNR={metrics['PSNR']:.2f}dB")
