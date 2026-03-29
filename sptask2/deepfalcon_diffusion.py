"""
DeepFALCON GSoC 2026 – Specific Task 2
Denoising Diffusion Probabilistic Model (DDPM) for Jet Image Generation

Architecture: U-Net with sinusoidal time embeddings
Dataset:      Quark/Gluon jet images (3 × 125 × 125)
Reference:    Ho et al. "Denoising Diffusion Probabilistic Models" (2020)

Evaluation vs VAE:
  - FID-style pixel distribution comparison
  - MSE / MAE reconstruction
  - Side-by-side original vs reconstructed panels
  - Channel-wise energy histograms
"""

# ─────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────
import os, random, time, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from tqdm import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. Dataset  (same preprocessing as Task 1)
# ─────────────────────────────────────────────
class JetImageDataset(Dataset):
    def __init__(self, filepath, max_samples=None):
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            x_key = "X" if "X" in keys else "jetImage"
            y_key = "y" if "y" in keys else "jetLabel"
            X = f[x_key][:]
            self.labels = torch.tensor(f[y_key][:], dtype=torch.long)

        if X.ndim == 4 and X.shape[-1] == 3:
            X = X.transpose(0, 3, 1, 2)
        X = X.astype(np.float32)
        X = np.log1p(X)
        for c in range(3):
            p = np.percentile(X[:, c], 99)
            if p > 0: X[:, c] = np.clip(X[:, c] / p, 0, 1)

        if max_samples:
            X = X[:max_samples]
            self.labels = self.labels[:max_samples]

        # Scale to [-1, 1] — diffusion models work in this range
        X = X * 2.0 - 1.0

        self.data = torch.tensor(X, dtype=torch.float32)
        print(f"Loaded {len(self.data)} events, shape {tuple(self.data.shape[1:])}")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────
# 2. DDPM Noise Schedule
# ─────────────────────────────────────────────
class DDPMScheduler:
    """
    Linear beta schedule from Ho et al. 2020.
    
    Forward process:  q(x_t | x_0) = N(sqrt(ᾱ_t) x_0,  (1-ᾱ_t) I)
    Reverse process:  p_θ(x_{t-1} | x_t) learned by U-Net
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        # β schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)

        # Derived quantities
        self.alphas        = 1.0 - self.betas
        self.alpha_bar     = torch.cumprod(self.alphas, dim=0)          # ᾱ_t
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)

        self.sqrt_alpha_bar       = self.alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1 - self.alpha_bar).sqrt()

        # For reverse step
        self.sqrt_recip_alpha  = (1.0 / self.alphas).sqrt()
        self.posterior_variance = (self.betas *
                                   (1 - self.alpha_bar_prev) /
                                   (1 - self.alpha_bar))

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion: add noise to x0 at timestep t.
        x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_scalar):
        """
        One reverse denoising step.
        """
        t_batch = torch.full((x_t.shape[0],), t_scalar,
                             device=self.device, dtype=torch.long)
        pred_noise = model(x_t, t_batch)

        # Compute x_{t-1}
        betas_t          = self.betas[t_scalar]
        sqrt_one_minus   = self.sqrt_one_minus_alpha_bar[t_scalar]
        sqrt_recip_alpha = self.sqrt_recip_alpha[t_scalar]

        mean = sqrt_recip_alpha * (
            x_t - betas_t / sqrt_one_minus * pred_noise
        )

        if t_scalar == 0:
            return mean
        else:
            var   = self.posterior_variance[t_scalar]
            noise = torch.randn_like(x_t)
            return mean + var.sqrt() * noise

    @torch.no_grad()
    def sample(self, model, shape, show_progress=True):
        """Full reverse chain: pure noise → clean image."""
        x = torch.randn(shape, device=self.device)
        iterator = range(self.T - 1, -1, -1)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling", leave=False)
        for t in iterator:
            x = self.p_sample(model, x, t)
        return x

    @torch.no_grad()
    def reconstruct(self, model, x0, t_noise=500):
        """
        Reconstruction: corrupt x0 to timestep t_noise, then denoise back.
        Used for side-by-side comparison like the VAE.
        """
        t_batch = torch.full((x0.shape[0],), t_noise,
                             device=self.device, dtype=torch.long)
        x_noisy, _ = self.q_sample(x0, t_batch)

        x = x_noisy.clone()
        for t in range(t_noise - 1, -1, -1):
            x = self.p_sample(model, x, t)
        return x


# ─────────────────────────────────────────────
# 3. U-Net with Time Embeddings
# ─────────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for timestep t.
    Maps scalar t → vector of dim `dim`.
    Same idea as transformer positional encodings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=device) / (half - 1)
        )
        args   = t[:, None].float() * freqs[None]
        emb    = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb   # (B, dim)


class ResBlock(nn.Module):
    """
    Residual block with time embedding injection.
    Time embedding is added after first conv via FiLM-style shift.
    """
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention at bottleneck — captures global jet structure."""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads=4,
                                           batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h


class UNet(nn.Module):
    """
    U-Net for noise prediction in DDPM.

    Input:  noisy image (B, 3, 125, 125) + timestep t (B,)
    Output: predicted noise (B, 3, 125, 125)

    Architecture:
      Encoder: 3 downsampling levels
      Bottleneck: ResBlock + Attention + ResBlock
      Decoder: 3 upsampling levels with skip connections
    """
    def __init__(self, in_ch=3, base_ch=64, time_dim=256):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial conv
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ── Encoder ──────────────────────────────────
        # Level 1: base_ch  → 125×125
        self.enc1a = ResBlock(base_ch,     base_ch,   time_dim)
        self.enc1b = ResBlock(base_ch,     base_ch,   time_dim)
        self.down1 = nn.Conv2d(base_ch,    base_ch*2, 4, stride=2, padding=1)  # →62

        # Level 2: base_ch*2 → 62×62
        self.enc2a = ResBlock(base_ch*2,   base_ch*2, time_dim)
        self.enc2b = ResBlock(base_ch*2,   base_ch*2, time_dim)
        self.down2 = nn.Conv2d(base_ch*2,  base_ch*4, 4, stride=2, padding=1)  # →31

        # Level 3: base_ch*4 → 31×31
        self.enc3a = ResBlock(base_ch*4,   base_ch*4, time_dim)
        self.enc3b = ResBlock(base_ch*4,   base_ch*4, time_dim)
        self.down3 = nn.Conv2d(base_ch*4,  base_ch*8, 4, stride=2, padding=1)  # →15

        # ── Bottleneck ───────────────────────────────
        self.mid1  = ResBlock(base_ch*8,   base_ch*8, time_dim)
        self.mid_attn = AttentionBlock(base_ch*8)
        self.mid2  = ResBlock(base_ch*8,   base_ch*8, time_dim)

        # ── Decoder ──────────────────────────────────
        # Level 3 up: base_ch*8 + base_ch*4 → base_ch*4
        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, stride=2, padding=1)
        self.dec3a = ResBlock(base_ch*8,   base_ch*4, time_dim)   # *8 = skip+up
        self.dec3b = ResBlock(base_ch*4,   base_ch*4, time_dim)

        # Level 2 up: base_ch*4 + base_ch*2 → base_ch*2
        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1)
        self.dec2a = ResBlock(base_ch*4,   base_ch*2, time_dim)
        self.dec2b = ResBlock(base_ch*2,   base_ch*2, time_dim)

        # Level 1 up: base_ch*2 + base_ch → base_ch
        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch,   4, stride=2, padding=1)
        self.dec1a = ResBlock(base_ch*2,   base_ch,   time_dim)
        self.dec1b = ResBlock(base_ch,     base_ch,   time_dim)

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 1)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)   # (B, time_dim)

        # Initial conv
        x = self.init_conv(x)        # (B, base_ch, 125, 125)

        # Encoder
        e1 = self.enc1b(self.enc1a(x,  t_emb), t_emb)   # (B, C,  125, 125)
        e2 = self.enc2b(self.enc2a(self.down1(e1), t_emb), t_emb)
        e3 = self.enc3b(self.enc3a(self.down2(e2), t_emb), t_emb)
        b  = self.down3(e3)

        # Bottleneck
        b = self.mid1(b, t_emb)
        b = self.mid_attn(b)
        b = self.mid2(b, t_emb)

        # Decoder with skip connections + spatial alignment
        d = self.up3(b)
        d = F.interpolate(d, size=e3.shape[2:], mode="nearest")
        d = self.dec3b(self.dec3a(torch.cat([d, e3], dim=1), t_emb), t_emb)

        d = self.up2(d)
        d = F.interpolate(d, size=e2.shape[2:], mode="nearest")
        d = self.dec2b(self.dec2a(torch.cat([d, e2], dim=1), t_emb), t_emb)

        d = self.up1(d)
        d = F.interpolate(d, size=e1.shape[2:], mode="nearest")
        d = self.dec1b(self.dec1a(torch.cat([d, e1], dim=1), t_emb), t_emb)

        return self.out_conv(d)   # (B, 3, 125, 125)


# ─────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────
def train_diffusion(model, scheduler, train_loader, val_loader,
                    n_epochs=50, lr=2e-4, save_dir="checkpoints_diff"):
    os.makedirs(save_dir, exist_ok=True)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────
        model.train()
        tr_loss = 0
        for x0, _ in tqdm(train_loader, desc=f"Ep {epoch}", leave=False):
            x0 = x0.to(DEVICE)
            # Random timestep per sample
            t  = torch.randint(0, scheduler.T, (x0.shape[0],), device=DEVICE)
            x_noisy, noise = scheduler.q_sample(x0, t)
            pred_noise = model(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # ── Val ────────────────────────────────────
        model.eval()
        vl_loss = 0
        with torch.no_grad():
            for x0, _ in val_loader:
                x0 = x0.to(DEVICE)
                t  = torch.randint(0, scheduler.T, (x0.shape[0],), device=DEVICE)
                x_noisy, noise = scheduler.q_sample(x0, t)
                pred_noise = model(x_noisy, t)
                vl_loss += F.mse_loss(pred_noise, noise).item()
        vl_loss /= len(val_loader)

        sched.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "ddpm_best.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Ep {epoch:3d}/{n_epochs} | "
                  f"train {tr_loss:.5f} | val {vl_loss:.5f} | "
                  f"{time.time()-t0:.1f}s")

    return history


# ─────────────────────────────────────────────
# 5. Evaluation Metrics
# ─────────────────────────────────────────────
def to_01(x):
    """Convert [-1,1] back to [0,1] for display and metrics."""
    return (x.clamp(-1, 1) + 1) / 2


def compute_metrics(originals, reconstructions):
    """
    originals, reconstructions: tensors in [-1,1], shape (N,3,H,W)
    Returns dict of metrics.
    """
    o = to_01(originals).cpu().numpy()
    r = to_01(reconstructions).cpu().numpy()

    mse  = float(np.mean((o - r)**2))
    mae  = float(np.mean(np.abs(o - r)))
    psnr = float(10 * np.log10(1.0 / (mse + 1e-8)))

    # Per-channel Wasserstein-1 proxy: mean absolute difference of CDFs
    w1_per_ch = []
    for c in range(3):
        bins  = np.linspace(0, 1, 100)
        h_o,_ = np.histogram(o[:,c].flatten(), bins=bins, density=True)
        h_r,_ = np.histogram(r[:,c].flatten(), bins=bins, density=True)
        w1_per_ch.append(float(np.mean(np.abs(
            np.cumsum(h_o) - np.cumsum(h_r)
        )) / len(bins)))

    return {
        "MSE":  mse,
        "MAE":  mae,
        "PSNR": psnr,
        "W1_ECAL":  w1_per_ch[0],
        "W1_HCAL":  w1_per_ch[1],
        "W1_Tracks": w1_per_ch[2],
        "W1_mean":  float(np.mean(w1_per_ch)),
    }


# ─────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────
CHANNEL_NAMES = ["ECAL", "HCAL", "Tracks"]
CHANNEL_CMAPS = ["inferno", "plasma", "viridis"]


def plot_loss_curves(history, out):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a2e")
    ax.plot(epochs, history["train_loss"], color="#00e5ff", lw=1.8, label="Train")
    ax.plot(epochs, history["val_loss"],   color="#ff6e40", lw=1.8,
            ls="--", label="Val")
    ax.set_title("DDPM Training Loss (noise prediction MSE)",
                 color="white", fontsize=12)
    ax.set_xlabel("Epoch", color="#aaa")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_recon_comparison(originals, reconstructions, n=6,
                           title="DDPM – Original vs Reconstructed", out="recon.png"):
    """Side-by-side: orig ECAL | recon ECAL | orig HCAL | recon HCAL | ..."""
    o = to_01(originals[:n]).cpu().numpy()
    r = to_01(reconstructions[:n]).cpu().numpy()

    fig, axes = plt.subplots(n, 6, figsize=(16, n * 2.6), facecolor="#0d0d0d")
    for row in range(n):
        for ch in range(3):
            vmax = max(o[row, ch].max(), r[row, ch].max(), 1e-6)
            for j, (data, suffix) in enumerate([(o, "Orig."), (r, "Recon.")]):
                ax = axes[row, ch * 2 + j]
                ax.imshow(data[row, ch], cmap=CHANNEL_CMAPS[ch],
                          vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"{CHANNEL_NAMES[ch]}\n{suffix}",
                                 color="white", fontsize=9)
            if ch == 0:
                axes[row, 0].set_ylabel(f"Event {row+1}",
                                         color="white", fontsize=8)
    fig.suptitle(title, color="white", fontsize=13, y=1.01)
    plt.tight_layout(pad=0.4)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_pixel_histograms(originals, reconstructions, out):
    o = to_01(originals).cpu().numpy()
    r = to_01(reconstructions).cpu().numpy()
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
        ax.set_title(CHANNEL_NAMES[ch], color="white")
        ax.set_xlabel("Pixel value", color="#aaa")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    fig.suptitle("Pixel Distributions – Original vs DDPM Reconstructed",
                 color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_noise_levels(scheduler, sample_img, out):
    """Show a single jet at several noise levels t."""
    ts = [0, 100, 250, 500, 750, 999]
    fig, axes = plt.subplots(3, len(ts), figsize=(len(ts)*2.5, 9),
                              facecolor="#0d0d0d")
    x0 = sample_img.unsqueeze(0).to(DEVICE)
    for col, t_val in enumerate(ts):
        t_b = torch.tensor([t_val], device=DEVICE)
        x_noisy, _ = scheduler.q_sample(x0, t_b)
        img = to_01(x_noisy[0]).cpu().numpy()
        for ch in range(3):
            ax = axes[ch, col]
            ax.imshow(img[ch], cmap=CHANNEL_CMAPS[ch], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ch == 0: ax.set_title(f"t={t_val}", color="white", fontsize=9)
            if col == 0: ax.set_ylabel(CHANNEL_NAMES[ch],
                                        color="white", fontsize=8)
    fig.suptitle("Forward Diffusion – Noise Levels",
                 color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_metrics_comparison(ddpm_metrics, vae_metrics=None, out="metrics.png"):
    """Bar chart comparing DDPM vs VAE metrics."""
    keys   = ["MSE", "MAE", "W1_ECAL", "W1_HCAL", "W1_Tracks"]
    d_vals = [ddpm_metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a2e")
    x = np.arange(len(keys))
    w = 0.35

    bars1 = ax.bar(x - w/2 if vae_metrics else x, d_vals,
                   width=w if vae_metrics else 0.5,
                   color="#00e5ff", alpha=0.8, label="DDPM")
    if vae_metrics:
        v_vals = [vae_metrics[k] for k in keys]
        bars2  = ax.bar(x + w/2, v_vals, width=w,
                        color="#ff6e40", alpha=0.8, label="VAE")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")

    ax.set_xticks(x); ax.set_xticklabels(keys, color="white", rotation=15)
    ax.set_title("Reconstruction Metrics: DDPM vs VAE",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_generated_samples(scheduler, model, n=6, out="generated.png"):
    """Generate brand-new jets from pure Gaussian noise."""
    model.eval()
    samples = scheduler.sample(model, shape=(n, 3, 125, 125),
                               show_progress=True)
    imgs = to_01(samples).cpu().numpy()

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
    fig.suptitle("DDPM Generated Jet Samples (z ~ N(0,I))",
                 color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


# ─────────────────────────────────────────────
# 7. Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepFALCON DDPM – Specific Task 2")
    parser.add_argument("--data",         type=str,   required=True)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch",        type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--T",            type=int,   default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--t_recon",      type=int,   default=500,
                        help="Noise level for reconstruction comparison")
    parser.add_argument("--base_ch",      type=int,   default=64,
                        help="U-Net base channels (reduce to 32 for CPU)")
    parser.add_argument("--max_samples",  type=int,   default=None)
    parser.add_argument("--out_dir",      type=str,   default="outputs_diff")
    parser.add_argument("--ckpt_dir",     type=str,   default="checkpoints_diff")
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--num_workers",  type=int,   default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────
    full_ds = JetImageDataset(args.data, max_samples=args.max_samples)
    n_tr = int(0.80 * len(full_ds))
    n_vl = int(0.10 * len(full_ds))
    n_te = len(full_ds) - n_tr - n_vl
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_tr, n_vl, n_te],
        generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, args.batch, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   args.batch, shuffle=False,
                              num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  args.batch, shuffle=False,
                              num_workers=args.num_workers)

    # ── Scheduler & Model ────────────────────
    scheduler = DDPMScheduler(T=args.T, device=DEVICE)
    model     = UNet(in_ch=3, base_ch=args.base_ch).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net parameters: {n_params:,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))

    # ── Noise level visualisation ────────────
    sample_img = full_ds[0][0]
    plot_noise_levels(scheduler, sample_img,
                      out=os.path.join(args.out_dir, "noise_levels.png"))

    # ── Train ────────────────────────────────
    history = train_diffusion(model, scheduler, train_loader, val_loader,
                               n_epochs=args.epochs, lr=args.lr,
                               save_dir=args.ckpt_dir)

    model.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, "ddpm_best.pt"),
                   map_location=DEVICE))

    # ── Loss curves ──────────────────────────
    plot_loss_curves(history,
                     out=os.path.join(args.out_dir, "loss_curves.png"))

    # ── Reconstruction comparison ─────────────
    model.eval()
    test_imgs, test_labels = [], []
    for x, y in test_loader:
        test_imgs.append(x); test_labels.append(y)
        if sum(len(t) for t in test_imgs) >= 50: break
    test_imgs = torch.cat(test_imgs)[:50].to(DEVICE)

    print("Reconstructing test events...")
    recons = scheduler.reconstruct(model, test_imgs, t_noise=args.t_recon)

    # Quark events
    plot_recon_comparison(
        test_imgs[:6], recons[:6],
        title=f"DDPM – Original vs Reconstructed (t_noise={args.t_recon})",
        out=os.path.join(args.out_dir, "recon_comparison.png"))

    # ── Pixel histograms ─────────────────────
    plot_pixel_histograms(test_imgs, recons,
                          out=os.path.join(args.out_dir, "pixel_histograms.png"))

    # ── Metrics ──────────────────────────────
    ddpm_metrics = compute_metrics(test_imgs, recons)
    print("\n── DDPM Reconstruction Metrics ──────────────")
    for k, v in ddpm_metrics.items():
        print(f"  {k:12s}: {v:.6f}")

    plot_metrics_comparison(ddpm_metrics,
                            out=os.path.join(args.out_dir, "metrics.png"))

    # ── Generated samples ────────────────────
    print("\nGenerating new samples from noise...")
    plot_generated_samples(scheduler, model, n=6,
                           out=os.path.join(args.out_dir, "generated.png"))

    print(f"\n✓ All outputs saved to: {args.out_dir}")
