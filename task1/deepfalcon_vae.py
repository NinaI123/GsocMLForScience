"""
DeepFALCON GSoC 2026 – Common Task 1  (Improved VAE v2)
========================================================
Improvements over v1:
  1. U-Net skip connections  (encoder→decoder spatial shortcuts)
  2. Squeeze-and-Excitation (SE) blocks  (channel attention)
  3. GroupNorm instead of BatchNorm  (works better with small batches)
  4. PixelShuffle upsampling  (sharper than ConvTranspose)
  5. Combined loss: MSE + SSIM + β·KL
  6. Conditional VAE  (class label injected into latent space)
  7. Data augmentation  (random flips + mild Gaussian noise)
  8. AdamW + OneCycleLR scheduler
  9. Spectral normalisation on encoder convs
  10. Early stopping + best-checkpoint saving
  11. Auto flat-dim calculation  (no shape crashes)
  12. Smart RAM/lazy dataset loader

Usage:
  python deepfalcon_vae_v2.py --data path/to/file.hdf5 --epochs 50 --max_samples 20000
"""

# ─────────────────────────────────────────────
# 0. Imports
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

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
PIN_MEMORY  = torch.cuda.is_available()

print(f"Device: {DEVICE} | workers: {NUM_WORKERS} | pin_memory: {PIN_MEMORY}")


# ─────────────────────────────────────────────
# 1. Dataset  (with augmentation)
# ─────────────────────────────────────────────
class JetImageDataset(Dataset):
    """
    RAM load for ≤20k samples, lazy HDF5 for larger.
    Augmentation: random horizontal/vertical flip + tiny Gaussian noise.
    """
    RAM_THRESHOLD = 20_000

    def __init__(self, filepath, max_samples=None, augment=False):
        super().__init__()
        self.filepath = filepath
        self.augment  = augment
        self.lazy     = False

        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            print(f"HDF5 keys: {keys}")
            self.x_key = ("X_jets" if "X_jets" in keys
                          else ("X" if "X" in keys else "jetImage"))
            self.y_key = "y" if "y" in keys else "jetLabel"
            total      = f[self.x_key].shape[0]
            self.n     = min(max_samples, total) if max_samples else total
            self.labels = torch.tensor(f[self.y_key][:self.n], dtype=torch.long)

            if self.n <= self.RAM_THRESHOLD:
                print(f"Loading {self.n:,} events into RAM...")
                X = f[self.x_key][:self.n]
                if X.ndim == 4 and X.shape[-1] == 3:
                    X = X.transpose(0, 3, 1, 2)
                X = self._preprocess(X.astype(np.float32))
                self.data = torch.tensor(X, dtype=torch.float32)
                print(f"Loaded. Shape: {tuple(self.data.shape[1:])}")
            else:
                print(f"Large dataset ({self.n:,}) — using lazy loading.")
                self.lazy = True
                sample = f[self.x_key][:min(2000, self.n)]
                if sample.ndim == 4 and sample.shape[-1] == 3:
                    sample = sample.transpose(0, 3, 1, 2)
                sample   = np.log1p(sample.astype(np.float32))
                self.p99 = [float(np.percentile(sample[:, c], 99))
                            for c in range(3)]

    def _preprocess(self, X):
        X = np.log1p(X)
        for c in range(3):
            p = np.percentile(X[:, c], 99)
            if p > 0: X[:, c] = np.clip(X[:, c] / p, 0, 1)
        return X

    def _augment(self, x):
        """Random horizontal + vertical flip, tiny noise."""
        if random.random() > 0.5:
            x = torch.flip(x, dims=[2])   # horizontal flip
        if random.random() > 0.5:
            x = torch.flip(x, dims=[1])   # vertical flip
        if random.random() > 0.5:
            x = x + torch.randn_like(x) * 0.01   # tiny noise
            x = x.clamp(0, 1)
        return x

    def __len__(self): return self.n

    def __getitem__(self, idx):
        lbl = self.labels[idx]
        if not self.lazy:
            x = self.data[idx].clone()
        else:
            with h5py.File(self.filepath, "r") as f:
                raw = f[self.x_key][idx]
            if raw.ndim == 3 and raw.shape[-1] == 3:
                raw = raw.transpose(2, 0, 1)
            raw = np.log1p(raw.astype(np.float32))
            for c in range(3):
                if self.p99[c] > 0:
                    raw[c] = np.clip(raw[c] / self.p99[c], 0, 1)
            x = torch.tensor(raw, dtype=torch.float32)

        if self.augment:
            x = self._augment(x)
        return x, lbl


# ─────────────────────────────────────────────
# 2. Improved Building Blocks
# ─────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Learns to re-weight each channel based on global context.
    Think of it as 'which channels are most important for this image?'
    """
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x).flatten(1)          # (B, C)
        w = self.fc(w).view(-1, x.shape[1], 1, 1)
        return x * w                          # re-scale channels


class ResBlockGN(nn.Module):
    """
    Residual block with:
      - GroupNorm instead of BatchNorm (better for small batches)
      - Squeeze-and-Excitation channel attention
      - Optional spatial dropout for regularisation
    """
    def __init__(self, ch, dropout=0.0, groups=8):
        super().__init__()
        # GroupNorm: split channels into groups, normalise each group
        # Works well even with batch_size=1, unlike BatchNorm
        g = min(groups, ch)   # can't have more groups than channels
        self.net = nn.Sequential(
            nn.GroupNorm(g, ch),
            nn.SiLU(),                        # SiLU smoother than LeakyReLU
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )
        self.se      = SEBlock(ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.dropout(self.se(self.net(x)))


class DownBlock(nn.Module):
    """
    Encoder downsampling block.
    Strided conv + 2× ResBlockGN.
    Spectral norm on the strided conv for training stability.
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.down = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1))
        self.res1 = ResBlockGN(out_ch, dropout)
        self.res2 = ResBlockGN(out_ch, dropout)

    def forward(self, x):
        return self.res2(self.res1(self.down(x)))


class UpBlock(nn.Module):
    """
    Decoder upsampling block using PixelShuffle.
    PixelShuffle rearranges (B, C*r², H, W) → (B, C, H*r, W*r).
    Produces sharper images than ConvTranspose2d which causes checkerboard.
    Takes skip connection from encoder (U-Net style).
    """
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        # PixelShuffle upscale by 2: need in_ch → out_ch*4 channels
        self.shuffle_conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)       # ×2 spatial, ÷4 channels
        # After concat with skip: out_ch + skip_ch channels
        self.res1 = ResBlockGN(out_ch + skip_ch, dropout)
        self.res2 = ResBlockGN(out_ch + skip_ch, dropout)
        self.proj = nn.Conv2d(out_ch + skip_ch, out_ch, 1)  # project back

    def forward(self, x, skip):
        x = self.pixel_shuffle(self.shuffle_conv(x))     # upsample
        # Align spatial size with skip (handles odd dimensions)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)                  # U-Net skip concat
        return self.proj(self.res2(self.res1(x)))


# ─────────────────────────────────────────────
# 3. Conditional VAE
# ─────────────────────────────────────────────
class EncoderV2(nn.Module):
    """
    U-Net encoder — returns (mu, log_var) AND skip connections
    for the decoder to use.
    """
    def __init__(self, in_ch=3, latent_dim=256, base_ch=32, dropout=0.1):
        super().__init__()
        B = base_ch
        self.init_conv = nn.Conv2d(in_ch, B, 3, padding=1)

        self.down1 = DownBlock(B,    B*2,  dropout)   # 125→62
        self.down2 = DownBlock(B*2,  B*4,  dropout)   # 62→31
        self.down3 = DownBlock(B*4,  B*8,  dropout)   # 31→15
        self.down4 = DownBlock(B*8,  B*8,  dropout)   # 15→7

        # Bottleneck with self-attention
        self.bottleneck = nn.Sequential(
            ResBlockGN(B*8, dropout),
            ResBlockGN(B*8, dropout),
        )

        # Auto-calculate flat dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 125, 125)
            s0    = self.init_conv(dummy)
            s1    = self.down1(s0)
            s2    = self.down2(s1)
            s3    = self.down3(s2)
            s4    = self.down4(s3)
            bot   = self.bottleneck(s4)
            self.spatial_shape = bot.shape[2:]
            self.flat_dim      = int(bot.flatten(1).shape[1])
        print(f"Encoder: spatial={self.spatial_shape}  flat={self.flat_dim}")

        self.fc_mu  = nn.Linear(self.flat_dim, latent_dim)
        self.fc_lv  = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        s0 = self.init_conv(x)          # (B, base_ch, 125, 125)
        s1 = self.down1(s0)              # skip 1
        s2 = self.down2(s1)              # skip 2
        s3 = self.down3(s2)              # skip 3
        s4 = self.down4(s3)              # skip 4
        h  = self.bottleneck(s4)

        flat   = h.flatten(1)
        mu     = self.fc_mu(flat)
        log_var = self.fc_lv(flat)
        # Return skips for decoder U-Net connections
        return mu, log_var, (s0, s1, s2, s3)


class DecoderV2(nn.Module):
    """
    U-Net decoder with PixelShuffle upsampling.
    Receives skip connections from encoder.
    Also accepts class label embedding (conditional VAE).
    """
    def __init__(self, out_ch=3, latent_dim=256, base_ch=32,
                 n_classes=2, spatial_shape=None, flat_dim=None,
                 dropout=0.1):
        super().__init__()
        B = base_ch
        self.B            = B
        self.spatial_shape = spatial_shape or (4, 4)
        self.flat_dim      = flat_dim or B * 8 * 16

        # Class embedding — injects quark/gluon label into latent space
        self.class_embed = nn.Embedding(n_classes, latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.flat_dim),
            nn.SiLU(),
        )

        # U-Net decoder: each UpBlock takes skip from corresponding encoder level
        self.up4 = UpBlock(B*8,  B*4,  B*8,  dropout)   # +skip3
        self.up3 = UpBlock(B*8,  B*2,  B*4,  dropout)   # +skip2
        self.up2 = UpBlock(B*4,  B,    B*2,  dropout)   # +skip1
        self.up1 = UpBlock(B*2,  B,    B,    dropout)   # +skip0

        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, B), B),
            nn.SiLU(),
            nn.Conv2d(B, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, label, skips):
        s0, s1, s2, s3 = skips

        # Inject class information by adding class embedding to z
        # This is what makes it a Conditional VAE
        z = z + self.class_embed(label)

        h = self.fc(z).view(-1, self.B * 8, *self.spatial_shape)
        h = self.up4(h,  s3)
        h = self.up3(h,  s2)
        h = self.up2(h,  s1)
        h = self.up1(h,  s0)

        out = self.out_conv(h)
        if out.shape[-2:] != (125, 125):
            out = F.interpolate(out, (125, 125),
                                mode="bilinear", align_corners=False)
        return out


class VAEv2(nn.Module):
    """
    Conditional VAE with U-Net skip connections + SE attention.
    The 'conditional' part means the decoder knows whether it's
    reconstructing a quark or gluon jet — helps class-specific features.
    """
    def __init__(self, in_ch=3, latent_dim=256, base_ch=32,
                 n_classes=2, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = EncoderV2(in_ch, latent_dim, base_ch, dropout)
        self.decoder    = DecoderV2(
            in_ch, latent_dim, base_ch, n_classes,
            spatial_shape=self.encoder.spatial_shape,
            flat_dim=self.encoder.flat_dim,
            dropout=dropout)

    def reparameterise(self, mu, lv):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return mu

    def forward(self, x, label):
        mu, lv, skips = self.encoder(x)
        z             = self.reparameterise(mu, lv)
        recon         = self.decoder(z, label, skips)
        return recon, mu, lv

    def sample(self, n, labels, device):
        """Generate n images conditioned on class labels."""
        z     = torch.randn(n, self.latent_dim, device=device)
        # Need dummy skips of the right shape for sampling
        # Use zero tensors as skips (unconditional generation)
        B     = self.decoder.B
        ss    = self.encoder.spatial_shape
        dummy_skips = (
            torch.zeros(n, B,   125, 125, device=device),
            torch.zeros(n, B*2,  62,  62, device=device),
            torch.zeros(n, B*4,  31,  31, device=device),
            torch.zeros(n, B*8,  15,  15, device=device),
        )
        return self.decoder(z, labels, dummy_skips)


# ─────────────────────────────────────────────
# 4. Improved Loss: MSE + SSIM + β·KL
# ─────────────────────────────────────────────
def ssim_loss(pred, target, window_size=7):
    """
    Structural Similarity loss.
    SSIM measures perceptual similarity — luminance, contrast, structure.
    Better than pure MSE for images because MSE treats all pixels equally
    but SSIM weighs based on local structure (more like human perception).
    """
    C1, C2 = 0.01**2, 0.03**2
    mu1    = F.avg_pool2d(pred,   window_size, 1, window_size//2)
    mu2    = F.avg_pool2d(target, window_size, 1, window_size//2)
    mu1_sq = mu1 * mu1;  mu2_sq = mu2 * mu2;  mu12 = mu1 * mu2

    s1  = F.avg_pool2d(pred   * pred,   window_size, 1, window_size//2) - mu1_sq
    s2  = F.avg_pool2d(target * target, window_size, 1, window_size//2) - mu2_sq
    s12 = F.avg_pool2d(pred   * target, window_size, 1, window_size//2) - mu12

    ssim_map = ((2*mu12 + C1) * (2*s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return 1 - ssim_map.mean()


def vae_loss_v2(recon, x, mu, lv, beta=1.0, ssim_weight=0.3):
    """
    Combined loss:
      mse_loss    — pixel accuracy
      ssim_loss   — perceptual/structural similarity
      kl          — latent space regularization
    """
    mse  = F.mse_loss(recon, x)
    ssim = ssim_loss(recon, x)
    kl   = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    recon_loss = (1 - ssim_weight) * mse + ssim_weight * ssim
    total      = recon_loss + beta * kl
    return total, mse.item(), ssim.item(), kl.item()


# ─────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────
def train_vae(model, train_loader, val_loader,
              n_epochs=50, lr=3e-4, beta=1.0, beta_warmup=10,
              ssim_weight=0.3, save_dir="checkpoints_v2", patience=10):
    os.makedirs(save_dir, exist_ok=True)

    # AdamW — better weight decay than Adam
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # OneCycleLR — ramps up then down, very effective
    total_steps = n_epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos")

    history    = {k: [] for k in
                  ["train_loss","val_loss","train_mse","train_ssim","train_kl"]}
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        t0       = time.time()
        beta_eff = min(beta, beta * epoch / max(beta_warmup, 1))

        # ── Train ──────────────────────────────────────────────────
        model.train()
        tl = tm = ts = tk = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            recon, mu, lv   = model(xb, yb)
            loss, mse, ssim, kl = vae_loss_v2(recon, xb, mu, lv,
                                               beta_eff, ssim_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item(); tm += mse; ts += ssim; tk += kl

        n = len(train_loader)
        tl /= n; tm /= n; ts /= n; tk /= n

        # ── Validation ─────────────────────────────────────────────
        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                recon, mu, lv = model(xb, yb)
                loss, _, _, _ = vae_loss_v2(recon, xb, mu, lv,
                                             beta_eff, ssim_weight)
                vl += loss.item()
        vl /= len(val_loader)

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_mse"].append(tm)
        history["train_ssim"].append(ts)
        history["train_kl"].append(tk)

        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{n_epochs} | "
              f"loss {tl:.5f} (mse {tm:.5f} ssim {ts:.4f} kl {tk:.5f}) | "
              f"val {vl:.5f} | β {beta_eff:.3f} | {dt:.1f}s")

        if vl < best_val:
            best_val   = vl
            no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "vae_v2_best.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val:.5f}")
    return history


# ─────────────────────────────────────────────
# 6. Evaluation metrics
# ─────────────────────────────────────────────
def compute_metrics(model, dataset, n_samples=500):
    model.eval()
    n  = min(n_samples, len(dataset))
    xs, ys = [], []
    for i in range(n):
        x, y = dataset[i]; xs.append(x); ys.append(y)
    xs = torch.stack(xs).to(DEVICE)
    ys = torch.stack(ys).to(DEVICE)

    with torch.no_grad():
        recon, _, _ = model(xs, ys)
    o = xs.cpu().numpy()
    r = recon.cpu().numpy()

    mse  = float(np.mean((o - r) ** 2))
    mae  = float(np.mean(np.abs(o - r)))
    psnr = float(10 * np.log10(1.0 / (mse + 1e-8)))

    # Per-channel Wasserstein-1 proxy
    w1_scores = []
    for c in range(3):
        bins  = np.linspace(0, 1, 100)
        ho, _ = np.histogram(o[:, c].flatten(), bins=bins, density=True)
        hr, _ = np.histogram(r[:, c].flatten(), bins=bins, density=True)
        w1_scores.append(float(np.mean(np.abs(
            np.cumsum(ho) - np.cumsum(hr))) / len(bins)))

    print(f"\n── Reconstruction Metrics ({n} test events) ──")
    print(f"  MSE       : {mse:.6f}")
    print(f"  MAE       : {mae:.6f}")
    print(f"  PSNR      : {psnr:.2f} dB")
    print(f"  W1 ECAL   : {w1_scores[0]:.6f}")
    print(f"  W1 HCAL   : {w1_scores[1]:.6f}")
    print(f"  W1 Tracks : {w1_scores[2]:.6f}")
    return {"MSE": mse, "MAE": mae, "PSNR": psnr,
            "W1_ECAL": w1_scores[0], "W1_HCAL": w1_scores[1],
            "W1_Tracks": w1_scores[2]}


# ─────────────────────────────────────────────
# 7. Visualisations
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
        samples.append((img, lbl))
        if len(samples) == n_events: break

    if not samples:
        print(f"No samples for label_filter={label_filter}, skipping.")
        return

    xs  = torch.stack([s[0] for s in samples]).to(DEVICE)
    ys  = torch.stack([s[1] for s in samples]).to(DEVICE)
    labels_list = [s[1].item() for s in samples]

    with torch.no_grad():
        recon, _, _ = model(xs, ys)
    x     = xs.cpu().numpy()
    recon = recon.cpu().numpy()

    label_map = {0: "Gluon", 1: "Quark"}
    fig, axes = plt.subplots(n_events, 6,
                              figsize=(16, n_events * 2.6),
                              facecolor="#0d0d0d")
    for row in range(n_events):
        for ch in range(3):
            vmax = max(x[row, ch].max(), recon[row, ch].max(), 1e-6)
            for j, (data, suffix) in enumerate([(x, "Orig."), (recon, "Recon.")]):
                ax = axes[row, ch * 2 + j]
                ax.imshow(data[row, ch], cmap=CHANNEL_CMAPS[ch],
                          vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.spines[:].set_visible(False)
                if row == 0:
                    ax.set_title(f"{CHANNEL_NAMES[ch]}\n{suffix}",
                                 color="white", fontsize=9)
        axes[row, 0].set_ylabel(
            f"Event {row+1}\n({label_map.get(labels_list[row], '?')})",
            color="white", fontsize=8)

    fig.suptitle(title or "VAE v2 – Original vs Reconstructed",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_loss_curves(history, out_path="loss_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor="#0d0d0d")
    specs = [
        ("Total Loss",   "train_loss",  "val_loss"),
        ("MSE",          "train_mse",   None),
        ("SSIM Loss",    "train_ssim",  None),
        ("KL Divergence","train_kl",    None),
    ]
    for ax, (title, tk, vk) in zip(axes, specs):
        ax.plot(epochs, history[tk], color="#00e5ff", lw=1.8, label="Train")
        if vk:
            ax.plot(epochs, history[vk], color="#ff6e40",
                    lw=1.8, ls="--", label="Val")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("Epoch", color="#aaa")
        ax.tick_params(colors="white")
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")
        if vk: ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    fig.suptitle("VAE v2 Training Curves", color="white", fontsize=13)
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
            mu, _, _ = model.encoder(xb.to(DEVICE))
            mus.append(mu.cpu().numpy())
            lbls.append(yb.numpy())
    mus  = np.concatenate(mus)
    lbls = np.concatenate(lbls)
    pca  = PCA(n_components=2)
    z2   = pca.fit_transform(mus)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a2e")
    for lbl, color, name in [(0,"#ff6e40","Gluon"),(1,"#00e5ff","Quark")]:
        mask = lbls == lbl
        ax.scatter(z2[mask,0], z2[mask,1], s=4, alpha=0.5,
                   color=color, label=name, rasterized=True)
    ax.set_title("Latent Space – PCA (2D)", color="white", fontsize=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                  color="#aaa")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                  color="#aaa")
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
    xs, ys = [], []
    for i in range(n):
        x, y = dataset[i]; xs.append(x); ys.append(y)
    xs = torch.stack(xs).to(DEVICE)
    ys = torch.stack(ys).to(DEVICE)
    with torch.no_grad():
        recon, _, _ = model(xs, ys)
    o = xs.cpu().numpy(); r = recon.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0d0d0d")
    bins = np.linspace(0, 1, 80)
    for ch, ax in enumerate(axes):
        ax.set_facecolor("#1a1a2e")
        ax.hist(o[:,ch].flatten(), bins=bins, density=True,
                color="#00e5ff", alpha=0.6, label="Original",
                histtype="stepfilled")
        ax.hist(r[:,ch].flatten(), bins=bins, density=True,
                color="#ff6e40", alpha=0.6, label="Recon.",
                histtype="stepfilled")
        ax.set_title(CHANNEL_NAMES[ch], color="white", fontsize=11)
        ax.set_xlabel("Pixel value", color="#aaa")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    fig.suptitle("Pixel Distributions – Original vs Reconstructed",
                 color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_generated_samples(model, n=6, out_path="generated.png"):
    """Generate class-conditional samples: 3 quarks + 3 gluons."""
    model.eval()
    labels = torch.tensor([0, 0, 0, 1, 1, 1], device=DEVICE)
    with torch.no_grad():
        imgs = model.sample(n, labels, DEVICE).cpu().numpy()

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5), facecolor="#0d0d0d")
    label_map = {0: "Gluon", 1: "Quark"}
    label_list = [0, 0, 0, 1, 1, 1]
    for row in range(n):
        for ch in range(3):
            ax = axes[row, ch]
            ax.imshow(imgs[row, ch], cmap=CHANNEL_CMAPS[ch],
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(CHANNEL_NAMES[ch], color="white")
            if ch == 0:  ax.set_ylabel(
                f"{label_map[label_list[row]]} {row%3+1}",
                color="white", fontsize=8)
    fig.suptitle("Conditional VAE – Generated Samples",
                 color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_metrics_bar(metrics_v1, metrics_v2, out_path="metrics_comparison.png"):
    """Bar chart comparing v1 vs v2 metrics side by side."""
    keys = ["MSE", "MAE", "W1_ECAL", "W1_HCAL", "W1_Tracks"]
    x    = np.arange(len(keys))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a2e")
    ax.bar(x - w/2, [metrics_v1.get(k, 0) for k in keys],
           width=w, color="#ff6e40", alpha=0.8, label="VAE v1")
    ax.bar(x + w/2, [metrics_v2.get(k, 0) for k in keys],
           width=w, color="#00e5ff", alpha=0.8, label="VAE v2 (improved)")
    ax.set_xticks(x); ax.set_xticklabels(keys, color="white", rotation=10)
    ax.set_title("VAE v1 vs v2 – Reconstruction Metrics",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────
# 8. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepFALCON VAE v2 – Task 1 Improved")
    parser.add_argument("--data",         type=str,   required=True)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch",        type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--latent",       type=int,   default=256)
    parser.add_argument("--beta",         type=float, default=1.0)
    parser.add_argument("--beta_warmup",  type=int,   default=10)
    parser.add_argument("--base_ch",      type=int,   default=32)
    parser.add_argument("--ssim_weight",  type=float, default=0.3,
                        help="Weight for SSIM in reconstruction loss")
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--max_samples",  type=int,   default=None)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--augment",      action="store_true",
                        help="Enable data augmentation")
    parser.add_argument("--out_dir",      type=str,   default="outputs_v2")
    parser.add_argument("--ckpt_dir",     type=str,   default="checkpoints_v2")
    parser.add_argument("--resume",       type=str,   default=None)
    # v1 metrics for comparison (optional)
    parser.add_argument("--v1_mse",       type=float, default=None)
    parser.add_argument("--v1_mae",       type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir,  exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────
    full_ds  = JetImageDataset(args.data, max_samples=args.max_samples,
                                augment=False)   # no augment on val/test
    train_ds_raw = JetImageDataset(args.data,
                                    max_samples=args.max_samples,
                                    augment=args.augment)

    n_tr = int(0.80 * len(full_ds))
    n_vl = int(0.10 * len(full_ds))
    n_te = len(full_ds) - n_tr - n_vl

    # Use augmented version for train split
    _, val_ds, test_ds = random_split(
        full_ds, [n_tr, n_vl, n_te],
        generator=torch.Generator().manual_seed(SEED))
    train_ds, _, _ = random_split(
        train_ds_raw, [n_tr, n_vl, n_te],
        generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, args.batch, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   args.batch, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  args.batch, shuffle=False,
                              num_workers=NUM_WORKERS)

    print(f"Train/Val/Test: {n_tr}/{n_vl}/{n_te}")

    # ── Model ────────────────────────────────────────────────────────
    model    = VAEv2(in_ch=3, latent_dim=args.latent, base_ch=args.base_ch,
                     n_classes=2, dropout=args.dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAE v2 parameters: {n_params:,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        print(f"Resumed from {args.resume}")

    # ── Train ────────────────────────────────────────────────────────
    history = train_vae(
        model, train_loader, val_loader,
        n_epochs=args.epochs, lr=args.lr,
        beta=args.beta, beta_warmup=args.beta_warmup,
        ssim_weight=args.ssim_weight,
        save_dir=args.ckpt_dir, patience=args.patience)

    model.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, "vae_v2_best.pt"),
                   map_location=DEVICE))

    # ── Metrics ──────────────────────────────────────────────────────
    metrics_v2 = compute_metrics(model, test_ds, n_samples=500)

    # ── Plots ────────────────────────────────────────────────────────
    plot_loss_curves(history,
                     os.path.join(args.out_dir, "loss_curves.png"))

    plot_original_vs_recon(model, test_ds, n_events=6, label_filter=1,
                            title="Quark Jets – VAE v2 Reconstruction",
                            out_path=os.path.join(args.out_dir,
                                                   "recon_quarks.png"))
    plot_original_vs_recon(model, test_ds, n_events=6, label_filter=0,
                            title="Gluon Jets – VAE v2 Reconstruction",
                            out_path=os.path.join(args.out_dir,
                                                   "recon_gluons.png"))

    plot_latent_space(model, test_loader,
                      os.path.join(args.out_dir, "latent_space.png"))

    plot_channel_histograms(model, test_ds,
                             os.path.join(args.out_dir,
                                          "pixel_histograms.png"))

    plot_generated_samples(model,
                            os.path.join(args.out_dir, "generated.png"))

    # Compare v1 vs v2 if v1 metrics provided
    if args.v1_mse is not None:
        metrics_v1 = {"MSE": args.v1_mse, "MAE": args.v1_mae or 0,
                      "W1_ECAL": 0, "W1_HCAL": 0, "W1_Tracks": 0}
        plot_metrics_bar(metrics_v1, metrics_v2,
                         os.path.join(args.out_dir, "metrics_comparison.png"))

    print(f"\n✓ Done. Outputs saved to: {args.out_dir}")
    print(f"  MSE={metrics_v2['MSE']:.6f}  "
          f"MAE={metrics_v2['MAE']:.6f}  "
          f"PSNR={metrics_v2['PSNR']:.2f}dB")
