"""
DeepFALCON GSoC 2026 – Common Task 2
Graph Neural Network for Quark/Gluon Jet Classification

Pipeline:
  1. Image → Point Cloud  (keep non-zero pixels only)
  2. Point Cloud → Graph   (k-NN edges in (η,φ,E) space)
  3. GNN (EdgeConv / DGCNN-style) → binary classification

Author: GSoC Candidate
"""

# ─────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────
import os, random, time, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# PyG (PyTorch Geometric)
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (EdgeConv, global_mean_pool,
                                 global_max_pool, BatchNorm)
from torch_geometric.utils import to_undirected
from torch_cluster import knn_graph

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────
# 1. Image → Point Cloud
# ─────────────────────────────────────────────────────────
"""
Each event has 3 channels (ECAL, HCAL, Tracks), each 125×125.
Pixel (i, j) maps to detector coordinates:
  η (pseudorapidity) ∝ row index i   (range ≈ [-1.3, 1.3])
  φ (azimuthal angle) ∝ col index j  (range ≈ [-π, π])

For each event we:
  1. Union non-zero pixels across all 3 channels
  2. Each pixel becomes a NODE with features:
       [η, φ, E_ECAL, E_HCAL, E_Tracks, E_total, ΔR_from_centroid]
  3. Build k-NN graph in (η, φ, E_total) space
"""

IMG_SIZE = 125
ETA_RANGE = 2.6     # total pseudorapidity window
PHI_RANGE = 2 * math.pi

def image_to_pointcloud(img: np.ndarray, threshold: float = 0.0):
    """
    img: (3, 125, 125) float32  – log-normalised, [0,1]
    Returns:
        coords   (N, 2)  – (η, φ) of non-zero pixels
        features (N, 7)  – [η, φ, E_ECAL, E_HCAL, E_Tracks, E_tot, ΔR]
    Returns None if event has 0 non-zero pixels.
    """
    C, H, W = img.shape   # 3, 125, 125

    # pixel → η, φ
    rows = np.arange(H)
    cols = np.arange(W)
    eta_vals = (rows / (H - 1)) * ETA_RANGE - ETA_RANGE / 2   # [-1.3, 1.3]
    phi_vals = (cols / (W - 1)) * PHI_RANGE - PHI_RANGE / 2   # [-π,   π ]

    # Non-zero mask: pixel active in ANY channel
    mask = np.any(img > threshold, axis=0)   # (H, W) bool
    ri, ci = np.where(mask)

    if len(ri) == 0:
        return None, None

    eta = eta_vals[ri]          # (N,)
    phi = phi_vals[ci]          # (N,)
    e_ecal  = img[0, ri, ci]    # (N,)
    e_hcal  = img[1, ri, ci]    # (N,)
    e_track = img[2, ri, ci]    # (N,)
    e_tot   = e_ecal + e_hcal + e_track

    # Energy-weighted centroid
    if e_tot.sum() > 0:
        eta_c = np.sum(eta * e_tot) / e_tot.sum()
        phi_c = np.sum(phi * e_tot) / e_tot.sum()
    else:
        eta_c, phi_c = 0.0, 0.0

    delta_r = np.sqrt((eta - eta_c)**2 + (phi - phi_c)**2)

    features = np.stack([eta, phi, e_ecal, e_hcal, e_track, e_tot, delta_r],
                         axis=1).astype(np.float32)   # (N, 7)
    coords   = np.stack([eta, phi], axis=1).astype(np.float32)   # (N, 2)
    return coords, features


# ─────────────────────────────────────────────────────────
# 2. Point Cloud → PyG Graph
# ─────────────────────────────────────────────────────────
def pointcloud_to_graph(features: np.ndarray, k: int = 8,
                         max_nodes: int = 400) -> Data:
    """
    features: (N, 7)
    Edge strategy: k-NN in (η, φ, E_tot) space
      – spatial proximity captures clustering in the jet
      – energy dimension groups deposits of similar magnitude
    Returns a PyG Data object.
    """
    # Subsample if too many nodes (rare but keeps batch sizes sane)
    if features.shape[0] > max_nodes:
        idx = np.random.choice(features.shape[0], max_nodes, replace=False)
        features = features[idx]

    x   = torch.tensor(features, dtype=torch.float32)            # (N, 7)
    # Use (η, φ, E_tot) as the 3-D space for k-NN
    pos = x[:, [0, 1, 5]]                                        # (N, 3)

    # k-NN graph (directed; we symmetrize below)
    edge_index = knn_graph(pos, k=k, loop=False)                 # (2, N*k)
    edge_index = to_undirected(edge_index)                       # symmetrize

    # Edge features: Δη, Δφ, ΔE_tot, ΔR (spatial distance)
    src, dst = edge_index
    d_eta  = x[dst, 0] - x[src, 0]
    d_phi  = x[dst, 1] - x[src, 1]
    d_etot = x[dst, 5] - x[src, 5]
    d_r    = torch.sqrt(d_eta**2 + d_phi**2)
    edge_attr = torch.stack([d_eta, d_phi, d_etot, d_r], dim=1) # (E, 4)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


# ─────────────────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────────────────
class JetGraphDataset(Dataset):
    """
    Converts raw HDF5 jet images to PyG graphs on-the-fly (with cache).
    """
    def __init__(self, filepath: str, k: int = 8, max_nodes: int = 400,
                 max_samples: int | None = None, threshold: float = 0.0):
        super().__init__()
        self.k = k; self.max_nodes = max_nodes; self.threshold = threshold

        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            x_key = "X" if "X" in keys else "jetImage"
            y_key = "y" if "y" in keys else "jetLabel"
            X = f[x_key][:]
            Y = f[y_key][:]

        if X.ndim == 4 and X.shape[-1] == 3:
            X = X.transpose(0, 3, 1, 2)          # (N,3,H,W)
        X = X.astype(np.float32)
        X = np.log1p(X)
        for c in range(3):
            p = np.percentile(X[:, c], 99)
            if p > 0: X[:, c] = np.clip(X[:, c] / p, 0, 1)

        if max_samples:
            X = X[:max_samples]; Y = Y[:max_samples]

        self.X = X
        self.Y = Y.astype(np.int64)
        print(f"Dataset: {len(self.X)} events loaded.")

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        lbl = int(self.Y[idx])
        coords, feats = image_to_pointcloud(img, self.threshold)
        if feats is None:
            # degenerate event: single zero-node graph
            feats = np.zeros((1, 7), dtype=np.float32)
        graph = pointcloud_to_graph(feats, k=self.k, max_nodes=self.max_nodes)
        graph.y = torch.tensor([lbl], dtype=torch.long)
        return graph


def collate_fn(batch):
    return Batch.from_data_list(batch)


# ─────────────────────────────────────────────────────────
# 4. GNN Model — DGCNN (Dynamic Graph CNN) with EdgeConv
# ─────────────────────────────────────────────────────────
"""
Architecture: ParticleNet-inspired DGCNN
  3 × EdgeConv blocks with increasing channel widths
  Each EdgeConv: MLP on (x_i ‖ x_j − x_i) pairs → new node features
  Aggregation: cat(GlobalMeanPool, GlobalMaxPool)
  Classifier: FC → FC → output (2 classes)

Why DGCNN / EdgeConv?
  – EdgeConv considers BOTH the node's own features AND the relative
    features to its neighbors: captures local jet substructure well
  – The graph is rebuilt dynamically in feature space after each block
    (here we keep the original graph for simplicity, a standard variant)
  – ParticleNet (the HEP adaptation of DGCNN) is the SOTA baseline
    for jet tagging from point clouds
"""

class MLP(nn.Module):
    def __init__(self, dims: list[int], act=nn.LeakyReLU(0.2)):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1], bias=False),
                       nn.BatchNorm1d(dims[i+1]),
                       act]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class EdgeConvBlock(nn.Module):
    """
    Standard EdgeConv:
      h_i = Aggreg_j( MLP(x_i ‖ x_j - x_i) )
    Input dim: in_ch  (per node)
    Output dim: out_ch (per node)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # EdgeConv expects MLP that maps 2*in_ch → out_ch
        self.conv = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_ch, out_ch, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Linear(out_ch, out_ch, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2),
            ),
            aggr="max"
        )
        # Skip connection with 1×1 conv if dims differ
        self.skip = (nn.Sequential(
                        nn.Linear(in_ch, out_ch, bias=False),
                        nn.BatchNorm1d(out_ch))
                     if in_ch != out_ch else nn.Identity())
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        return self.act(self.conv(x, edge_index) + self.skip(x))


class JetGNN(nn.Module):
    """
    DGCNN-style classifier for quark/gluon jets.

    Node input features (7):
      [η, φ, E_ECAL, E_HCAL, E_Tracks, E_tot, ΔR]

    Architecture:
      EdgeConv(7  → 64)
      EdgeConv(64 → 128)
      EdgeConv(128 → 256)
      ↓
      cat(GlobalMeanPool, GlobalMaxPool)   → 512-d
      FC(512 → 256) → Dropout(0.3)
      FC(256 → 128) → Dropout(0.3)
      FC(128 → 2)   → log-softmax
    """
    def __init__(self, in_ch: int = 7, dropout: float = 0.3):
        super().__init__()
        self.ec1 = EdgeConvBlock(in_ch, 64)
        self.ec2 = EdgeConvBlock(64,   128)
        self.ec3 = EdgeConvBlock(128,  256)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.ec1(x, edge_index)
        x = self.ec2(x, edge_index)
        x = self.ec3(x, edge_index)

        # Dual pooling: mean + max captures both average and peak activity
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)   # (B, 512)

        return self.classifier(x)   # (B, 2) — raw logits


# ─────────────────────────────────────────────────────────
# 5. Training & Evaluation
# ─────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch)
        loss   = F.cross_entropy(logits, batch.y.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        preds = logits.argmax(dim=1)
        total_loss    += loss.item() * batch.num_graphs
        total_correct += (preds == batch.y.squeeze()).sum().item()
        total_n       += batch.num_graphs
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = total_correct = total_n = 0
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss   = F.cross_entropy(logits, batch.y.squeeze())
        probs  = F.softmax(logits, dim=1)[:, 1]   # P(quark)
        preds  = logits.argmax(dim=1)
        total_loss    += loss.item() * batch.num_graphs
        total_correct += (preds == batch.y.squeeze()).sum().item()
        total_n       += batch.num_graphs
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch.y.squeeze().cpu().numpy())
    loss_avg = total_loss / total_n
    acc      = total_correct / total_n
    probs_np = np.concatenate(all_probs)
    labs_np  = np.concatenate(all_labels)
    auc      = roc_auc_score(labs_np, probs_np)
    return loss_avg, acc, auc, probs_np, labs_np


def train(model, train_loader, val_loader, n_epochs=50, lr=1e-3,
          save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs,
                                                        eta_min=1e-5)
    history = {k: [] for k in
               ["train_loss","train_acc","val_loss","val_acc","val_auc"]}
    best_auc = 0.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, DEVICE)
        vl_loss, vl_acc, vl_auc, _, _ = eval_epoch(model, val_loader, DEVICE)
        sched.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "gnn_best.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Ep {epoch:3d}/{n_epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {vl_loss:.4f} acc {vl_acc:.4f} AUC {vl_auc:.4f} | "
                  f"{time.time()-t0:.1f}s")

    return history


# ─────────────────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────────────────
def plot_training_curves(history, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#0d0d0d")
    epochs = range(1, len(history["train_loss"])+1)
    specs = [
        ("Loss",     "train_loss","val_loss"),
        ("Accuracy", "train_acc", "val_acc"),
        ("Val AUC",  "val_auc",   None),
    ]
    for ax, (title, tk, vk) in zip(axes, specs):
        ax.plot(epochs, history[tk], color="#00e5ff", lw=1.8, label="Train")
        if vk: ax.plot(epochs, history[vk], color="#ff6e40", lw=1.8,
                       ls="--", label="Val")
        ax.set_title(title, color="white"); ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white"); ax.set_xlabel("Epoch", color="#aaa")
        if vk: ax.legend(facecolor="#1a1a2e", labelcolor="white")
    fig.suptitle("GNN Training Curves", color="white", fontsize=13)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_roc(labels, probs, out):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a2e")
    ax.plot(fpr, tpr, color="#00e5ff", lw=2, label=f"GNN  AUC={auc:.4f}")
    ax.plot([0,1],[0,1], color="#555", ls="--", lw=1)
    ax.set_xlabel("False Positive Rate", color="#aaa")
    ax.set_ylabel("True Positive Rate",  color="#aaa")
    ax.set_title("ROC Curve – Quark/Gluon", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_confusion(labels, probs, out, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Gluon","Quark"], color="white")
    ax.set_yticklabels(["Gluon","Quark"], color="white")
    ax.set_xlabel("Predicted", color="#aaa")
    ax.set_ylabel("True",      color="#aaa")
    ax.set_title("Confusion Matrix", color="white")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] < cm.max()*0.6 else "black",
                    fontsize=14)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_score_distribution(labels, probs, out):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a2e")
    bins = np.linspace(0, 1, 50)
    ax.hist(probs[labels==0], bins=bins, density=True, alpha=0.6,
            color="#ff6e40", label="Gluon", histtype="stepfilled")
    ax.hist(probs[labels==1], bins=bins, density=True, alpha=0.6,
            color="#00e5ff", label="Quark", histtype="stepfilled")
    ax.set_xlabel("P(Quark)", color="#aaa")
    ax.set_ylabel("Density",  color="#aaa")
    ax.set_title("Classifier Score Distribution", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_jet_graph(dataset, idx=0, out="jet_graph_vis.png"):
    """Visualise a single jet as a graph (nodes in η-φ plane)."""
    from torch_geometric.utils import to_networkx
    import networkx as nx

    graph = dataset[idx]
    label_map = {0: "Gluon", 1: "Quark"}
    lbl = label_map.get(graph.y.item(), "?")

    pos_np  = graph.x[:, [0,1]].numpy()   # (η, φ)
    e_tot   = graph.x[:, 5].numpy()       # E_tot per node
    e_norm  = (e_tot - e_tot.min()) / (e_tot.max() - e_tot.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d0d0d")

    # Left: scatter in η-φ coloured by E_tot
    ax = axes[0]; ax.set_facecolor("#1a1a2e")
    sc = ax.scatter(pos_np[:,0], pos_np[:,1], c=e_tot, cmap="inferno",
                    s=10, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="E_tot")
    ax.set_xlabel("η", color="#aaa"); ax.set_ylabel("φ", color="#aaa")
    ax.set_title(f"Point Cloud – {lbl} Jet\n{len(pos_np)} nodes",
                 color="white")
    ax.tick_params(colors="white")

    # Right: graph edges in η-φ
    ax2 = axes[1]; ax2.set_facecolor("#1a1a2e")
    ei  = graph.edge_index.numpy()
    # Draw edges
    for s, d in zip(ei[0], ei[1]):
        ax2.plot([pos_np[s,0], pos_np[d,0]],
                 [pos_np[s,1], pos_np[d,1]],
                 color="#334455", lw=0.3, alpha=0.5)
    ax2.scatter(pos_np[:,0], pos_np[:,1], c=e_tot, cmap="inferno",
                s=8, alpha=0.9, zorder=5)
    ax2.set_xlabel("η", color="#aaa"); ax2.set_ylabel("φ", color="#aaa")
    ax2.set_title(f"k-NN Graph – {lbl} Jet\n{ei.shape[1]} edges",
                  color="white")
    ax2.tick_params(colors="white")

    fig.suptitle("Jet Graph Visualisation", color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


def plot_pointcloud_stats(dataset, n=2000, out="pointcloud_stats.png"):
    """Distribution of number of nodes and mean ΔR per event."""
    n_nodes, mean_dr, labels = [], [], []
    for i in range(min(n, len(dataset))):
        g = dataset[i]
        n_nodes.append(g.x.shape[0])
        mean_dr.append(g.x[:, 6].mean().item())
        labels.append(g.y.item())
    labels  = np.array(labels)
    n_nodes = np.array(n_nodes)
    mean_dr = np.array(mean_dr)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#0d0d0d")
    for ax, data, title, xlabel in zip(
        axes,
        [n_nodes, mean_dr],
        ["Nodes per event", "Mean ΔR per event"],
        ["# nodes", "ΔR"]
    ):
        ax.set_facecolor("#1a1a2e"); ax.tick_params(colors="white")
        bins = 40
        ax.hist(data[labels==0], bins=bins, density=True, alpha=0.6,
                color="#ff6e40", label="Gluon", histtype="stepfilled")
        ax.hist(data[labels==1], bins=bins, density=True, alpha=0.6,
                color="#00e5ff", label="Quark", histtype="stepfilled")
        ax.set_title(title, color="white")
        ax.set_xlabel(xlabel, color="#aaa")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    fig.suptitle("Point Cloud Statistics", color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────
# 7. Entry Point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepFALCON GNN – Task 2")
    parser.add_argument("--data",        type=str,   required=True)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch",       type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--k",           type=int,   default=8,
                        help="k for k-NN graph construction")
    parser.add_argument("--max_nodes",   type=int,   default=400)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--max_samples", type=int,   default=None)
    parser.add_argument("--out_dir",     type=str,   default="outputs_gnn")
    parser.add_argument("--ckpt_dir",    type=str,   default="checkpoints_gnn")
    parser.add_argument("--resume",      type=str,   default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Build dataset ────────────────────────────────────
    full_ds = JetGraphDataset(args.data, k=args.k, max_nodes=args.max_nodes,
                               max_samples=args.max_samples)

    n_tr = int(0.80 * len(full_ds))
    n_vl = int(0.10 * len(full_ds))
    n_te = len(full_ds) - n_tr - n_vl
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_tr, n_vl, n_te],
        generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2)

    print(f"Train/Val/Test: {n_tr}/{n_vl}/{n_te}")

    # ── Point cloud stats ────────────────────────────────
    plot_pointcloud_stats(full_ds, n=2000,
                          out=os.path.join(args.out_dir, "pointcloud_stats.png"))

    # ── Graph visualisation (first quark + first gluon) ──
    plot_jet_graph(full_ds, idx=0,
                   out=os.path.join(args.out_dir, "jet_graph_vis.png"))

    # ── Model ─────────────────────────────────────────────
    model = JetGNN(in_ch=7, dropout=args.dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"JetGNN parameters: {n_params:,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))

    # ── Train ─────────────────────────────────────────────
    history = train(model, train_loader, val_loader,
                    n_epochs=args.epochs, lr=args.lr,
                    save_dir=args.ckpt_dir)

    # Load best
    model.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, "gnn_best.pt"),
                   map_location=DEVICE))

    # ── Test evaluation ───────────────────────────────────
    te_loss, te_acc, te_auc, te_probs, te_labels = eval_epoch(
        model, test_loader, DEVICE)
    print(f"\n{'='*55}")
    print(f"  Test Loss : {te_loss:.4f}")
    print(f"  Test Acc  : {te_acc:.4f}  ({te_acc*100:.2f}%)")
    print(f"  Test AUC  : {te_auc:.4f}")
    print(f"{'='*55}")
    print(classification_report(te_labels, (te_probs >= 0.5).astype(int),
                                 target_names=["Gluon","Quark"]))

    # ── Plots ─────────────────────────────────────────────
    plot_training_curves(history,
                         os.path.join(args.out_dir, "training_curves.png"))
    plot_roc(te_labels, te_probs,
             os.path.join(args.out_dir, "roc_curve.png"))
    plot_confusion(te_labels, te_probs,
                   os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_score_distribution(te_labels, te_probs,
                             os.path.join(args.out_dir, "score_dist.png"))

    print(f"\n✓ All outputs saved to: {args.out_dir}")
