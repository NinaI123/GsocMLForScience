# DeepFALCON — GSoC 2026 · ML4SCI
**Applicant:** Sakina Ismail · si378@drexel.edu · [github.com/NinaI123](https://github.com/NinaI123)  
**Project:** Diffusion Models for Fast Detector Simulation  
**Dataset:** Quark/Gluon jet images — 139,306 events · 3 × 125 × 125 (ECAL, HCAL, Tracks)

---

## Results Notebook

Open `DeepFALCON_GSoC2026_SakinaIsmail.ipynb` to view all results — it is fully pre-executed with every plot and metric already embedded. No installation or dataset needed to read it.

---

## Tasks Completed

| Task | File | Description |
|------|------|-------------|
| Common Task 1 | `deepfalcon_vae.py` | Conditional U-Net VAE with skip connections, SE attention, PixelShuffle |
| Common Task 2 | `deepfalcon_gnn.py` | Dynamic Graph CNN (EdgeConv) on k-NN jet graphs |
| Specific Task 2 | `deepfalcon_diffusion.py` | DDPM with cosine schedule, sinusoidal time embeddings |

---

## Key Results

| Model | MSE | PSNR | AUC |
|-------|-----|------|-----|
| VAE v2 | 0.00697 | 21.56 dB | — |
| GNN | — | — | 0.697 |
| **DDPM** | **0.000602** | **32.20 dB** | — |

DDPM W1 mean: **0.000840** across all three calorimeter channels.

---

## Reproducing the Results

### Requirements
```bash
pip install torch numpy h5py matplotlib scikit-learn tqdm torch-geometric
```

### Common Task 1 — VAE
```bash
python deepfalcon_vae.py \
  --data path/to/quark-gluon_data-set_n139306.hdf5 \
  --epochs 50 \
  --max_samples 20000 \
  --base_ch 16 \
  --latent 128
```
Outputs saved to `outputs/`

### Common Task 2 — GNN
```bash
python deepfalcon_gnn.py \
  --data path/to/quark-gluon_data-set_n139306.hdf5 \
  --epochs 50 \
  --max_samples 5000 \
  --k 8 \
  --batch 32
```
Outputs saved to `outputs_gnn/`

### Specific Task 2 — DDPM
```bash
python deepfalcon_diffusion.py \
  --data path/to/quark-gluon_data-set_n139306.hdf5 \
  --epochs 50 \
  --max_samples 5000 \
  --base_ch 32 \
  --batch 32 \
  --T 1000 \
  --t_recon 150 \
  --img_size 64
```
Outputs saved to `outputs_diff/`

> All three scripts were run on CPU. Training times: VAE ~340s/epoch, GNN ~1270s/epoch, DDPM ~70s/epoch. GPU will be significantly faster.

---

## Repository Structure

```
├── DeepFALCON_GSoC2026_SakinaIsmail.ipynb   ← pre-executed results notebook
├── deepfalcon_vae.py                          ← Task 1: Conditional VAE
├── deepfalcon_gnn.py                          ← Task 2: GNN classifier
├── deepfalcon_diffusion.py                    ← Specific Task 2: DDPM
├── outputs/                                   ← VAE plots
├── outputs_gnn/                               ← GNN plots
└── outputs_diff/                              ← DDPM plots
```

---

## Dataset

The quark/gluon jet dataset is available from the ML4SCI DeepFALCON project page.  
HDF5 keys used: `X_jets` (images), `y` (labels), `pt`, `m0`.
