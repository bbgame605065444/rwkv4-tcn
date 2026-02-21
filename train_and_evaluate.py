"""
Training & evaluation pipeline for MSRWKV-2DTCN on synthetic PV data.

Generates realistic synthetic solar PV + meteorological data, properly
splits into train/val/test (chronological, no leakage), normalises using
only training statistics, trains for 1 epoch, and produces evaluation plots.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from msrwkv_2dtcn import MSRWKV2DTCN

# ─── reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── hyper-parameters ────────────────────────────────────────────────────────
SEQ_LEN = 168        # 7 days of hourly data as input context
PRED_LEN = 24        # predict next 24 hours
NUM_FEATURES = 7     # GHI, DNI, DHI, temperature, humidity, wind_speed, PV_power
TOTAL_HOURS = 365 * 24  # 1 year of hourly data (8760 samples)
BATCH_SIZE = 32
LR = 1e-3
D_MODEL = 64
N_LAYERS = 2
TOP_K_PERIODS = 3
TCN_CHANNELS = 32
TCN_LAYERS = 2

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════════════════════
def generate_synthetic_pv_data(n_hours: int = TOTAL_HOURS) -> np.ndarray:
    """
    Generate realistic synthetic PV + meteorological hourly data.

    Features (7 columns):
        0: GHI  (Global Horizontal Irradiance, W/m^2)
        1: DNI  (Direct Normal Irradiance, W/m^2)
        2: DHI  (Diffuse Horizontal Irradiance, W/m^2)
        3: Temperature (°C)
        4: Humidity (%)
        5: Wind Speed (m/s)
        6: PV Power (kW)  — target
    """
    t = np.arange(n_hours, dtype=np.float64)
    hour_of_day = t % 24
    day_of_year = (t // 24) % 365

    # --- Solar geometry approximation ---
    # Declination angle (seasonal variation)
    declination = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
    # Hour angle
    hour_angle = (hour_of_day - 12) * 15  # degrees
    # Solar elevation (simplified, latitude ~ 35°N)
    latitude = 35.0
    sin_elev = (np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
                np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) *
                np.cos(np.radians(hour_angle)))
    sin_elev = np.clip(sin_elev, 0, 1)

    # --- GHI ---
    clear_sky_ghi = 1000 * sin_elev
    # Cloud factor: slow-varying + random
    cloud = 0.7 + 0.3 * np.sin(2 * np.pi * day_of_year / 365 + 1.2)
    cloud = cloud + 0.1 * np.random.randn(n_hours)
    cloud = np.clip(cloud, 0.2, 1.0)
    ghi = clear_sky_ghi * cloud + np.random.randn(n_hours) * 10
    ghi = np.clip(ghi, 0, 1200)

    # --- DNI, DHI from GHI ---
    dni = ghi * (0.6 + 0.2 * np.random.rand(n_hours))
    dni = np.clip(dni, 0, 1000)
    dhi = ghi - dni * sin_elev
    dhi = np.clip(dhi, 0, 500)

    # --- Temperature ---
    temp_seasonal = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    temp_daily = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    temp = temp_seasonal + temp_daily + np.random.randn(n_hours) * 2

    # --- Humidity ---
    humidity = 60 - 15 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    humidity = humidity + 10 * np.sin(2 * np.pi * hour_of_day / 24 + 2)
    humidity = humidity + np.random.randn(n_hours) * 5
    humidity = np.clip(humidity, 10, 100)

    # --- Wind Speed ---
    wind = 3 + 2 * np.sin(2 * np.pi * day_of_year / 365)
    wind = wind + 1.5 * np.sin(2 * np.pi * hour_of_day / 24)
    wind = wind + np.abs(np.random.randn(n_hours)) * 1.5
    wind = np.clip(wind, 0, 20)

    # --- PV Power (target) ---
    # PV ~ f(GHI, temp, wind) with nonlinear effects
    panel_eff = 0.18  # base efficiency
    temp_coeff = -0.004  # efficiency drops with temperature
    eff = panel_eff + temp_coeff * (temp - 25)
    eff = np.clip(eff, 0.05, 0.25)
    pv_power = ghi * eff * 10  # 10 m^2 panel area, output in W -> /1000 -> kW
    pv_power = pv_power / 1000
    # Add noise & clip
    pv_power = pv_power + np.random.randn(n_hours) * 0.02
    pv_power = np.clip(pv_power, 0, None)

    data = np.stack([ghi, dni, dhi, temp, humidity, wind, pv_power], axis=1)
    return data.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ═══════════════════════════════════════════════════════════════════════════════
class PVDataset(Dataset):
    """Sliding window dataset for sequence-to-sequence forecasting."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        """
        data: (N, num_features) numpy array, already normalised
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]             # (seq_len, F)
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, -1:]  # (pred_len, 1)
        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("MSRWKV-2DTCN  —  PV Power Forecasting Pipeline")
    print("=" * 60)

    # ── generate data ────────────────────────────────────────────────────────
    print("\n[1/6] Generating synthetic PV data ...")
    raw_data = generate_synthetic_pv_data(TOTAL_HOURS)
    print(f"  Data shape: {raw_data.shape}  (hours × features)")

    # ── chronological split (70 / 15 / 15) ──────────────────────────────────
    print("\n[2/6] Splitting data chronologically (70/15/15) ...")
    n = len(raw_data)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    # n_test = n - n_train - n_val

    train_raw = raw_data[:n_train]
    val_raw = raw_data[n_train: n_train + n_val]
    test_raw = raw_data[n_train + n_val:]

    print(f"  Train: {len(train_raw)}  Val: {len(val_raw)}  Test: {len(test_raw)}")

    # ── normalise using ONLY training statistics (prevent leakage) ───────────
    print("\n[3/6] Normalising with training statistics only ...")
    train_mean = train_raw.mean(axis=0)
    train_std = train_raw.std(axis=0) + 1e-8

    train_norm = (train_raw - train_mean) / train_std
    val_norm = (val_raw - train_mean) / train_std
    test_norm = (test_raw - train_mean) / train_std

    # ── create datasets & loaders ────────────────────────────────────────────
    train_ds = PVDataset(train_norm, SEQ_LEN, PRED_LEN)
    val_ds = PVDataset(val_norm, SEQ_LEN, PRED_LEN)
    test_ds = PVDataset(test_norm, SEQ_LEN, PRED_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train batches: {len(train_loader)}  "
          f"Val batches: {len(val_loader)}  "
          f"Test batches: {len(test_loader)}")

    # ── build model ──────────────────────────────────────────────────────────
    print("\n[4/6] Building MSRWKV-2DTCN model ...")
    model = MSRWKV2DTCN(
        input_dim=NUM_FEATURES,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        pred_len=PRED_LEN,
        top_k_periods=TOP_K_PERIODS,
        tcn_channels=TCN_CHANNELS,
        tcn_layers=TCN_LAYERS,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {DEVICE}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── train for 1 epoch ────────────────────────────────────────────────────
    print("\n[5/6] Training for 1 epoch ...")
    model.train()
    batch_losses = []

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_losses.append(loss.item())
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"    batch {batch_idx + 1}/{len(train_loader)}  loss={loss.item():.6f}")

    epoch_loss = np.mean(batch_losses)
    print(f"  Epoch loss (mean): {epoch_loss:.6f}")

    # ── evaluate on test set ─────────────────────────────────────────────────
    print("\n[6/6] Evaluating on test set ...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds_norm = np.concatenate(all_preds, axis=0)   # (N, pred_len, 1)
    targets_norm = np.concatenate(all_targets, axis=0)

    # De-normalise (PV power is the last feature, index 6)
    pv_mean = train_mean[-1]
    pv_std = train_std[-1]
    preds_real = preds_norm * pv_std + pv_mean
    targets_real = targets_norm * pv_std + pv_mean

    # Flatten for metrics
    preds_flat = preds_real.flatten()
    targets_flat = targets_real.flatten()

    mae = mean_absolute_error(targets_flat, preds_flat)
    mse = mean_squared_error(targets_flat, preds_flat)
    r2 = r2_score(targets_flat, preds_flat)

    print(f"\n{'='*40}")
    print(f"  Test MAE  : {mae:.6f} kW")
    print(f"  Test MSE  : {mse:.6f} kW²")
    print(f"  Test RMSE : {math.sqrt(mse):.6f} kW")
    print(f"  Test R²   : {r2:.6f}")
    print(f"{'='*40}")

    # ═════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\nGenerating plots ...")

    # --- Plot 1: Training Loss Curve ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(batch_losses, linewidth=0.8, color="#1f77b4")
    ax.set_xlabel("Batch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss (1 Epoch)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/loss_curve.png")

    # --- Plot 2: Prediction vs Actual (scatter) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    subsample = max(1, len(preds_flat) // 5000)  # subsample for readability
    ax.scatter(targets_flat[::subsample], preds_flat[::subsample],
               alpha=0.3, s=6, color="#2ca02c")
    lo = min(targets_flat.min(), preds_flat.min())
    hi = max(targets_flat.max(), preds_flat.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Ideal (y=x)")
    ax.set_xlabel("Actual PV Power (kW)")
    ax.set_ylabel("Predicted PV Power (kW)")
    ax.set_title(f"Pred vs Actual  |  R²={r2:.4f}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pred_vs_actual_scatter.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/pred_vs_actual_scatter.png")

    # --- Plot 3: Actual vs Predicted Time Series (first 5 days of test) ---
    n_show = min(120, len(targets_flat))  # 5 days * 24h
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(range(n_show), targets_flat[:n_show], label="Actual", linewidth=1.2, color="#1f77b4")
    ax.plot(range(n_show), preds_flat[:n_show], label="Predicted", linewidth=1.2,
            color="#ff7f0e", linestyle="--")
    ax.set_xlabel("Hour")
    ax.set_ylabel("PV Power (kW)")
    ax.set_title("Actual vs Predicted PV Power (Test Set — First 5 Days)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "actual_vs_pred_line.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/actual_vs_pred_line.png")

    # --- Plot 4: Metrics Bar Chart ---
    fig, ax = plt.subplots(figsize=(6, 4))
    metric_names = ["MAE (kW)", "MSE (kW²)", f"RMSE (kW)", "R²"]
    metric_vals = [mae, mse, math.sqrt(mse), r2]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(metric_names, metric_vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Test Set Evaluation Metrics")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "metrics_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/metrics_bar.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
