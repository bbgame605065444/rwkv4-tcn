"""
MSRWKV-2DTCN: Multi-Scale RWKV with 2D Temporal Convolutional Network
for Short-Term Photovoltaic Power Forecasting.

Based on RWKV-v4 architecture with:
  - FFT-based multi-period detection
  - Multi-scale Time Mixing (replacing single-scale RWKV time mixing)
  - Multi-scale 2D TCN (replacing RWKV channel mixing)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. FFT period detection
# ---------------------------------------------------------------------------
def detect_periods(x: torch.Tensor, top_k: int = 3) -> list[int]:
    """
    Use FFT to detect top-k dominant periods from a batch of sequences.

    Args:
        x: (B, T, C) tensor
        top_k: number of dominant periods to return

    Returns:
        List of period lengths (integers >= 2).
    """
    B, T, C = x.shape
    # Average over batch & channels -> (T,)
    x_mean = x.mean(dim=(0, 2))
    # Remove DC component
    x_mean = x_mean - x_mean.mean()
    # FFT
    freqs = torch.fft.rfft(x_mean)
    amplitudes = freqs.abs()
    # Ignore DC (index 0) and Nyquist
    amplitudes[0] = 0
    # Get top-k frequency indices
    _, top_indices = torch.topk(amplitudes, min(top_k, len(amplitudes) - 1))
    periods = []
    for idx in top_indices:
        idx_val = idx.item()
        if idx_val > 0:
            period = max(2, round(T / idx_val))
            if period not in periods and period < T:
                periods.append(period)
    # Fallback
    if len(periods) == 0:
        periods = [T // 4, T // 2]
    return periods[:top_k]


# ---------------------------------------------------------------------------
# 2. RWKV-v4 Token Shift helper
# ---------------------------------------------------------------------------
class TokenShift(nn.Module):
    """Lerp between current token and previous token (RWKV-v4 style)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mu = nn.Parameter(torch.ones(d_model) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        Returns shifted x of same shape.
        """
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))  # shift right, pad zero at t=0
        return self.mu * x + (1 - self.mu) * x_prev


# ---------------------------------------------------------------------------
# 3. RWKV-v4 Single-Scale Time Mixing
# ---------------------------------------------------------------------------
class RWKVTimeMixing(nn.Module):
    """
    RWKV-v4 Time Mixing block with linear attention (WKV operator).

    Computes:
        r_t = sigmoid(W_r * shift_r(x_t))
        k_t = W_k * shift_k(x_t)
        v_t = W_v * shift_v(x_t)
        wkv_t = (sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i  +  e^{u+k_t} v_t)
               / (sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i}       +  e^{u+k_t})
        out_t = W_o * (r_t * wkv_t)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model
        self.shift_r = TokenShift(d_model)
        self.shift_k = TokenShift(d_model)
        self.shift_v = TokenShift(d_model)

        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Learned decay w and bonus u (per-channel)
        self.w = nn.Parameter(torch.ones(d_model) * -0.5)  # log-space decay
        self.u = nn.Parameter(torch.zeros(d_model))  # bonus for current token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape

        r = torch.sigmoid(self.W_r(self.shift_r(x)))  # (B, T, D)
        k = self.W_k(self.shift_k(x))
        v = self.W_v(self.shift_v(x))

        # WKV computation (sequential for correctness)
        wkv = self._wkv(k, v, T, B, D)
        return self.W_o(r * wkv)

    def _wkv(self, k, v, T, B, D):
        """Compute WKV with recurrent formulation for numerical stability."""
        w = -torch.exp(self.w)  # Ensure decay is negative (stable)
        u = self.u

        out = torch.zeros_like(k)
        # State: numerator a, denominator b, max for stability p
        a = torch.zeros(B, D, device=k.device)
        b = torch.zeros(B, D, device=k.device)
        p = torch.full((B, D), -1e30, device=k.device)  # max exponent seen

        for t in range(T):
            kt = k[:, t, :]  # (B, D)
            vt = v[:, t, :]

            # Current token contribution: e^(u + kt)
            q = torch.max(p, u + kt)
            e1 = torch.exp(p - q)
            e2 = torch.exp(u + kt - q)
            num = e1 * a + e2 * vt
            den = e1 * b + e2
            out[:, t, :] = num / (den + 1e-8)

            # Update state for next step
            q2 = torch.max(p + w, kt)
            e1 = torch.exp(p + w - q2)
            e2 = torch.exp(kt - q2)
            a = e1 * a + e2 * vt
            b = e1 * b + e2
            p = q2

        return out


# ---------------------------------------------------------------------------
# 4. Multi-Scale Time Mixing
# ---------------------------------------------------------------------------
class MultiScaleTimeMixing(nn.Module):
    """
    Applies RWKV Time Mixing at multiple temporal scales derived from FFT
    periods, then adaptively aggregates results.

    For each period p, the sequence is reshaped into (B*num_segments, p, D),
    processed by a shared RWKV Time Mixing block, then reshaped back.
    """

    def __init__(self, d_model: int, max_scales: int = 3):
        super().__init__()
        self.d_model = d_model
        self.max_scales = max_scales
        # Shared time-mixing across scales for parameter efficiency
        self.time_mixing = RWKVTimeMixing(d_model)
        # Adaptive aggregation weights
        self.scale_attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, periods: list[int]) -> torch.Tensor:
        """
        x: (B, T, D)
        periods: list of detected period lengths
        """
        B, T, D = x.shape
        scale_outputs = []

        for p in periods:
            if p >= T:
                p = T
            # Pad sequence so length is divisible by period
            pad_len = (p - T % p) % p
            if pad_len > 0:
                x_padded = F.pad(x, (0, 0, 0, pad_len))
            else:
                x_padded = x

            T_padded = x_padded.shape[1]
            num_seg = T_padded // p

            # Reshape: (B, num_seg, p, D) -> (B*num_seg, p, D)
            x_reshape = x_padded.reshape(B, num_seg, p, D)
            x_reshape = x_reshape.reshape(B * num_seg, p, D)

            # Apply time mixing at this scale
            out = self.time_mixing(x_reshape)

            # Reshape back: (B, num_seg, p, D) -> (B, T_padded, D)
            out = out.reshape(B, num_seg, p, D).reshape(B, T_padded, D)

            # Remove padding
            out = out[:, :T, :]
            scale_outputs.append(out)

        if len(scale_outputs) == 1:
            return scale_outputs[0]

        # Stack: (B, T, D, num_scales)
        stacked = torch.stack(scale_outputs, dim=-1)
        # Attention weights per scale
        weights = self.scale_attn(stacked.permute(0, 1, 3, 2))  # (B, T, num_scales, 1)
        weights = F.softmax(weights, dim=2)  # softmax over scales
        # Weighted sum
        out = (stacked.permute(0, 1, 3, 2) * weights).sum(dim=2)  # (B, T, D)
        return out


# ---------------------------------------------------------------------------
# 5. 2D TCN Block with Dilated Causal Convolution
# ---------------------------------------------------------------------------
class DilatedCausalConv2d(nn.Module):
    """Single dilated causal 2D convolution block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple, dilation: int = 1):
        super().__init__()
        # Causal padding: pad only on the left/top (time axis)
        self.pad_t = (kernel_size[0] - 1) * dilation
        self.pad_c = (kernel_size[1] - 1) // 2  # symmetric padding on channel axis

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=(dilation, 1),
            padding=(0, self.pad_c),
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        # Residual projection if dimensions mismatch
        self.residual = (nn.Conv2d(in_channels, out_channels, 1, bias=False)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T, D_feat)"""
        # Causal pad on time dimension (left only)
        x_pad = F.pad(x, (0, 0, self.pad_t, 0))
        out = self.act(self.bn(self.conv(x_pad)))
        return out + self.residual(x)


class MultiScale2DTCN(nn.Module):
    """
    Multi-Scale 2D TCN module that replaces RWKV Channel Mixing.

    For each FFT-detected period, constructs a 2D representation
    (time_segments x period_length) and applies dilated causal convolutions
    with kernel sizes derived from the period, then aggregates.
    """

    def __init__(self, d_model: int, tcn_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.tcn_channels = tcn_channels
        self.num_layers = num_layers

        # Input projection: 1 -> tcn_channels
        self.input_conv = nn.Conv2d(1, tcn_channels, 1)
        # Stacked dilated convolutions
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.tcn_layers.append(
                DilatedCausalConv2d(
                    tcn_channels, tcn_channels,
                    kernel_size=(3, 3),
                    dilation=2 ** i
                )
            )
        # Output projection: tcn_channels -> 1
        self.output_conv = nn.Conv2d(tcn_channels, 1, 1)
        # Adaptive aggregation
        self.agg_linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, periods: list[int]) -> torch.Tensor:
        """
        x: (B, T, D)
        periods: detected period lengths
        Returns: (B, T, D)
        """
        B, T, D = x.shape
        scale_outputs = []

        for p in periods:
            if p >= T:
                p = T
            pad_len = (p - T % p) % p
            if pad_len > 0:
                x_padded = F.pad(x, (0, 0, 0, pad_len))
            else:
                x_padded = x
            T_padded = x_padded.shape[1]
            num_seg = T_padded // p

            # Reshape to 2D: (B*D, 1, num_seg, p)
            x_2d = x_padded.permute(0, 2, 1)  # (B, D, T_padded)
            x_2d = x_2d.reshape(B * D, 1, num_seg, p)

            # Apply TCN
            h = self.input_conv(x_2d)  # (B*D, C_tcn, num_seg, p)
            for layer in self.tcn_layers:
                h = layer(h)
            h = self.output_conv(h)  # (B*D, 1, num_seg, p)

            # Reshape back: (B, D, T_padded) -> (B, T_padded, D) -> trim
            h = h.reshape(B, D, num_seg, p).reshape(B, D, T_padded)
            h = h.permute(0, 2, 1)[:, :T, :]
            scale_outputs.append(h)

        if len(scale_outputs) == 1:
            return scale_outputs[0]

        stacked = torch.stack(scale_outputs, dim=-1)  # (B, T, D, num_scales)
        weights = self.agg_linear(stacked.permute(0, 1, 3, 2))  # (B, T, S, 1)
        weights = F.softmax(weights, dim=2)
        out = (stacked.permute(0, 1, 3, 2) * weights).sum(dim=2)
        return out


# ---------------------------------------------------------------------------
# 6. Full MSRWKV-2DTCN Block
# ---------------------------------------------------------------------------
class MSRWKVBlock(nn.Module):
    """
    Single MSRWKV-2DTCN block:
        LayerNorm -> Multi-Scale Time Mixing -> Residual
        LayerNorm -> Multi-Scale 2D TCN       -> Residual
    """

    def __init__(self, d_model: int, tcn_channels: int = 32, tcn_layers: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ms_time_mixing = MultiScaleTimeMixing(d_model)
        self.ms_2dtcn = MultiScale2DTCN(d_model, tcn_channels, tcn_layers)

    def forward(self, x: torch.Tensor, periods: list[int]) -> torch.Tensor:
        # Time mixing with residual
        x = x + self.ms_time_mixing(self.ln1(x), periods)
        # 2D TCN with residual
        x = x + self.ms_2dtcn(self.ln2(x), periods)
        return x


# ---------------------------------------------------------------------------
# 7. MSRWKV-2DTCN Full Model
# ---------------------------------------------------------------------------
class MSRWKV2DTCN(nn.Module):
    """
    MSRWKV-2DTCN for short-term PV power forecasting.

    Architecture:
        Input Embedding -> N x MSRWKVBlock -> Projection -> Forecast

    Args:
        input_dim:  number of input features (e.g., irradiance, temp, humidity, ...)
        d_model:    hidden dimension
        n_layers:   number of MSRWKV blocks
        pred_len:   prediction horizon length
        top_k_periods: number of FFT periods to detect
        tcn_channels:  internal TCN channel width
        tcn_layers:    number of dilated conv layers in TCN
    """

    def __init__(self, input_dim: int = 7, d_model: int = 64, n_layers: int = 2,
                 pred_len: int = 24, top_k_periods: int = 3,
                 tcn_channels: int = 32, tcn_layers: int = 2):
        super().__init__()
        self.pred_len = pred_len
        self.top_k_periods = top_k_periods

        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        # Stacked MSRWKV blocks
        self.blocks = nn.ModuleList([
            MSRWKVBlock(d_model, tcn_channels, tcn_layers)
            for _ in range(n_layers)
        ])
        # Output projection
        self.ln_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)  # predict PV power (univariate output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim) — historical multivariate input
        Returns: (B, pred_len, 1) — predicted PV power
        """
        B, T, _ = x.shape

        # Detect periods from raw input
        with torch.no_grad():
            periods = detect_periods(x, self.top_k_periods)

        # Embed
        h = self.input_proj(x)  # (B, T, d_model)

        # Pass through MSRWKV blocks
        for block in self.blocks:
            h = block(h, periods)

        # Take last pred_len time steps for forecasting
        h = self.ln_out(h[:, -self.pred_len:, :])
        out = self.output_proj(h)  # (B, pred_len, 1)
        return out
