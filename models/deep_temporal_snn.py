"""
FB-SpikeSSVEPformer V6: Deep SNN + Temporal Coding
====================================================
Two major upgrades over V5:
1. Deep SNN: 4-layer PLIFEncoder (was 2) with residual connections
2. Temporal Coding: time-weighted spike aggregation instead of simple rate mean

Architecture:
    Raw EEG (B, 11, 250) -- 1s
        ↓
    FFT → Filter-Bank (3 subbands) → real+imag → (B, 11, 298)
        ↓
    3× Independent Subnet (ChComb + PLIFEncoder×4) → (B, 22, 298)
        ↓
    Flatten → 3× (B, 6556)
        ↓
    PLIFFeatureFusion (learnable tau, T=12 rate coding) → (B, 6556)
        ↓
    MlpHead-Temporal (6556 → 240 → LIF(T=12) → Temporal Weight → 40)

Temporal Coding:
    Instead of simple mean over T steps:  out = mean(spike_t for t=1..T)
    We use time-weighted aggregation:      out = sum(w_t * spike_t) / sum(w_t)
    where w_t = exp(-t/τ) or (T-t)/T, giving higher weight to early spikes.
    This makes the model sensitive to spike timing, not just spike count.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based import functional


class FreqFilterBank(nn.Module):
    """Frequency-domain Filter-Bank (same as V4)."""
    def __init__(self, fs=250, nfft=1000, n_subbands=3):
        super().__init__()
        self.fs = fs
        self.nfft = nfft
        self.n_subbands = n_subbands
        freqs = torch.fft.fftfreq(nfft, d=1.0/fs)
        band_ranges = [(8, 45), (16, 45), (24, 45)]
        masks = []
        for low, high in band_ranges:
            mask = ((freqs >= low) & (freqs <= high)) | ((freqs >= -high) & (freqs <= -low))
            masks.append(mask.float())
        self.register_buffer('masks', torch.stack(masks))
        self.subband_scale = nn.Parameter(torch.ones(n_subbands))
    
    def forward(self, x_fft):
        masked = []
        for s in range(self.n_subbands):
            mask = self.masks[s].view(1, 1, -1)
            x_m = x_fft * mask * self.subband_scale[s]
            masked.append(x_m)
        return masked


class ChComb(nn.Module):
    """Channel Combination (same as V4)."""
    def __init__(self, Chans=8, Samples=220, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(Chans // 2, Chans, 1, padding='same')
        self.ln = nn.LayerNorm(Samples)
        self.act = nn.GELU()
        self.do = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.do(self.act(self.ln(self.conv(x))))


class PLIFBlock(nn.Module):
    """Single PLIF block with residual connection.
    
    Structure: LayerNorm → Conv1d → GELU → Dropout (residual) →
               LayerNorm → MLP → PLIF rate coding → Dropout (residual)
    """
    def __init__(self, Chans=16, Samples=220, dropout=0.5, T_snn=12):
        super().__init__()
        self.channels = Chans
        self.T_snn = T_snn
        
        # Block 1: Conv + GELU
        self.ln1 = nn.LayerNorm(Samples)
        self.conv = nn.Conv1d(Chans, Chans, 31, padding='same')
        self.ln2 = nn.LayerNorm(Samples)
        self.act = nn.GELU()
        self.do = nn.Dropout(p=dropout)
        
        # Block 2: MLP + PLIF
        self.ln3 = nn.LayerNorm(Samples)
        self.proj = nn.Linear(Chans, Samples)
        self.do2 = nn.Dropout(p=dropout)
        
        # PLIF: learnable tau
        self.lif = neuron.ParametricLIFNode(
            init_tau=2.0,
            surrogate_function=surrogate.ATan(),
            v_threshold=1.0,
            v_reset=0.0
        )
    
    def forward(self, x):
        # Block 1 with residual
        shortcut = x
        x = self.conv(self.ln1(x))
        x = self.act(self.ln2(x))
        x = self.do(x) + shortcut
        
        # Block 2: MLP projection
        shortcut = x
        x = self.ln3(x)
        output_channels = []
        for i in range(self.channels):
            c = self.proj(x[:, :, i]).unsqueeze(1)
            output_channels.append(c)
        x = torch.cat(output_channels, 1)
        
        # PLIF rate coding
        x_snn = x.permute(2, 0, 1)  # (Samples, B, Chans)
        self.lif.reset()
        spike_out = []
        for t in range(min(self.T_snn, x_snn.shape[0])):
            spike = self.lif(x_snn[t])
            spike_out.append(spike)
        x_snn = torch.stack(spike_out).mean(dim=0)  # (B, Chans)
        x_snn = x_snn.unsqueeze(2).expand(-1, -1, x.shape[2])
        
        x = self.do2(x_snn) + shortcut
        return x


class DeepPLIFEncoder(nn.Module):
    """Deep PLIF Encoder: stack of N PLIFBlocks with residual between blocks."""
    def __init__(self, Chans=16, Samples=220, dropout=0.5, T_snn=12, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            PLIFBlock(Chans, Samples, dropout, T_snn)
            for _ in range(n_layers)
        ])
        # Additional inter-block residual projections (if needed)
        self.use_inter_residual = True
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Subnet(nn.Module):
    """Single Filter-Bank subnetwork with deep PLIF encoders."""
    def __init__(self, Chans, Samples, drop_rate, T_snn, n_encoders=4):
        super().__init__()
        self.chcomb = ChComb(Chans, Samples, drop_rate)
        self.encoders = DeepPLIFEncoder(Chans, Samples, drop_rate, T_snn, n_layers=n_encoders)
    
    def forward(self, x):
        x = self.chcomb(x)
        x = self.encoders(x)
        return x


class PLIFFeatureFusion(nn.Module):
    """Feature fusion with PLIF (learnable tau per neuron)."""
    def __init__(self, feat_dim, n_subbands=3, T_snn=12, dropout=0.3):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_subbands = n_subbands
        self.T_snn = T_snn
        
        self.ln = nn.LayerNorm(feat_dim)
        self.lif = neuron.ParametricLIFNode(
            init_tau=2.0,
            surrogate_function=surrogate.ATan(),
            v_threshold=0.5,
            v_reset=0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fusion_weights = nn.Parameter(torch.ones(n_subbands))
    
    def forward(self, x):
        x = self.ln(x)
        self.lif.reset()
        spike_trains = []
        for _ in range(self.T_snn):
            spike = self.lif(x)
            spike_trains.append(spike)
        x_rate = torch.stack(spike_trains).mean(dim=0)
        x_rate = self.dropout(x_rate)
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = (x_rate * weights.view(1, -1, 1)).sum(dim=1)
        return fused


class MlpHeadTemporal(nn.Module):
    """MLP Classification Head with Temporal Coding.
    
    Instead of simple rate mean over T steps, we use time-weighted aggregation:
        weight[t] = exp(-t / tau_temporal) or linear decay
    
    This gives higher importance to early spikes, making the model sensitive
    to spike timing, not just spike count.
    
    Structure: Linear(6556→240) → LIF(T steps) → Temporal Weighted Sum → 
               LayerNorm → GELU → Dropout → Linear(240→40)
    """
    def __init__(self, Chans, Samples, n_classes, drop_rate=0.5, T_snn=12, 
                 temporal_mode='exp', temporal_tau=3.0):
        super().__init__()
        self.T_snn = T_snn
        self.temporal_mode = temporal_mode
        self.temporal_tau = temporal_tau
        
        self.drop = nn.Dropout(drop_rate)
        self.linear1 = nn.Linear(Chans * Samples, 6 * n_classes)
        
        # LIF bottleneck
        self.lif = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0
        )
        
        self.norm = nn.LayerNorm(6 * n_classes)
        self.activation = nn.GELU()
        self.drop2 = nn.Dropout(drop_rate)
        self.linear2 = nn.Linear(6 * n_classes, n_classes)
        
        # Compute temporal weights (fixed, not learnable)
        self._compute_weights()
    
    def _compute_weights(self):
        """Precompute temporal weights for each time step."""
        if self.temporal_mode == 'exp':
            # w_t = exp(-t / tau)
            weights = torch.exp(-torch.arange(self.T_snn).float() / self.temporal_tau)
        elif self.temporal_mode == 'linear':
            # w_t = (T - t) / T, linear decay
            weights = 1.0 - torch.arange(self.T_snn).float() / self.T_snn
        elif self.temporal_mode == 'inverse':
            # w_t = 1 / (1 + t), heavy early emphasis
            weights = 1.0 / (1.0 + torch.arange(self.T_snn).float())
        else:  # uniform = rate coding
            weights = torch.ones(self.T_snn)
        
        # Normalize so weights sum to T_snn (comparable to rate coding)
        weights = weights / weights.sum() * self.T_snn
        self.register_buffer('temporal_weights', weights)
    
    def forward(self, x):
        x = self.drop(x)
        x = self.linear1(x)  # (B, 240)
        
        # LIF rate coding: record all T steps
        self.lif.reset()
        spikes = []
        for _ in range(self.T_snn):
            spikes.append(self.lif(x))
        spikes = torch.stack(spikes)  # (T, B, 240)
        
        # Temporal weighted aggregation: sum_t(w_t * spike_t) / sum(w_t)
        # temporal_weights: (T,)
        w = self.temporal_weights.view(-1, 1, 1)  # (T, 1, 1)
        x = (spikes * w).sum(dim=0) / self.temporal_weights.sum()  # (B, 240)
        
        x = self.norm(x)
        x = self.activation(x)
        x = self.drop2(x)
        x = self.linear2(x)  # (B, 40)
        return x
    
    def get_temporal_weights(self):
        """Return current temporal weights for visualization."""
        return self.temporal_weights.cpu().numpy()


class FBSpikeSSVEPformerV6(nn.Module):
    """V6: Deep SNN + Temporal Coding.
    
    Upgrades:
    - 4-layer PLIFEncoder (was 2) with deeper residual connections
    - Temporal-weighted MlpHead instead of simple rate mean
    """
    def __init__(self, Chans=11, n_classes=40, fs=250,
                 band=[8, 45], resolution=0.25, drop_rate=0.5,
                 n_subbands=3, T_snn=12, n_encoders=4,
                 temporal_mode='exp', temporal_tau=3.0):
        super().__init__()
        self.n_subbands = n_subbands
        self.T_snn = T_snn
        self.n_encoders = n_encoders
        
        self.fs = fs
        self.resolution = resolution
        self.nfft = round(fs / resolution)
        self.fft_start = int(round(band[0] / self.resolution))
        self.fft_end = int(round(band[1] / self.resolution)) + 1
        n_freq = self.fft_end - self.fft_start
        samples = n_freq * 2
        filters = 2 * Chans
        
        print(f"FBSpikeSSVEPformerV6 (1s): Chans={Chans}, n_freq={n_freq}, "
              f"samples={samples}, filters={filters}, subbands={n_subbands}")
        print(f"  Deep PLIF Encoder: {n_encoders} layers, T_snn={T_snn}")
        print(f"  Temporal Fusion: T_snn={T_snn}")
        print(f"  MlpHead-Temporal: mode={temporal_mode}, tau={temporal_tau}")
        
        self.filter_bank = FreqFilterBank(fs=fs, nfft=self.nfft, n_subbands=n_subbands)
        self.subnets = nn.ModuleList([
            Subnet(filters, samples, drop_rate, T_snn, n_encoders=n_encoders)
            for _ in range(n_subbands)
        ])
        
        feat_dim = filters * samples
        self.fusion = PLIFFeatureFusion(feat_dim, n_subbands, T_snn)
        self.head = MlpHeadTemporal(filters, samples, n_classes, drop_rate, T_snn,
                                    temporal_mode=temporal_mode, temporal_tau=temporal_tau)
        self.init_weights()
    
    def init_weights(self):
        for module in self.modules():
            if hasattr(module, 'weight'):
                cls_name = module.__class__.__name__
                if not ("BatchNorm" in cls_name or "LayerNorm" in cls_name):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        B = x.shape[0]
        x_fft = torch.fft.fft(x, n=self.nfft) / x.shape[-1]
        x_subbands = self.filter_bank(x_fft)
        
        features = []
        for s in range(self.n_subbands):
            x_s = x_subbands[s]
            real = x_s.real[:, :, self.fft_start:self.fft_end]
            imag = x_s.imag[:, :, self.fft_start:self.fft_end]
            x_input = torch.cat([real, imag], dim=-1)
            functional.reset_net(self.subnets[s])
            h = self.subnets[s](x_input)
            features.append(h.flatten(1))
        
        x_stack = torch.stack(features, dim=1)
        fused = self.fusion(x_stack)
        out = self.head(fused)
        return out


def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    dummy = torch.randn(2, 11, 250).to(device)
    
    print("\n=== FBSpikeSSVEPformerV6 (4-layer Deep SNN + Temporal Coding) ===")
    model = FBSpikeSSVEPformerV6(
        Chans=11, n_classes=40, fs=250,
        band=[8, 45], resolution=0.25, drop_rate=0.5,
        n_subbands=3, T_snn=12, n_encoders=4,
        temporal_mode='exp', temporal_tau=3.0
    ).to(device)
    functional.reset_net(model)
    out = model(dummy)
    print(f"Input: {dummy.shape}, Output: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"Temporal weights: {model.head.get_temporal_weights()}")
    print("\nModel test passed!")


if __name__ == '__main__':
    test_model()
