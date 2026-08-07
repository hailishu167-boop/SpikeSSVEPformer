"""
FB-SpikeSSVEPformer V4-PLIF: PLIF + MlpHead-LIF enhancement
============================================================
Based on V4 with two SNN upgrades:
1. PLIFNode (Parametric LIF): learnable time constant tau per neuron
2. MlpHead-LIF: LIF rate-coding in the classification head

Architecture:
    Raw EEG (B, 11, 250) -- 1s
        ↓
    FFT → Filter-Bank (3 subbands) → real+imag → (B, 11, 298)
        ↓
    3× Independent Subnet (ChComb + PLIFEncoder×2) → (B, 22, 298)
        ↓
    Flatten → 3× (B, 6556)
        ↓
    PLIFFeatureFusion (learnable tau, T=4 rate coding) → (B, 6556)
        ↓
    MlpHead-LIF (6556 → 240 → LIF → 40) → (B, 40)
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


class PLIFEncoder(nn.Module):
    """Encoder with PLIF (Parametric LIF) enhancement.
    
    PLIFNode learns tau independently per neuron, allowing different
    subnetworks/subbands to adapt their own temporal dynamics.
    """
    def __init__(self, Chans=16, Samples=220, dropout=0.5, T_snn=8):
        super().__init__()
        self.channels = Chans
        self.T_snn = T_snn
        self.ln1 = nn.LayerNorm(Samples)
        self.conv = nn.Conv1d(Chans, Chans, 31, padding='same')
        self.ln2 = nn.LayerNorm(Samples)
        self.act = nn.GELU()
        self.do = nn.Dropout(p=dropout)
        self.ln3 = nn.LayerNorm(Samples)
        self.proj = nn.Linear(Chans, Samples)
        self.do2 = nn.Dropout(p=dropout)
        
        # PLIF: learnable tau (init=2.0), each neuron adapts independently
        self.lif = neuron.ParametricLIFNode(
            init_tau=2.0,
            surrogate_function=surrogate.ATan(),
            v_threshold=1.0,
            v_reset=0.0
        )
    
    def forward(self, x):
        shortcut1 = x
        x = self.conv(self.ln1(x))
        x = self.act(self.ln2(x))
        x = self.do(x) + shortcut1
        
        shortcut2 = x
        x = self.ln3(x)
        output_channels = []
        for i in range(self.channels):
            c = self.proj(x[:, :, i]).unsqueeze(1)
            output_channels.append(c)
        x = torch.cat(output_channels, 1)
        
        # PLIF rate coding: T_snn steps
        x_snn = x.permute(2, 0, 1)  # (Samples, B, Chans)
        self.lif.reset()
        spike_out = []
        for t in range(min(self.T_snn, x_snn.shape[0])):
            spike = self.lif(x_snn[t])
            spike_out.append(spike)
        x_snn = torch.stack(spike_out).mean(dim=0)  # (B, Chans)
        x_snn = x_snn.unsqueeze(2).expand(-1, -1, x.shape[2])
        
        x = self.do2(x_snn) + shortcut2
        return x


class Subnet(nn.Module):
    """Single Filter-Bank subnetwork with PLIF encoders."""
    def __init__(self, Chans, Samples, drop_rate, T_snn):
        super().__init__()
        self.chcomb = ChComb(Chans, Samples, drop_rate)
        self.encoder1 = PLIFEncoder(Chans, Samples, drop_rate, T_snn)
        self.encoder2 = PLIFEncoder(Chans, Samples, drop_rate, T_snn)
    
    def forward(self, x):
        x = self.chcomb(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        return x


class PLIFFeatureFusion(nn.Module):
    """Feature fusion with PLIF (learnable tau per neuron)."""
    def __init__(self, feat_dim, n_subbands=3, T_snn=8, dropout=0.3):
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


class MlpHeadLIF(nn.Module):
    """MLP Classification Head with LIF rate-coding.
    
    Structure: Linear(6556→240) → LIF(T=4) → LayerNorm → GELU → Dropout → Linear(240→40)
    
    The LIF introduces threshold-based spiking nonlinearity between the two
    linear layers, acting as a 'spiking bottleneck' for the final decision.
    """
    def __init__(self, Chans, Samples, n_classes, drop_rate=0.5, T_snn=8):
        super().__init__()
        self.T_snn = T_snn
        self.drop = nn.Dropout(drop_rate)
        self.linear1 = nn.Linear(Chans * Samples, 6 * n_classes)
        
        # LIF bottleneck: learnable threshold-based activation
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
    
    def forward(self, x):
        x = self.drop(x)
        x = self.linear1(x)  # (B, 240)
        
        # LIF rate coding: T_snn steps on the hidden features
        self.lif.reset()
        spikes = []
        for _ in range(self.T_snn):
            spikes.append(self.lif(x))
        x = torch.stack(spikes).mean(dim=0)  # (B, 240)
        
        x = self.norm(x)
        x = self.activation(x)
        x = self.drop2(x)
        x = self.linear2(x)  # (B, 40)
        return x


class FBSpikeSSVEPformerV4PLIF(nn.Module):
    """V4-PLIF: FB-SSVEPformer + PLIF + MlpHead-LIF."""
    def __init__(self, Chans=11, n_classes=40, fs=250,
                 band=[8, 45], resolution=0.25, drop_rate=0.5,
                 n_subbands=3, T_snn=8):
        super().__init__()
        self.n_subbands = n_subbands
        self.T_snn = T_snn
        
        self.fs = fs
        self.resolution = resolution
        self.nfft = round(fs / resolution)
        self.fft_start = int(round(band[0] / self.resolution))
        self.fft_end = int(round(band[1] / self.resolution)) + 1
        n_freq = self.fft_end - self.fft_start
        samples = n_freq * 2
        filters = 2 * Chans
        
        print(f"FBSpikeSSVEPformerV4PLIF (1s): Chans={Chans}, n_freq={n_freq}, "
              f"samples={samples}, filters={filters}, subbands={n_subbands}")
        print(f"  PLIF Encoder: T_snn={T_snn}, PLIF Fusion: T_snn={T_snn}")
        print(f"  MlpHead-LIF: T_snn={T_snn}")
        
        self.filter_bank = FreqFilterBank(fs=fs, nfft=self.nfft, n_subbands=n_subbands)
        self.subnets = nn.ModuleList([
            Subnet(filters, samples, drop_rate, T_snn)
            for _ in range(n_subbands)
        ])
        
        feat_dim = filters * samples
        self.fusion = PLIFFeatureFusion(feat_dim, n_subbands, T_snn)
        self.head = MlpHeadLIF(filters, samples, n_classes, drop_rate, T_snn)
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


class FBSSVEPformerV4PLIFBaseline(nn.Module):
    """V4-PLIF ANN-only baseline (no LIF/PLIF anywhere, standard GELU)."""
    def __init__(self, Chans=11, n_classes=40, fs=250,
                 band=[8, 45], resolution=0.25, drop_rate=0.5,
                 n_subbands=3):
        super().__init__()
        self.n_subbands = n_subbands
        self.fs = fs
        self.resolution = resolution
        self.nfft = round(fs / resolution)
        self.fft_start = int(round(band[0] / self.resolution))
        self.fft_end = int(round(band[1] / self.resolution)) + 1
        n_freq = self.fft_end - self.fft_start
        samples = n_freq * 2
        filters = 2 * Chans
        
        self.filter_bank = FreqFilterBank(fs=fs, nfft=self.nfft, n_subbands=n_subbands)
        self.subnets = nn.ModuleList([
            nn.Sequential(
                ChComb(filters, samples, drop_rate),
                nn.Sequential(  # ANN Encoder (no LIF)
                    nn.LayerNorm(samples),
                    nn.Conv1d(filters, filters, 31, padding='same'),
                    nn.LayerNorm(samples),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                    nn.Identity(),  # no residual shortcut for simplicity
                ),
                nn.Sequential(
                    nn.LayerNorm(samples),
                    nn.Identity(),  # simplified MLP
                )
            ) for _ in range(n_subbands)
        ])
        
        feat_dim = filters * samples
        self.fusion_ln = nn.LayerNorm(feat_dim)
        self.fusion_weights = nn.Parameter(torch.ones(n_subbands))
        self.fusion_dropout = nn.Dropout(drop_rate)
        
        # Standard 2-layer MLP head (no LIF)
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 6 * n_classes),
            nn.LayerNorm(6 * n_classes),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(6 * n_classes, n_classes)
        )
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
        x_fft = torch.fft.fft(x, n=self.nfft) / x.shape[-1]
        x_subbands = self.filter_bank(x_fft)
        features = []
        for s in range(self.n_subbands):
            x_s = x_subbands[s]
            real = x_s.real[:, :, self.fft_start:self.fft_end]
            imag = x_s.imag[:, :, self.fft_start:self.fft_end]
            x_input = torch.cat([real, imag], dim=-1)
            h = self.subnets[s](x_input)
            features.append(h.flatten(1))
        x_stack = torch.stack(features, dim=1)
        x_stack = self.fusion_ln(x_stack)
        x_stack = self.fusion_dropout(x_stack)
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = (x_stack * weights.view(1, -1, 1)).sum(dim=1)
        out = self.head(fused)
        return out


def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    dummy = torch.randn(2, 11, 250).to(device)
    
    print("\n=== FBSpikeSSVEPformerV4PLIF (1s, PLIF + MlpHead-LIF) ===")
    model = FBSpikeSSVEPformerV4PLIF(
        Chans=11, n_classes=40, fs=250,
        band=[8, 45], resolution=0.25, drop_rate=0.5,
        n_subbands=3, T_snn=8
    ).to(device)
    functional.reset_net(model)
    out = model(dummy)
    print(f"Input: {dummy.shape}, Output: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    print("\n=== FBSSVEPformerV4PLIFBaseline (1s, ANN only) ===")
    model_base = FBSSVEPformerV4PLIFBaseline(
        Chans=11, n_classes=40, fs=250,
        band=[8, 45], resolution=0.25, drop_rate=0.5,
        n_subbands=3
    ).to(device)
    out_base = model_base(dummy)
    print(f"Input: {dummy.shape}, Output: {out_base.shape}")
    n_params_base = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    print(f"Parameters: {n_params_base:,}")
    
    print("\nModel test passed!")


if __name__ == '__main__':
    test_model()
