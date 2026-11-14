"""PANNs CNN14 as a frozen feature extractor for heart sounds (Round 86, the owner's choice of 2026-11-14).

Kong, Cao, Iqbal, Wang, Wang and Plumbley, "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern
Recognition", IEEE/ACM TASLP 28 (2020); weights `Cnn14_mAP=0.431.pth` from Zenodo, doi:10.5281/zenodo.3987831, under
CC BY 4.0, trained on AudioSet (no heart-sound corpus). The forward pass is rebuilt from the checkpoint itself: the STFT
is the stored pair of strided convolutions (hann window, 1,024 points, hop 320, reflect padding), the mel matrix is the
stored one (64 bands, 50 to 14,000 Hz at 32 kHz), then 10 log10 power, a batch norm over the bands, six blocks of two
3x3 convolutions with average pooling, the mean over frequency, max plus mean over time, and fc1 with ReLU: the 2,048-d
embedding. The AudioSet classifier is not used. Weights are read with torch.load(weights_only=True) and never shipped.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

RATE = 32000


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor, pool: tuple[int, int]) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool)


class Cnn14Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_real = nn.Conv1d(1, 513, 1024, stride=320, bias=False)
        self.conv_imag = nn.Conv1d(1, 513, 1024, stride=320, bias=False)
        self.melW = nn.Parameter(torch.rand(513, 64) / 513)     # the checkpoint replaces it
        self.bn0 = nn.BatchNorm2d(64)
        widths = (1, 64, 128, 256, 512, 1024, 2048)
        self.blocks = nn.ModuleList(_Block(widths[k], widths[k + 1]) for k in range(6))
        self.fc1 = nn.Linear(2048, 2048)

    @classmethod
    def from_checkpoint(cls, path) -> Cnn14Embedding:
        sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
        m = cls()
        mine = {"conv_real.weight": sd["spectrogram_extractor.stft.conv_real.weight"],
                "conv_imag.weight": sd["spectrogram_extractor.stft.conv_imag.weight"], "melW": sd["logmel_extractor.melW"]}
        for k, v in sd.items():
            if k.startswith("bn0.") or k.startswith("fc1."):
                mine[k] = v
            elif k.startswith("conv_block"):
                n = int(k[len("conv_block")]) - 1
                mine[f"blocks.{n}." + k.split(".", 1)[1]] = v
        missing, unexpected = m.load_state_dict(mine, strict=True), None
        del missing, unexpected
        return m.eval()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(batch, samples at 32 kHz) -> (batch, 2048)."""
        x = F.pad(wav[:, None, :], (512, 512), mode="reflect")
        spec = (self.conv_real(x) ** 2 + self.conv_imag(x) ** 2).transpose(1, 2)[:, None]      # (B, 1, T, 513)
        x = 10.0 * torch.log10(torch.clamp(spec @ self.melW, min=1e-10))                       # (B, 1, T, 64)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for k, blk in enumerate(self.blocks):
            x = blk(x, (2, 2) if k < 5 else (1, 1))
        x = x.mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        return F.relu(self.fc1(x))


def upsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Band-limited interpolation by zero-padding the spectrum: (batch, n) -> (batch, n * factor)."""
    n = x.shape[-1]
    spec = torch.fft.rfft(x.float(), dim=-1)
    out = torch.zeros(*spec.shape[:-1], n * factor // 2 + 1, dtype=spec.dtype, device=x.device)
    out[..., : spec.shape[-1]] = spec
    return torch.fft.irfft(out, n=n * factor, dim=-1) * factor
