"""
BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
import torch
import torch.nn as nn


class F2M(nn.Module):
    """This turns a normal STFT into a MEL Frequency STFT, using a conversion
       matrix.  This uses triangular filter banks.
    Args:
        n_mels (int): number of MEL bins
        sr (int): sample rate of audio signal
        f_max (float, optional): maximum frequency. default: sr // 2
        f_min (float): minimum frequency. default: 0
    """
    def __init__(self, n_mels=40, sr=16000, f_max=None, f_min=0., n_fft=40, onesided=True):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        if onesided:
            self.n_fft = self.n_fft // 2 + 1
        self._init_buffers()

    def _init_buffers(self):
        m_min = 0. if self.f_min == 0 else 2595 * np.log10(1. + (self.f_min / 700))
        m_max = 2595 * np.log10(1. + (self.f_max / 700))

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = (700 * (10**(m_pts / 2595) - 1))

        bins = torch.floor(((self.n_fft - 1) * 2) * f_pts / self.sr).long()

        fb = torch.zeros(self.n_fft, self.n_mels)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            if f_m != f_m_plus:
                fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
        self.register_buffer("fb", fb)

    def forward(self, spec_f):
        spec_m = torch.matmul(spec_f, self.fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m