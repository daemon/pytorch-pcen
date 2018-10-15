from f2m import F2M
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False, last_state=None, first_state=True):
    frames = x.split(1, -2)
    m_frames = []
    if first_state:
        last_state = None
    for frame in frames:
        if last_state is None:
            last_state = frame
            m_frames.append(frame)
            continue
        if training:
            m_frame = ((1 - s) * last_state).add_(s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    if training:
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_, last_state


class StreamingPCENTransform(nn.Module):

    def __init__(self, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=False, 
            use_cuda_kernel=False, **stft_kwargs):
        super().__init__()
        self.use_cuda_kernel = use_cuda_kernel
        if trainable:
            self.s = nn.Parameter(torch.Tensor([s]))
            self.alpha = nn.Parameter(torch.Tensor([alpha]))
            self.delta = nn.Parameter(torch.Tensor([delta]))
            self.r = nn.Parameter(torch.Tensor([r]))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable
        self.stft_kwargs = stft_kwargs
        self.register_buffer("last_state", torch.zeros(stft_kwargs["n_mels"]))
        mel_keys = {"n_mels", "sr", "f_max", "f_min", "n_fft"}
        mel_keys = set(stft_kwargs.keys()).intersection(mel_keys)
        mel_kwargs = {k: stft_kwargs[k] for k in mel_keys}
        stft_keys = set(stft_kwargs.keys()) - mel_keys
        self.n_fft = stft_kwargs["n_fft"]
        self.stft_kwargs = {k: stft_kwargs[k] for k in stft_keys}
        self.f2m = F2M(**mel_kwargs)
        self.reset()

    def reset(self):
        self.first_state = True

    def forward(self, x):
        x = torch.stft(x, self.n_fft, **self.stft_kwargs).norm(dim=-1, p=2)
        x = self.f2m(x.permute(0, 2, 1))
        if self.use_cuda_kernel:
            x, ls = pcen_cuda_kernel(x, self.eps, self.s, self.alpha, self.delta, self.r, self.trainable, self.last_state, self.first_state)
        else:
            x, ls = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable, self.last_state, self.first_state)
        self.last_state = ls.detach()
        self.first_state = False
        return x


if __name__ == "__main__":
    import time
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    transform = StreamingPCENTransform(n_mels=40, n_fft=480, hop_length=160).cuda()
    x = torch.tensor(librosa.core.load("yes.wav", sr=16000)[0]).unsqueeze(0).cuda()
    n = 200

    # Non-streaming
    a = time.perf_counter()
    for _ in range(n):
        y = transform(x)
        transform.reset()
    b = time.perf_counter()
    print(f"{(b - a) / n * 1000:.2} ms per second of audio.")

    # Streaming in chunks of 1600
    x_chunks = x.split(1600, 1)
    a = time.perf_counter()
    for _ in range(n):
        y_chunks = list(map(transform, x_chunks))
        transform.reset()
    b = time.perf_counter()
    print(f"{(b - a) / n * 1000:.2} ms per second of audio.")

    librosa.display.specshow(y[0].cpu().numpy().T)
    plt.title("Non-streaming")
    plt.show()

    librosa.display.specshow(torch.cat(y_chunks, 1)[0].cpu().numpy().T)
    plt.title("Streaming")
    plt.show()