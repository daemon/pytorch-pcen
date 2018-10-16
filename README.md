# PyTorch-PCEN
Efficient PyTorch reimplementation of [per-channel energy normalization](https://arxiv.org/pdf/1607.05666.pdf) with Mel 
spectrogram features.

## Overview

Robustness to loudness differences in near- and far-field conditions is critical in high-quality speech recognition applications. 
Obviously, spectrogram energies differ significantly between, say, shouting at arms-length and whispering from a distance. This 
phenomenon can worsen model quality, since the model itself would need to be robust across a wide range of input. The log-compression step in the log-Mel transform partially compresses the dynamic range of audio; however, it ignores per-channel 
energy differences and is static by definition.

[Per-channel energy normalization](https://arxiv.org/pdf/1607.05666.pdf) is one such solution to the aforementioned problems. 
It provides a per-channel, trainable front-end in place of the log compression, greatly improving model robustness in keyword spotting systems -- all the while being resource-efficient and easy to implement.

## Installation and Usage
1. PyTorch and NumPy are required. LibROSA and matplotlib are required only for the example.
2. To install via pip, run `pip install git+https://github.com/daemon/pytorch-pcen`. Otherwise, clone this repository and run `python setup.py install`.
3. To run the example in the module, place a 16kHz WAV file named `yes.wav` in the current directory. Then, do `python -m pcen.pcen`.

The following is a self-contained example for using a streaming PCEN layer:
```python
import pcen
import torch

# 40-dimensional features, 30-millisecond window, 10-millisecond shift; trainable is false by default
transform = pcen.StreamingPCENTransform(n_mels=40, n_fft=480, hop_length=160, trainable=True)
audio = torch.empty(1, 16000).normal_(0, 0.1) # Gaussian noise

# 1600 is an arbitrary chunk size; This step is unnecessary but demonstrates the streaming nature
streaming_chunks = audio.split(1600, 1)
pcen_chunks = [transform(chunk) for chunk in streaming_chunks] # Transform each chunk
transform.reset() # Reset the persistent streaming state
pcen_ = torch.cat(pcen_chunks, 1)
```

## Citation
Wang, Yuxuan, Pascal Getreuer, Thad Hughes, Richard F. Lyon, and Rif A. Saurous. [Trainable frontend for robust and far-field keyword spotting](https://arxiv.org/pdf/1607.05666.pdf). In _Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on_, pp. 5670-5674. IEEE, 2017.
```tex
@inproceedings{wang2017trainable,
  title={Trainable frontend for robust and far-field keyword spotting},
  author={Wang, Yuxuan and Getreuer, Pascal and Hughes, Thad and Lyon, Richard F and Saurous, Rif A},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on},
  pages={5670--5674},
  year={2017},
  organization={IEEE}
}
```
