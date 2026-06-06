import torch
import numpy as np

def bandpass_filter(bx, cutoff_freq, lowpass=True):
    assert cutoff_freq >= 0 and cutoff_freq <= 1, "cutoff must be in [0, 1]"
    fft = torch.fft.fftshift(torch.fft.fft2(bx))

    if not lowpass:
        cutoff_freq = 1 - cutoff_freq
    
    h, w = fft.shape[-2:]  # height and width
    cy, cx = h // 2, w // 2  # center y, center x
    ry, rx = int(cutoff_freq * cy), int(cutoff_freq * cx)
    
    if lowpass:
        mask = torch.zeros_like(fft)
        mask[:, cy-ry:cy+ry, cx-rx:cx+rx] = 1
    else:
        mask = torch.ones_like(fft)
        mask[:, cy-ry:cy+ry, cx-rx:cx+rx] = 0


    fft = torch.fft.ifft2(torch.fft.ifftshift(fft * mask)).real.clip(0, 1)
    return fft
