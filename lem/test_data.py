import numpy as np
import torch
import torch.nn as nn
import os

def fake_eeg(duration=1.0, sfreq=500.):
    times = np.arange(0, duration, 1.0 / sfreq)
    n_samples = times.size

    # Typical EEG frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta':  (12, 30)
    }

    signal = np.zeros(n_samples)
    # Add one sinusoid per band
    for fmin, fmax in bands.values():
        freq      = np.random.uniform(fmin, fmax)
        phase     = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(5, 20)  # arbitrary amplitude
        signal   += amplitude * np.sin(2 * np.pi * freq * times + phase)
    # Add broadband Gaussian noise
    noise = np.random.normal(loc=0, scale=5, size=n_samples)
    data = signal + noise

    return data

class FakeEEGDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1_000, duration=0.100, num_electrodes=32, mask_prob=0.1, mask_len=100,
            sfreq=500.,mmap_path='test_data.dat',create=True):
        self.num_samples = num_samples
        self.duration = duration
        self.sfreq = sfreq
        self.num_electrodes = num_electrodes
        self.mask_prob = mask_prob
        self.mask_len = mask_len

        data_len = fake_eeg(duration=self.duration, sfreq=self.sfreq).shape[0] * self.num_electrodes
        
        if create:
            mm = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(num_samples, data_len))
            for i in range(num_samples):
                mm[i] = np.concatenate([fake_eeg(duration=self.duration, sfreq=self.sfreq) for _ in range(self.num_electrodes)])
            mm.flush()
            self.data = mm
        else:
            self.data = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(num_samples, data_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        arr = self.data[idx]
        tensor = torch.from_numpy(arr)
        return tensor, tensor
                    