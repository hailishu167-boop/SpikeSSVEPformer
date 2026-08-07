"""
Preprocess and cache SSVEP dataset for fast loading.
Converts .mat files to preprocessed .npy files.
"""
import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from tqdm import tqdm

CHANNEL_NAMES_64 = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2',
    'FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4',
    'C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ',
    'O2','CB2'
]

OCCIPITAL_CHANNELS = ['O1','OZ','O2','PO3','POZ','PO4','PO7','PO8','P3','PZ','P4']

def get_ch_idx(ch_names, targets):
    idx=[]
    for t in targets:
        t=t.upper()
        if t in ch_names:
            idx.append(ch_names.index(t))
    return idx

def bandpass(data, fs=250, low=5, high=45):
    nyq=0.5*fs
    b,a=butter(4,[low/nyq,high/nyq],btype='band')
    return filtfilt(b,a,data,axis=1)

def zscore(data):
    m=np.mean(data,axis=1,keepdims=True)
    s=np.std(data,axis=1,keepdims=True)
    return (data-m)/(s+1e-7)

def preprocess_all(data_dir, cache_dir, time_window=5.0, fs=250, delay=0.14, max_subjects=35):
    os.makedirs(cache_dir, exist_ok=True)
    ch_idx = get_ch_idx(CHANNEL_NAMES_64, OCCIPITAL_CHANNELS)
    n_ch = len(ch_idx)
    n_pts = int(time_window * fs)
    pre_stim = int(0.5 * fs)
    delay_s = int(delay * fs)
    start0 = pre_stim + delay_s
    end0 = start0 + n_pts

    for sub in tqdm(range(1, max_subjects+1), desc="Preprocessing"):
        f = os.path.join(data_dir, f'S{sub}.mat')
        if not os.path.exists(f): f = os.path.join(data_dir, f'S{sub:02d}.mat')
        if not os.path.exists(f):
            print(f"Warning: {f} not found")
            continue

        mat = loadmat(f)
        eeg = mat['data']  # (64, 1500, 40, 6)
        start = start0
        end = end0
        if end > eeg.shape[1]:
            end = eeg.shape[1]
            start = end - n_pts
        data = eeg[ch_idx, start:end, :, :]
        for c in range(40):
            for b in range(6):
                data[:, :, c, b] = bandpass(data[:, :, c, b], fs=fs)
                data[:, :, c, b] = zscore(data[:, :, c, b])
        data = data.reshape(n_ch, n_pts, 40*6).transpose(2, 0, 1)
        labels = np.repeat(np.arange(40), 6)
        np.save(os.path.join(cache_dir, f'S{sub}_data.npy'), data.astype(np.float32))
        np.save(os.path.join(cache_dir, f'S{sub}_labels.npy'), labels.astype(np.int64))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_window', type=float, default=5.0)
    parser.add_argument('--data_dir', type=str, default='.')
    args = parser.parse_args()
    
    cache_dir = os.path.join(args.data_dir, f'cache_{int(args.time_window)}s')
    preprocess_all(args.data_dir, cache_dir, time_window=args.time_window, fs=250, delay=0.14, max_subjects=35)
    print(f"Cache saved to {cache_dir}")
    print(f"Files: {os.listdir(cache_dir)[:5]}")
