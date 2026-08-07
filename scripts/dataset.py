"""
SSVEP 40-class Dataset Loader and Preprocessing
BETA Dataset: 35 subjects, 64 channels, 250Hz, 40 classes, 6 blocks per subject
Data shape: [64, 1500, 40, 6] -> (channels, time_points, classes, blocks)
"""
import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset, DataLoader

# Channel names for 64-channel EEG (International 10-20 system)
# These are typical names for the Neuroscan 64-channel system
CHANNEL_NAMES_64 = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2'
]

# Occipital-parietal channels commonly used for SSVEP
OCCIPITAL_CHANNELS = ['O1', 'OZ', 'O2', 'PO3', 'POZ', 'PO4', 'PO7', 'PO8', 'P3', 'PZ', 'P4']

def get_channel_indices(channel_names, target_names):
    """Get indices of target channels from channel list."""
    indices = []
    for name in target_names:
        name_upper = name.upper()
        if name_upper in channel_names:
            indices.append(channel_names.index(name_upper))
        else:
            # Try matching with 'Z' vs 'z' variations
            alt_name = name_upper.replace('Z', 'z') if 'Z' in name_upper else name_upper.replace('z', 'Z')
            if alt_name in channel_names:
                indices.append(channel_names.index(alt_name))
    return indices


def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=5.0, highcut=45.0, fs=250.0, order=4):
    """
    Apply bandpass filter to EEG data.
    data: (channels, time_points) or (channels, time_points, ...)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered = filtfilt(b, a, data, axis=1)
    return filtered


def notch_filter(data, freq=50.0, fs=250.0, quality=30.0):
    """Apply notch filter to remove power line noise."""
    b, a = signal.iirnotch(freq, quality, fs)
    filtered = filtfilt(b, a, data, axis=1)
    return filtered


def standardize(data, axis=1, eps=1e-7):
    """Z-score standardization along time axis."""
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / (std + eps)


class SSVEPDataset(Dataset):
    """
    SSVEP Dataset for PyTorch.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing .mat files
    subject_ids : list
        List of subject IDs to load (e.g., [1, 2, 3])
    time_window : float
        Length of time window in seconds (e.g., 1.0 for 1s)
    fs : float
        Sampling frequency (default 250.0)
    use_channels : list or str
        List of channel names to use, or 'occipital' for default occipital channels,
        or 'all' for all 64 channels
    preprocess : bool
        Whether to apply bandpass filtering and standardization
    phase_delay : float
        Phase delay in seconds to account for visual response latency (default 0.14s)
    """
    def __init__(self, data_dir, subject_ids, time_window=1.0, fs=250.0,
                 use_channels='occipital', preprocess=True, phase_delay=0.14,
                 transform=None):
        self.data_dir = data_dir
        self.subject_ids = subject_ids
        self.time_window = time_window
        self.fs = fs
        self.sample_points = int(time_window * fs)
        self.preprocess = preprocess
        self.phase_delay = phase_delay
        self.transform = transform
        
        # Channel selection
        if use_channels == 'occipital':
            self.channel_indices = get_channel_indices(CHANNEL_NAMES_64, OCCIPITAL_CHANNELS)
        elif use_channels == 'all':
            self.channel_indices = list(range(64))
        else:
            self.channel_indices = get_channel_indices(CHANNEL_NAMES_64, use_channels)
        
        self.n_channels = len(self.channel_indices)
        
        # Load all data
        self.data, self.labels = self._load_data()
        
    def _load_data(self):
        """Load and preprocess all data from specified subjects."""
        all_data = []
        all_labels = []
        
        for sub_id in self.subject_ids:
            mat_file = os.path.join(self.data_dir, f'S{sub_id}.mat')
            if not os.path.exists(mat_file):
                # Try alternative naming (S01, S02, etc.)
                mat_file_alt = os.path.join(self.data_dir, f'S{sub_id:02d}.mat')
                if os.path.exists(mat_file_alt):
                    mat_file = mat_file_alt
                else:
                    print(f"Warning: File not found for subject {sub_id}: {mat_file}")
                    continue
            
            mat_data = sio.loadmat(mat_file)
            eeg_data = mat_data['data']  # (64, 1500, 40, 6)
            
            # Extract selected channels and time window
            # The data includes 500ms pre-stimulus (first 125 points at 250Hz)
            # and 5000ms post-stimulus (1250 points), total 1500 points
            pre_stimulus = int(0.5 * self.fs)  # 125 points
            
            # Apply phase delay (visual latency)
            delay_samples = int(self.phase_delay * self.fs)
            start_idx = pre_stimulus + delay_samples
            end_idx = start_idx + self.sample_points
            
            # Ensure we don't exceed data length
            if end_idx > eeg_data.shape[1]:
                end_idx = eeg_data.shape[1]
                start_idx = end_idx - self.sample_points
            
            # Extract data: (selected_channels, time_window, 40, 6)
            sub_data = eeg_data[self.channel_indices, start_idx:end_idx, :, :]  # (C, T, 40, 6)
            
            # Reshape to (C, T, 40*6) = (C, T, 240) trials
            n_classes = sub_data.shape[2]
            n_blocks = sub_data.shape[3]
            n_trials = n_classes * n_blocks
            
            sub_data = sub_data.reshape(self.n_channels, self.sample_points, n_trials)
            sub_data = sub_data.transpose(2, 0, 1)  # (n_trials, C, T)
            
            # Create labels: each class repeats 6 times (for 6 blocks)
            labels = np.repeat(np.arange(n_classes), n_blocks)  # (240,)
            
            # Preprocessing: bandpass filter + standardization per trial
            if self.preprocess:
                for i in range(n_trials):
                    trial = sub_data[i]  # (C, T)
                    # Bandpass filter 5-45Hz
                    trial = bandpass_filter(trial, lowcut=5.0, highcut=45.0, fs=self.fs)
                    # Standardize
                    trial = standardize(trial, axis=1)
                    sub_data[i] = trial
            
            all_data.append(sub_data)
            all_labels.append(labels)
        
        if len(all_data) == 0:
            raise ValueError("No data loaded! Check data directory and subject IDs.")
        
        data = np.concatenate(all_data, axis=0)  # (N_total, C, T)
        labels = np.concatenate(all_labels, axis=0)  # (N_total,)
        
        return data.astype(np.float32), labels.astype(np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]  # (C, T)
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def create_dataloaders(data_dir, subject_ids, train_blocks=None, test_blocks=None,
                       time_window=1.0, fs=250.0, use_channels='occipital',
                       batch_size=64, num_workers=0, preprocess=True):
    """
    Create train and test dataloaders for a specific subject with block-wise split.
    
    If train_blocks and test_blocks are None, uses all data (for cross-validation use).
    """
    if train_blocks is not None or test_blocks is not None:
        # This requires loading raw data and splitting by blocks
        # For simplicity, we'll load all data and split afterward
        dataset = SSVEPDataset(data_dir, subject_ids, time_window, fs, 
                               use_channels, preprocess)
        
        # Need to split by blocks - this is more complex
        # For now, let's create a simpler interface
        raise NotImplementedError("Block-wise splitting requires loading raw data.")
    else:
        dataset = SSVEPDataset(data_dir, subject_ids, time_window, fs, 
                               use_channels, preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=num_workers,
                               pin_memory=True)
        return dataloader


def load_subject_data(data_dir, subject_id, time_window=1.0, fs=250.0,
                     use_channels='occipital', preprocess=True, phase_delay=0.14):
    """
    Load data for a single subject and return train/test splits.
    Uses Leave-One-Block-Out (LOBO) cross-validation: 5 blocks for train, 1 block for test.
    
    Returns:
    --------
    train_data, train_labels, test_data, test_labels : np.ndarray
        Shape: (N_train, C, T), (N_train,), (N_test, C, T), (N_test,)
    """
    mat_file = os.path.join(data_dir, f'S{subject_id}.mat')
    if not os.path.exists(mat_file):
        mat_file = os.path.join(data_dir, f'S{subject_id:02d}.mat')
    
    mat_data = sio.loadmat(mat_file)
    eeg_data = mat_data['data']  # (64, 1500, 40, 6)
    
    # Channel selection
    if use_channels == 'occipital':
        ch_indices = get_channel_indices(CHANNEL_NAMES_64, OCCIPITAL_CHANNELS)
    elif use_channels == 'all':
        ch_indices = list(range(64))
    else:
        ch_indices = get_channel_indices(CHANNEL_NAMES_64, use_channels)
    
    n_channels = len(ch_indices)
    sample_points = int(time_window * fs)
    pre_stimulus = int(0.5 * fs)
    delay_samples = int(phase_delay * fs)
    start_idx = pre_stimulus + delay_samples
    end_idx = start_idx + sample_points
    
    if end_idx > eeg_data.shape[1]:
        end_idx = eeg_data.shape[1]
        start_idx = end_idx - sample_points
    
    # Extract: (channels, time, classes, blocks)
    data = eeg_data[ch_indices, start_idx:end_idx, :, :]  # (C, T, 40, 6)
    
    # Preprocess each trial
    if preprocess:
        for c in range(40):
            for b in range(6):
                trial = data[:, :, c, b]
                trial = bandpass_filter(trial, lowcut=5.0, highcut=45.0, fs=fs)
                trial = standardize(trial, axis=1)
                data[:, :, c, b] = trial
    
    # Reshape to (C, T, 240) where 240 = 40*6
    data = data.reshape(n_channels, sample_points, 40 * 6)
    data = data.transpose(2, 0, 1)  # (240, C, T)
    labels = np.repeat(np.arange(40), 6)  # (240,)
    
    # LOBO split: 5 blocks train, 1 block test
    # We'll create 6 different splits
    splits = []
    for test_block in range(6):
        train_blocks = [b for b in range(6) if b != test_block]
        
        # Get indices for train and test
        train_indices = []
        test_indices = []
        for cls in range(40):
            for b in range(6):
                idx = cls * 6 + b
                if b in train_blocks:
                    train_indices.append(idx)
                else:
                    test_indices.append(idx)
        
        train_data = data[train_indices].astype(np.float32)
        train_labels = labels[train_indices].astype(np.int64)
        test_data = data[test_indices].astype(np.float32)
        test_labels = labels[test_indices].astype(np.int64)
        
        splits.append((train_data, train_labels, test_data, test_labels))
    
    return splits


def create_loso_datasets(data_dir, test_subject_id, time_window=1.0, fs=250.0,
                        use_channels='occipital', preprocess=True, phase_delay=0.14):
    """
    Create leave-one-subject-out datasets.
    All subjects except test_subject are used for training.
    """
    train_subjects = [i for i in range(1, 36) if i != test_subject_id]
    
    train_dataset = SSVEPDataset(data_dir, train_subjects, time_window, fs,
                                use_channels, preprocess, phase_delay)
    test_dataset = SSVEPDataset(data_dir, [test_subject_id], time_window, fs,
                              use_channels, preprocess, phase_delay)
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    # Test dataset loading
    data_dir = '.'
    
    print("Testing dataset loading...")
    
    # Test single subject
    dataset = SSVEPDataset(data_dir, [1], time_window=1.0, use_channels='occipital')
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample shape: {x.shape}, Label: {y}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    x_batch, y_batch = next(iter(loader))
    print(f"Batch shape: {x_batch.shape}, Labels: {y_batch.shape}")
    
    print("Dataset test passed!")
