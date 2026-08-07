"""
Fixed-split training V6: S1-S34 train, S35 test
Key features: Deep SNN (4-layer PLIF) + Temporal Coding (time-weighted spike aggregation)

Usage:
    python train_v6_deep_temporal.py --n_encoders 4 --temporal_mode exp --temporal_tau 3.0
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import json
import matplotlib.pyplot as plt
from math import cos, pi

from models.model_v6_deep_temporal import FBSpikeSSVEPformerV6, functional


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_itr(accuracy, n_classes=40, time_window=1.0, gaze_shift=0.5):
    if accuracy <= 0 or accuracy >= 1.0:
        return 0.0
    T = time_window + gaze_shift
    itr = (60.0 / T) * (np.log2(n_classes) +
                        accuracy * np.log2(accuracy) +
                        (1 - accuracy) * np.log2((1 - accuracy) / (n_classes - 1)))
    return itr


def augment_eeg(x, noise_std=0.5, shift_max=20, scale_range=(0.9, 1.1)):
    """Augment a single EEG trial (Chans, Samples)."""
    x_aug = x.copy()
    
    # Amplitude scaling
    scale = np.random.uniform(*scale_range)
    x_aug = x_aug * scale
    
    # Circular time shift
    shift = np.random.randint(-shift_max, shift_max + 1)
    if shift != 0:
        x_aug = np.roll(x_aug, shift, axis=1)
    
    # Gaussian noise
    noise = np.random.normal(0, noise_std, x_aug.shape)
    x_aug = x_aug + noise
    
    # High-frequency noise shaping
    if np.random.rand() > 0.5:
        x_aug[:, 1:] = x_aug[:, 1:] + 0.1 * np.diff(x_aug, axis=1)
    
    return x_aug.astype(np.float32)


def augment_weak_classes(X, y, weak_classes, factor=3):
    """For each weak-class sample, generate `factor` augmented copies."""
    X_list = [X]
    y_list = [y]
    
    for c in weak_classes:
        mask = y == c
        if mask.sum() == 0:
            continue
        X_weak = X[mask]
        
        for _ in range(factor):
            X_aug = np.array([augment_eeg(x) for x in X_weak])
            y_aug = np.full(len(X_aug), c, dtype=y.dtype)
            X_list.append(X_aug)
            y_list.append(y_aug)
    
    X_new = np.concatenate(X_list, axis=0)
    y_new = np.concatenate(y_list, axis=0)
    return X_new, y_new


def train_epoch(model, dataloader, optimizer, criterion, device, use_snn=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        if use_snn:
            functional.reset_net(model)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100.0 * correct / total


def evaluate(model, dataloader, criterion, device, use_snn=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            if use_snn:
                functional.reset_net(model)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return total_loss / total, 100.0 * correct / total, np.array(all_preds), np.array(all_targets)


def run_fixed_split(data_dir, cache_dir, epochs=300, lr=0.001,
                    batch_size=256, save_dir='results_v6_deep_temporal', patience=80, T_snn=12,
                    warmup_epochs=10, drop_rate=0.5,
                    augment_factor=3, augment_classes=None,
                    n_encoders=4, temporal_mode='exp', temporal_tau=3.0):
    """Fixed split: S1-S34 train, S35 test. V6 with deep SNN + temporal coding."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: dropout={drop_rate}, Adam, T_snn={T_snn}, warmup={warmup_epochs}, patience={patience}")
    print(f"Deep SNN: {n_encoders} layers")
    print(f"Temporal Coding: mode={temporal_mode}, tau={temporal_tau}")
    print(f"Data Augmentation: factor={augment_factor}, classes={augment_classes}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load training data: S1-S34
    train_data_list = []
    train_labels_list = []
    for s in range(1, 35):
        try:
            d = np.load(os.path.join(cache_dir, f'S{s}_data.npy'))
            l = np.load(os.path.join(cache_dir, f'S{s}_labels.npy'))
            train_data_list.append(d)
            train_labels_list.append(l)
        except Exception as e:
            print(f"  Warning: Could not load S{s}: {e}")
            continue
    
    X_train = np.concatenate(train_data_list, axis=0)
    y_train = np.concatenate(train_labels_list, axis=0)
    X_test = np.load(os.path.join(cache_dir, 'S35_data.npy'))
    y_test = np.load(os.path.join(cache_dir, 'S35_labels.npy'))
    
    print(f"Train (S1-S34) before aug: {X_train.shape}")
    
    # Default weak classes: A(0), F(5), 4(29)
    if augment_classes is None:
        augment_classes = [0, 5, 29]
    
    # Augment weak classes
    X_train, y_train = augment_weak_classes(X_train, y_train, augment_classes, factor=augment_factor)
    print(f"Train (S1-S34) after aug:  {X_train.shape}")
    print(f"Test  (S35):                 {X_test.shape}")
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    model = FBSpikeSSVEPformerV6(
        Chans=11, n_classes=40, fs=250,
        band=[8, 45], resolution=0.25, drop_rate=drop_rate,
        n_subbands=3, T_snn=T_snn, n_encoders=n_encoders,
        temporal_mode=temporal_mode, temporal_tau=temporal_tau
    ).to(device)
    use_snn = True
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    
    # Warmup + CosineAnnealing scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.01 + 0.99 * (0.5 * (1 + cos(pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': [], 'lr': []}
    
    for ep in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, use_snn)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device, use_snn)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['lr'].append(current_lr)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = ep
            patience_counter = 0
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(save_dir, 'v6_best_model.pth'))
            torch.save(best_state, os.path.join(save_dir, 'v6_S35_model.pth'))
        else:
            patience_counter += 1
        
        if (ep + 1) % 20 == 0 or ep == 0 or ep == best_epoch:
            print(f"  Epoch {ep + 1:3d}: Train={train_acc:.2f}%, Test={test_acc:.2f}%, "
                  f"Best={best_acc:.2f}% (ep {best_epoch + 1}), LR={current_lr:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {ep + 1}")
            break
    
    # Final evaluation with best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'v6_best_model.pth'), map_location=device))
    _, final_acc, preds, targets = evaluate(model, test_loader, criterion, device, use_snn)
    
    itr = compute_itr(final_acc / 100.0, time_window=1.0)
    
    # Final per-class analysis
    final_class_acc = {}
    for c in range(40):
        mask = targets == c
        total = mask.sum()
        if total > 0:
            final_class_acc[c] = (preds[mask] == c).sum() / total
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS V6 (Deep SNN + Temporal Coding)")
    print(f"{'='*60}")
    print(f"Best Test Accuracy: {best_acc:.2f}% (epoch {best_epoch + 1})")
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"ITR: {itr:.2f} bits/min")
    print(f"Model saved: {save_dir}/v6_S35_model.pth")
    
    # Save temporal weights for reference
    temporal_weights = model.head.get_temporal_weights().tolist()
    print(f"Temporal weights: {[f'{w:.3f}' for w in temporal_weights]}")
    
    # Save results
    results = {
        'model_type': 'v6_deep_temporal',
        'train_subjects': 'S1-S34',
        'test_subject': 'S35',
        'best_accuracy': float(best_acc),
        'final_accuracy': float(final_acc),
        'best_epoch': int(best_epoch + 1),
        'itr': float(itr),
        'n_params': int(n_params),
        'dropout': drop_rate,
        'optimizer': 'Adam',
        'T_snn': T_snn,
        'n_encoders': n_encoders,
        'warmup_epochs': warmup_epochs,
        'temporal_mode': temporal_mode,
        'temporal_tau': temporal_tau,
        'temporal_weights': temporal_weights,
        'augment_factor': augment_factor,
        'augment_classes': augment_classes,
        'final_class_acc': {int(k): float(v) for k, v in final_class_acc.items()},
        'history': history
    }
    with open(os.path.join(save_dir, 'v6_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot curves
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best (ep {best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best (ep {best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate')
    plt.axvline(x=warmup_epochs, color='green', linestyle='--', label='Warmup end')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'v6_curves.png'), dpi=150)
    plt.close()
    print(f"Curves saved: {save_dir}/v6_curves.png")
    
    return best_acc, final_acc, itr


def main():
    parser = argparse.ArgumentParser(description='Train V6: S1-S34 train, S35 test (Deep SNN + Temporal Coding)')
    parser.add_argument('--data_dir', type=str, default=r'D:\\学习资料\\BCI\\40分类')
    parser.add_argument('--cache_dir', type=str, default='cache_1s')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='results_v6_deep_temporal')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--T_snn', type=int, default=12)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--n_encoders', type=int, default=4,
                        help='Number of PLIFEncoder layers per subband (default 4, was 2 in V5)')
    parser.add_argument('--temporal_mode', type=str, default='exp',
                        choices=['exp', 'linear', 'inverse', 'uniform'],
                        help='Temporal weighting mode: exp, linear, inverse, uniform')
    parser.add_argument('--temporal_tau', type=float, default=3.0,
                        help='Temporal decay constant for exp mode (default 3.0)')
    parser.add_argument('--augment_factor', type=int, default=3,
                        help='Number of augmented copies per weak-class sample')
    parser.add_argument('--augment_classes', type=str, default='0,5,29',
                        help='Comma-separated class IDs to augment')
    args = parser.parse_args()
    
    set_seed(args.seed)
    cache_dir = os.path.join(args.data_dir, args.cache_dir)
    
    augment_classes = [int(c.strip()) for c in args.augment_classes.split(',')]
    
    run_fixed_split(args.data_dir, cache_dir,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    save_dir=args.save_dir, patience=args.patience, T_snn=args.T_snn,
                    warmup_epochs=args.warmup_epochs, drop_rate=args.drop_rate,
                    augment_factor=args.augment_factor, augment_classes=augment_classes,
                    n_encoders=args.n_encoders, temporal_mode=args.temporal_mode,
                    temporal_tau=args.temporal_tau)


if __name__ == '__main__':
    main()
