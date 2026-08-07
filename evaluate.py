"""
Evaluation + Demo script for FB-SpikeSSVEPformer V4-PLIF (1s LOSO)

Loads all 35 LOSO-trained models and evaluates each on its held-out subject.
Prints overall accuracy summary and optionally generates per-subject bar chart.

Usage:
    python evaluate.py --results_dir results_plif --model_type v4_plif
    python evaluate.py --results_dir results_plif --model_type v4_plif --plot
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from models.plif_ssvepformer import FBSpikeSSVEPformerV4PLIF, FBSSVEPformerV4PLIFBaseline, functional


def load_model_for_subject(subject_id, model_type, results_dir, device):
    """Load the best model trained WITHOUT this subject."""
    model_path = os.path.join(results_dir, f'S{subject_id}_{model_type}_best.pth')
    if not os.path.exists(model_path):
        return None
    
    if model_type == 'v4_plif_base':
        model = FBSSVEPformerV4PLIFBaseline(
            Chans=11, n_classes=40, fs=250,
            band=[8, 45], resolution=0.25, drop_rate=0.5,
            n_subbands=3
        ).to(device)
    else:
        model = FBSpikeSSVEPformerV4PLIF(
            Chans=11, n_classes=40, fs=250,
            band=[8, 45], resolution=0.25, drop_rate=0.5,
            n_subbands=3, T_snn=8
        ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_all_subjects(cache_dir, results_dir, model_type='v4_plif', n_subjects=35):
    """Evaluate all LOSO models on their respective held-out subjects."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Evaluating {model_type} on {n_subjects} subjects (1s data)...")
    
    all_accs = []
    per_subject_results = []
    
    for sub in range(1, n_subjects + 1):
        # Load model
        model = load_model_for_subject(sub, model_type, results_dir, device)
        if model is None:
            print(f"  S{sub}: Model not found, skipping")
            continue
        
        # Load test data for this subject
        try:
            X_test = np.load(os.path.join(cache_dir, f'S{sub}_data.npy')).astype(np.float32)
            y_test = np.load(os.path.join(cache_dir, f'S{sub}_labels.npy')).astype(np.int64)
        except Exception as e:
            print(f"  S{sub}: Data not found, skipping")
            continue
        
        # Evaluate
        X_t = torch.from_numpy(X_test).to(device)
        y_t = torch.from_numpy(y_test).to(device)
        
        correct = 0
        total = 0
        all_preds = []
        
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(y_test), batch_size):
                xb = X_t[i:i+batch_size]
                yb = y_t[i:i+batch_size]
                if model_type != 'v4_plif_base':
                    functional.reset_net(model)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += pred.eq(yb).sum().item()
                total += len(yb)
                all_preds.extend(pred.cpu().numpy())
        
        acc = 100.0 * correct / total
        all_accs.append(acc)
        per_subject_results.append({
            'subject': sub,
            'accuracy': acc,
            'predictions': np.array(all_preds),
            'targets': y_test
        })
        print(f"  S{sub:2d}: {acc:.2f}%")
    
    if len(all_accs) == 0:
        print("No models found! Please run training first.")
        return None
    
    # Summary
    print(f"\n{'='*60}")
    print(f"OVERALL EVALUATION SUMMARY ({model_type})")
    print(f"{'='*60}")
    print(f"Subjects evaluated: {len(all_accs)}/{n_subjects}")
    print(f"Mean Accuracy: {np.mean(all_accs):.2f} +/- {np.std(all_accs):.2f}%")
    print(f"Min Accuracy: {np.min(all_accs):.2f}%")
    print(f"Max Accuracy: {np.max(all_accs):.2f}%")
    print(f"Median Accuracy: {np.median(all_accs):.2f}%")
    
    return all_accs, per_subject_results


def plot_results(all_accs, per_subject_results, model_type, save_dir='results_plif'):
    """Generate per-subject accuracy bar chart and confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Per-subject accuracy bar chart
    plt.figure(figsize=(14, 5))
    subjects = [r['subject'] for r in per_subject_results]
    colors = ['green' if a >= np.mean(all_accs) else 'orange' for a in all_accs]
    plt.bar(subjects, all_accs, color=colors, edgecolor='black')
    plt.axhline(y=np.mean(all_accs), color='red', linestyle='--', label=f'Mean: {np.mean(all_accs):.2f}%')
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Per-Subject Accuracy ({model_type}, 1s window, LOSO)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_per_subject_accuracy.png'), dpi=150)
    plt.close()
    print(f"Saved: {model_type}_per_subject_accuracy.png")
    
    # 2. Distribution histogram
    plt.figure(figsize=(8, 4))
    plt.hist(all_accs, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=np.mean(all_accs), color='red', linestyle='--', label=f'Mean: {np.mean(all_accs):.2f}%')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Number of Subjects')
    plt.title(f'Accuracy Distribution ({model_type})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_accuracy_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: {model_type}_accuracy_distribution.png")
    
    # 3. Overall confusion matrix (all subjects combined)
    all_preds = np.concatenate([r['predictions'] for r in per_subject_results])
    all_targets = np.concatenate([r['targets'] for r in per_subject_results])
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(label='Count')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Confusion Matrix ({model_type}, All {len(all_accs)} Subjects Combined)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Saved: {model_type}_confusion_matrix.png")
    
    # 4. Save summary to JSON
    import json
    summary = {
        'model_type': model_type,
        'n_subjects': len(all_accs),
        'mean_accuracy': float(np.mean(all_accs)),
        'std_accuracy': float(np.std(all_accs)),
        'min_accuracy': float(np.min(all_accs)),
        'max_accuracy': float(np.max(all_accs)),
        'median_accuracy': float(np.median(all_accs)),
        'per_subject': {f'S{r["subject"]}': float(r['accuracy']) for r in per_subject_results}
    }
    with open(os.path.join(save_dir, f'{model_type}_eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {model_type}_eval_summary.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate V4-PLIF LOSO models')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--cache_dir', type=str, default='cache_1s')
    parser.add_argument('--results_dir', type=str, default='results_plif')
    parser.add_argument('--model_type', type=str, default='v4_plif', choices=['v4_plif', 'v4_plif_base'])
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    cache_dir = os.path.join(args.data_dir, args.cache_dir)
    results_dir = os.path.join(args.data_dir, args.results_dir)
    
    # Evaluate
    ret = evaluate_all_subjects(cache_dir, results_dir, args.model_type)
    if ret is None:
        return
    
    all_accs, per_subject_results = ret
    
    # Plot if requested
    if args.plot:
        plot_results(all_accs, per_subject_results, args.model_type, results_dir)
    
    print(f"\nTo generate plots, run: python evaluate.py --model_type {args.model_type} --plot")


if __name__ == '__main__':
    main()
