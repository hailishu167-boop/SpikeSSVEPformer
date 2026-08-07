"""
分析 V2 模型在 S35 上每个类别的单独准确率，找出薄弱类别。
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_v4_plif import FBSpikeSSVEPformerV4PLIF, functional

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    cache_dir = 'cache_1s'
    X = np.load(os.path.join(cache_dir, 'S35_data.npy')).astype(np.float32)
    y = np.load(os.path.join(cache_dir, 'S35_labels.npy')).astype(np.int64)
    return X, y

def load_model():
    model = FBSpikeSSVEPformerV4PLIF(
        Chans=11, n_classes=40, fs=250,
        band=[8, 45], resolution=0.25, drop_rate=0.5,
        n_subbands=3, T_snn=12
    ).to(DEVICE)
    model.load_state_dict(torch.load('results_v4_plif_fixed_v2/v4_plif_S35_model.pth', map_location=DEVICE))
    model.eval()
    return model

def analyze():
    X, y = load_data()
    model = load_model()
    
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            functional.reset_net(model)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Per-class accuracy
    class_acc = {}
    class_counts = {}
    for c in range(40):
        mask = all_labels == c
        total = mask.sum()
        if total > 0:
            correct = (all_preds[mask] == c).sum()
            class_acc[c] = correct / total
            class_counts[c] = total
        else:
            class_acc[c] = 0.0
            class_counts[c] = 0
    
    avg_acc = np.mean(list(class_acc.values()))
    
    CLASS_LABELS = [
        'A','B','C','D','E','F','G','H','I','J',
        'K','L','M','N','O','P','Q','R','S','T',
        'U','V','W','X','Y','Z','1','2','3','4',
        '5','6','7','8','9','0','.',',','!','?'
    ]
    
    print(f"\n{'='*60}")
    print(f"S35 Per-Class Accuracy Analysis (Avg: {avg_acc*100:.1f}%)")
    print(f"{'='*60}")
    
    # Sort by accuracy ascending
    sorted_classes = sorted(class_acc.items(), key=lambda x: x[1])
    
    print(f"\n{'Rank':>4} {'Class':>5} {'Char':>4} {'Count':>6} {'Acc':>8} {'Status':>10}")
    print("-" * 45)
    
    weak_classes = []
    for rank, (c, acc) in enumerate(sorted_classes):
        char = CLASS_LABELS[c]
        count = class_counts[c]
        status = "WEAK" if acc < avg_acc else "OK"
        if acc < avg_acc:
            weak_classes.append(c)
        print(f"{rank+1:>4} {c:>5} {char:>4} {count:>6} {acc*100:>7.1f}% {status:>10}")
    
    # Compute class weights
    weights = np.ones(40, dtype=np.float32)
    for c in range(40):
        if class_acc[c] > 0 and class_acc[c] < avg_acc:
            # Weight = avg_acc / class_acc, higher for weak classes
            weights[c] = min(avg_acc / class_acc[c], 5.0)  # cap at 5x
    
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    
    print(f"\n{'='*60}")
    print(f"Suggested Class Weights (capped at 5x, normalized mean=1)")
    print(f"{'='*60}")
    print(f"{'Class':>5} {'Char':>4} {'Weight':>8}")
    print("-" * 25)
    for c in range(40):
        print(f"{c:>5} {CLASS_LABELS[c]:>4} {weights[c]:>7.2f}")
    
    print(f"\nWeak classes: {len(weak_classes)} / 40")
    print(f"Weak class chars: {''.join([CLASS_LABELS[c] for c in weak_classes])}")
    
    # Save
    os.makedirs('results_v4_plif_fixed_v2', exist_ok=True)
    with open('results_v4_plif_fixed_v2/class_weights.json', 'w') as f:
        json.dump({
            'avg_acc': float(avg_acc),
            'class_acc': {int(k): float(v) for k, v in class_acc.items()},
            'weights': weights.tolist(),
            'weak_classes': [int(c) for c in weak_classes]
        }, f, indent=2)
    print("\nSaved to results_v4_plif_fixed_v2/class_weights.json")

if __name__ == '__main__':
    analyze()
