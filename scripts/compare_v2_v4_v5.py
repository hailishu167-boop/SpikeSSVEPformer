import json

# V2 class_acc from analyze script (stored in class_weights.json)
with open('results_v4_plif_fixed_v2/class_weights.json') as f:
    v2 = json.load(f)
with open('results_v4_plif_fixed_v4/v4_plif_results.json') as f:
    v4 = json.load(f)
with open('results_v4_plif_fixed_v5/v4_plif_results.json') as f:
    v5 = json.load(f)

labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','1','2','3','4','5','6','7','8','9','0','.',',','!','?']

print(f"{'='*90}")
print(f"Per-Class Accuracy: V2 → V4 (FocalLoss) → V5 (Data Augment)")
print(f"{'='*90}")
print(f"Best Test Acc: V2=83.33%, V4=83.33%, V5=84.17%")
print(f"{'='*90}")
print(f"{'Char':>4} {'V2':>6} {'V4':>6} {'V5':>6} {'ΔV2→V5':>8} {'Status':>8}")
print('-'*50)

for c in range(40):
    acc2 = v2['class_acc'][str(c)] * 100
    acc4 = v4['final_class_acc'][str(c)] * 100
    acc5 = v5['final_class_acc'][str(c)] * 100
    delta = acc5 - acc2
    
    if acc5 >= 80: status = 'OK'
    elif acc5 >= acc2 + 5: status = 'UP'
    elif acc5 <= acc2 - 5: status = 'DOWN'
    else: status = 'SAME'
    
    print(f"{labels[c]:>4} {acc2:>5.1f}% {acc4:>5.1f}% {acc5:>5.1f}% {delta:>+7.1f}% {status:>8}")

print()
weak_v2 = [labels[c] for c in range(40) if v2['class_acc'][str(c)] < 0.8]
weak_v5 = [labels[c] for c in range(40) if v5['final_class_acc'][str(c)] < 0.8]
print(f"Weak classes (V2 < 80%): {len(weak_v2)} chars: {''.join(weak_v2)}")
print(f"Weak classes (V5 < 80%): {len(weak_v5)} chars: {''.join(weak_v5)}")

# Count improved
improved = sum(1 for c in range(40) if v5['final_class_acc'][str(c)] > v2['class_acc'][str(c)] + 0.05)
worsened = sum(1 for c in range(40) if v5['final_class_acc'][str(c)] < v2['class_acc'][str(c)] - 0.05)
print(f"\nImproved (>5%): {improved} classes | Worsened (>5%): {worsened} classes")

# Highlight augmented classes
aug_classes = v5.get('augment_classes', [0, 5, 29])
print(f"\n{'='*90}")
print(f"Augmented classes results:")
print(f"{'='*90}")
for c in aug_classes:
    acc2 = v2['class_acc'][str(c)] * 100
    acc5 = v5['final_class_acc'][str(c)] * 100
    print(f"  {labels[c]} (class {c}): V2={acc2:.1f}% → V5={acc5:.1f}% (Δ{acc5-acc2:+.1f}%)")
