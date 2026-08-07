import json

# V2 data from analyze_per_class.py output (stored in class_weights.json class_acc)
with open('results_v4_plif_fixed_v2/class_weights.json') as f:
    v2 = json.load(f)
with open('results_v4_plif_fixed_v3/v4_plif_results.json') as f:
    v3 = json.load(f)
with open('results_v4_plif_fixed_v4/v4_plif_results.json') as f:
    v4 = json.load(f)

labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','1','2','3','4','5','6','7','8','9','0','.',',','!','?']

print(f"{'='*80}")
print(f"Per-Class Accuracy: V2 → V3 → V4 (Focal Loss γ=2.0)")
print(f"{'='*80}")
print(f"Best Test Acc: V2=83.33%, V3=83.75%, V4=83.33%")
print(f"{'='*80}")
print(f"{'Char':>4} {'V2':>6} {'V3':>6} {'V4':>6} {'ΔV2→V4':>8} {'Status':>8}")
print('-'*50)

for c in range(40):
    acc2 = v2['class_acc'][str(c)] * 100
    acc3 = v3['final_class_acc'][str(c)] * 100
    acc4 = v4['final_class_acc'][str(c)] * 100
    delta = acc4 - acc2
    
    if acc4 >= 80: status = 'OK'
    elif acc4 >= acc2 + 5: status = 'UP'
    elif acc4 <= acc2 - 5: status = 'DOWN'
    else: status = 'SAME'
    
    print(f"{labels[c]:>4} {acc2:>5.1f}% {acc3:>5.1f}% {acc4:>5.1f}% {delta:>+7.1f}% {status:>8}")

print()
weak_v2 = [labels[c] for c in range(40) if v2['class_acc'][str(c)] < 0.8]
weak_v4 = [labels[c] for c in range(40) if v4['final_class_acc'][str(c)] < 0.8]
print(f"Weak classes (V2 < 80%): {len(weak_v2)} chars: {''.join(weak_v2)}")
print(f"Weak classes (V4 < 80%): {len(weak_v4)} chars: {''.join(weak_v4)}")

# Count improved
improved = sum(1 for c in range(40) if v4['final_class_acc'][str(c)] > v2['class_acc'][str(c)] + 0.05)
worsened = sum(1 for c in range(40) if v4['final_class_acc'][str(c)] < v2['class_acc'][str(c)] - 0.05)
print(f"\nImproved (>5%): {improved} classes | Worsened (>5%): {worsened} classes")
