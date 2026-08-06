import os
r = "results"
counts = {}
for f in os.listdir(r):
    if not f.startswith("ablation_train_"):
        continue
    parts = f.split("_")
    if len(parts) >= 5:
        model = parts[3]
        counts[model] = counts.get(model, 0) + 1
for m, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"{m}: {c}")
print(f"Total: {sum(counts.values())}")
