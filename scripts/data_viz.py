import matplotlib.pyplot as plt
from data_loader import load_data_from_json
import numpy as np
import os
import re

# === Load your JSON file ===

data = load_data_from_json("Data/results/ndcg_scores_v5.json")

# === Create output directory ===
output_dir = "ndcg_plots_biggeralpha"
os.makedirs(output_dir, exist_ok=True)

# === Utility: sanitize query name for filenames ===
def sanitize_filename(name):
    return re.sub(r'[^\w\-_. ]', '_', name).replace(' ', '_')

# === Plotting ===
for query, topk_data in data.items():
    fig, axes = plt.subplots(1, len(topk_data), figsize=(6 * len(topk_data), 5))
    if len(topk_data) == 1:
        axes = [axes]

    for ax, (top_k, methods) in zip(axes, topk_data.items()):
        bars = []
        labels = []

        # Add sparse and dense
        bars.extend([methods.get("sparse", 0.0), methods.get("dense", 0.0)])
        labels.extend(["sparse", "dense"])

        # Add hybrid (sorted by alpha)
        hybrid = methods.get("hybrid", {})
        for alpha in sorted(hybrid, key=lambda x: float(x)):
            bars.append(hybrid[alpha])
            labels.append(f"hybrid α={alpha}")

        # Plotting
        x = np.arange(len(labels))
        ax.bar(x, bars, color='skyblue')
        ax.set_title(f"Query: '{query}'\nTop-k: {top_k}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel("nDCG")

    plt.tight_layout()

    # Save figure
    filename = f"{sanitize_filename(query)}.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close(fig)

    print(f"Saved plot for query '{query}' to {path}")
