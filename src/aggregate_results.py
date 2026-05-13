# aggregate_results.py

import os
import json
import numpy as np

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")

# =====================================================
# SELECT WHICH RESULT FILES TO AGGREGATE
# Set MODE = "standard" or MODE = "query_gallery"
# =====================================================
MODE = "query_gallery"   # change to "standard" if needed

# =====================================================
# FIND RESULT FILES
# =====================================================
all_files = os.listdir(OUTPUT_DIR)

if MODE == "standard":
    # Files produced by evaluate_ft.py
    result_files = [
        f for f in all_files
        if f.startswith("results_seed")
        and f.endswith(".json")
        and "query_gallery" not in f
    ]
else:
    # Files produced by evaluate_ft_query_gallery.py
    result_files = [
        f for f in all_files
        if f.startswith("results_query_gallery_seed")
        and f.endswith(".json")
    ]

if len(result_files) == 0:
    print("No result files found.")
    exit()

print(f"Loaded {len(result_files)} result files.")

# =====================================================
# LOAD RESULTS
# =====================================================
all_results = []

for filename in sorted(result_files):
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "r") as f:
        all_results.append(json.load(f))

# =====================================================
# METRICS
# =====================================================
metric_names = [
    "Recall@5", "Recall@10", "Recall@15",
    "mAP@5", "mAP@10", "mAP@15",
    "NDCG@5", "NDCG@10", "NDCG@15"
]

# =====================================================
# PRINT SUMMARY
# =====================================================
print("\n========================================")
print(f"MEAN ± STANDARD DEVIATION ({MODE.upper()})")
print("========================================")

for metric in metric_names:
    values = [
        r[metric]
        for r in all_results
        if metric in r
    ]

    if len(values) == 0:
        continue

    mean = np.mean(values)
    std = np.std(values)

    print(f"{metric:10s}: {mean:.4f} ± {std:.4f}")

# =====================================================
# PRINT INDIVIDUAL RUNS
# =====================================================
print("\n========================================")
print("INDIVIDUAL RUNS")
print("========================================")

for r in all_results:
    seed = r.get("seed", "N/A")
    alpha = r.get("alpha", "N/A")
    recall5 = r.get("Recall@5", 0.0)
    map5 = r.get("mAP@5", 0.0)
    ndcg5 = r.get("NDCG@5", 0.0)

    print(
        f"Seed={seed}, Alpha={alpha} -> "
        f"Recall@5={recall5:.4f}, "
        f"mAP@5={map5:.4f}, "
        f"NDCG@5={ndcg5:.4f}"
    )

# =====================================================
# FIND BEST CONFIGURATION (BY mAP@5)
# =====================================================
best = max(
    all_results,
    key=lambda x: x.get("mAP@5", 0.0)
)

print("\n========================================")
print("BEST CONFIGURATION")
print("========================================")
print(f"Seed      : {best.get('seed', 'N/A')}")
print(f"Alpha     : {best.get('alpha', 'N/A')}")
print(f"Recall@5  : {best.get('Recall@5', 0.0):.4f}")
print(f"mAP@5     : {best.get('mAP@5', 0.0):.4f}")
print(f"NDCG@5    : {best.get('NDCG@5', 0.0):.4f}")

if "evaluation_protocol" in best:
    print(f"Protocol  : {best['evaluation_protocol']}")

if "valid_queries" in best:
    print(f"Queries   : {best['valid_queries']}")