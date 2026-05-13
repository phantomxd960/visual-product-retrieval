# evaluate_ft_query_gallery.py

import os
import json
import argparse
import faiss
import numpy as np
from tqdm import tqdm

# =====================================================
# ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=2023032,
    help="Model seed"
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.7,
    help="Image-text fusion weight"
)
args = parser.parse_args()

SEED = args.seed
ALPHA = args.alpha
ALPHA_TAG = str(ALPHA).replace(".", "")

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")

INDEX_FILE = os.path.join(
    OUTPUT_DIR,
    f"faiss_seed{SEED}_alpha{ALPHA_TAG}.index"
)

GALLERY_PATH_FILE = os.path.join(
    OUTPUT_DIR,
    f"image_paths_seed{SEED}_alpha{ALPHA_TAG}.json"
)

GALLERY_EMBED_FILE = os.path.join(
    OUTPUT_DIR,
    f"embeddings_seed{SEED}_alpha{ALPHA_TAG}.npy"
)

QUERY_PATH_FILE = os.path.join(
    OUTPUT_DIR,
    "query_paths.json"
)

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    f"results_query_gallery_seed{SEED}_alpha{ALPHA_TAG}.json"
)

# =====================================================
# SETTINGS
# =====================================================
K_VALUES = [5, 10, 15]

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def get_item_id(path):
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("id_"):
            return p
    return None


def recall_at_k(flags, k):
    return 1.0 if any(flags[:k]) else 0.0


def average_precision_at_k(flags, k):
    hits = 0
    score = 0.0

    for i in range(min(k, len(flags))):
        if flags[i]:
            hits += 1
            score += hits / (i + 1)

    if hits == 0:
        return 0.0

    return score / hits


def dcg_at_k(flags, k):
    score = 0.0

    for i in range(min(k, len(flags))):
        rel = 1 if flags[i] else 0
        score += rel / np.log2(i + 2)

    return score


def ndcg_at_k(flags, k):
    dcg = dcg_at_k(flags, k)
    ideal = sorted(flags, reverse=True)
    idcg = dcg_at_k(ideal, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# =====================================================
# LOAD DATA
# =====================================================
print("Loading index...")
index = faiss.read_index(INDEX_FILE)

print("Loading gallery paths...")
with open(GALLERY_PATH_FILE, "r") as f:
    gallery_paths = json.load(f)

print("Loading gallery embeddings...")
gallery_embeddings = np.load(GALLERY_EMBED_FILE).astype("float32")

print("Loading query paths...")
with open(QUERY_PATH_FILE, "r") as f:
    query_paths = json.load(f)

print(f"Gallery size: {len(gallery_paths)}")
print(f"Query size: {len(query_paths)}")
print(f"Seed: {SEED}")
print(f"Alpha: {ALPHA}")

# =====================================================
# BUILD PATH -> EMBEDDING MAP
# =====================================================
gallery_map = {
    path: gallery_embeddings[i]
    for i, path in enumerate(gallery_paths)
}

# =====================================================
# INITIALIZE METRICS
# =====================================================
metrics = {
    k: {
        "recall": [],
        "map": [],
        "ndcg": []
    }
    for k in K_VALUES
}

# =====================================================
# EVALUATION
# =====================================================
print("\nRunning explicit query/gallery evaluation...\n")

valid_queries = 0

for query_path in tqdm(query_paths):
    query_id = get_item_id(query_path)

    if query_id is None:
        continue

    # If query image itself is not in gallery, use another image
    # from the same item_id that exists in the gallery.
    if query_path in gallery_map:
        query_vec = gallery_map[query_path]
    else:
        query_vec = None

        for gallery_path in gallery_paths:
            if get_item_id(gallery_path) == query_id:
                query_vec = gallery_map[gallery_path]
                break

        if query_vec is None:
            # No gallery image with same ID
            continue

    query_vec = query_vec.reshape(1, -1).astype("float32")

    # Search gallery index
    _, I = index.search(query_vec, max(K_VALUES))

    flags = []

    for idx in I[0]:
        candidate_id = get_item_id(gallery_paths[idx])
        flags.append(candidate_id == query_id)

    # Compute metrics
    for k in K_VALUES:
        metrics[k]["recall"].append(
            recall_at_k(flags, k)
        )
        metrics[k]["map"].append(
            average_precision_at_k(flags, k)
        )
        metrics[k]["ndcg"].append(
            ndcg_at_k(flags, k)
        )

    valid_queries += 1

# =====================================================
# RESULTS
# =====================================================
print("\n==============================")
print("EXPLICIT QUERY/GALLERY RESULTS")
print("==============================")
print(f"Valid Queries Evaluated: {valid_queries}")

results = {
    "seed": SEED,
    "alpha": ALPHA,
    "evaluation_protocol": "explicit_query_gallery_split",
    "valid_queries": valid_queries
}

for k in K_VALUES:
    recall = float(np.mean(metrics[k]["recall"]))
    mAP = float(np.mean(metrics[k]["map"]))
    ndcg = float(np.mean(metrics[k]["ndcg"]))

    results[f"Recall@{k}"] = recall
    results[f"mAP@{k}"] = mAP
    results[f"NDCG@{k}"] = ndcg

    print(f"\n@{k}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"mAP@{k}:    {mAP:.4f}")
    print(f"NDCG@{k}:   {ndcg:.4f}")

# =====================================================
# SAVE RESULTS
# =====================================================
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved results to:")
print(RESULTS_FILE)