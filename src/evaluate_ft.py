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
parser.add_argument("--seed", type=int, default=2023032)
parser.add_argument("--alpha", type=float, default=0.7)
args = parser.parse_args()

SEED = args.seed
ALPHA = args.alpha
ALPHA_TAG = str(ALPHA).replace(".", "")

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"faiss_seed{SEED}_alpha{ALPHA_TAG}.index"
)

GALLERY_PATH_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"image_paths_seed{SEED}_alpha{ALPHA_TAG}.json"
)

GALLERY_EMBED_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"embeddings_seed{SEED}_alpha{ALPHA_TAG}.npy"
)

QUERY_PATH_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    "query_paths.json"
)

RESULTS_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"results_seed{SEED}_alpha{ALPHA_TAG}.json"
)

# =====================================================
# SETTINGS
# =====================================================
K_VALUES = [5, 10, 15]

# =====================================================
# LOAD
# =====================================================
print("Loading index...")
index = faiss.read_index(INDEX_FILE)

with open(GALLERY_PATH_FILE, "r") as f:
    gallery_paths = json.load(f)

gallery_embeddings = np.load(GALLERY_EMBED_FILE).astype("float32")

with open(QUERY_PATH_FILE, "r") as f:
    query_paths = json.load(f)

print("Gallery size:", len(gallery_paths))
print("Query size:", len(query_paths))

# =====================================================
# HELPER
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
    return score / hits if hits > 0 else 0.0

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
    return dcg / idcg if idcg > 0 else 0.0

# =====================================================
# MAP QUERY PATH -> EMBEDDING
# =====================================================
gallery_map = {
    path: gallery_embeddings[i]
    for i, path in enumerate(gallery_paths)
}

# =====================================================
# EVALUATION
# =====================================================
metrics = {
    k: {"recall": [], "map": [], "ndcg": []}
    for k in K_VALUES
}

print("\nRunning explicit query/gallery evaluation...\n")

for query_path in tqdm(query_paths):
    if query_path not in gallery_map:
        # If query image is not in gallery, skip
        continue

    query_vec = gallery_map[query_path].reshape(1, -1).astype("float32")
    query_id = get_item_id(query_path)

    if query_id is None:
        continue

    _, I = index.search(query_vec, max(K_VALUES))

    flags = []

    for idx in I[0]:
        cand_id = get_item_id(gallery_paths[idx])
        flags.append(cand_id == query_id)

    for k in K_VALUES:
        metrics[k]["recall"].append(recall_at_k(flags, k))
        metrics[k]["map"].append(average_precision_at_k(flags, k))
        metrics[k]["ndcg"].append(ndcg_at_k(flags, k))

# =====================================================
# RESULTS
# =====================================================
results = {
    "seed": SEED,
    "alpha": ALPHA,
    "evaluation_protocol": "explicit_query_gallery_split"
}

print("\n==============================")
print("EXPLICIT QUERY/GALLERY RESULTS")
print("==============================")

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

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved to:")
print(RESULTS_FILE)