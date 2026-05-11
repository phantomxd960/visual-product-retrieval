import os
import json
import faiss
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_FILE = os.path.join(BASE_PATH, "outputs", "faiss_hnsw.index")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths.json")
EMBED_FILE = os.path.join(BASE_PATH, "outputs", "embeddings.npy")

# =====================================================
# SETTINGS
# =====================================================
K_VALUES = [5, 10, 15]

# =====================================================
# LOAD DATA
# =====================================================
print("Loading index...")
index = faiss.read_index(INDEX_FILE)

print("Loading paths...")
with open(PATH_FILE, "r") as f:
    image_paths = json.load(f)

print("Loading embeddings...")
embeddings = np.load(EMBED_FILE).astype("float32")

print("Total items:", len(image_paths))
print("Embedding shape:", embeddings.shape)

# =====================================================
# HELPERS
# =====================================================
def get_item_id(path):
    """
    Example:
    img/MEN/Denim/id_00003879/02_1_front.jpg
                     ^^^^^^^^^^^
    """
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("id_"):
            return p
    return None


def recall_at_k(relevant_flags, k):
    return 1.0 if any(relevant_flags[:k]) else 0.0


def average_precision_at_k(relevant_flags, k):
    hits = 0
    score = 0.0

    for i in range(min(k, len(relevant_flags))):
        if relevant_flags[i]:
            hits += 1
            score += hits / (i + 1)

    if hits == 0:
        return 0.0

    return score / hits


def dcg_at_k(relevant_flags, k):
    score = 0.0
    for i in range(min(k, len(relevant_flags))):
        rel = 1 if relevant_flags[i] else 0
        score += rel / np.log2(i + 2)
    return score


def ndcg_at_k(relevant_flags, k):
    dcg = dcg_at_k(relevant_flags, k)

    ideal = sorted(relevant_flags, reverse=True)
    idcg = dcg_at_k(ideal, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# =====================================================
# EVALUATION
# =====================================================
metrics = {
    k: {
        "recall": [],
        "map": [],
        "ndcg": []
    }
    for k in K_VALUES
}

print("\nRunning evaluation...\n")

for query_idx in tqdm(range(len(image_paths))):

    query_path = image_paths[query_idx]
    query_id = get_item_id(query_path)

    if query_id is None:
        continue

    query_vec = embeddings[query_idx].reshape(1, -1).astype("float32")

    max_k = max(K_VALUES) + 1  # +1 because query may retrieve itself
    D, I = index.search(query_vec, max_k)

    retrieved = []

    for idx in I[0]:
        if idx == query_idx:
            continue
        retrieved.append(idx)

    retrieved = retrieved[:max(K_VALUES)]

    relevant_flags = []

    for idx in retrieved:
        candidate_id = get_item_id(image_paths[idx])
        relevant_flags.append(candidate_id == query_id)

    # compute metrics
    for k in K_VALUES:
        metrics[k]["recall"].append(
            recall_at_k(relevant_flags, k)
        )

        metrics[k]["map"].append(
            average_precision_at_k(relevant_flags, k)
        )

        metrics[k]["ndcg"].append(
            ndcg_at_k(relevant_flags, k)
        )

# =====================================================
# FINAL RESULTS
# =====================================================
print("\n==============================")
print("FINAL EVALUATION RESULTS")
print("==============================")

for k in K_VALUES:
    recall = np.mean(metrics[k]["recall"])
    mAP = np.mean(metrics[k]["map"])
    ndcg = np.mean(metrics[k]["ndcg"])

    print(f"\n@{k}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"mAP@{k}:    {mAP:.4f}")
    print(f"NDCG@{k}:   {ndcg:.4f}")