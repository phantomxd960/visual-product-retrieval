import os
import json
import faiss
import numpy as np
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_FILE = os.path.join(BASE_PATH, "outputs", "faiss_hnsw_ft.index")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths_ft.json")
EMBED_FILE = os.path.join(BASE_PATH, "outputs", "embeddings_ft.npy")

# =====================================================
# SETTINGS
# =====================================================
K_VALUES = [5, 10, 15]

# =====================================================
# LOAD
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
# EVAL
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

    max_k = max(K_VALUES) + 1
    D, I = index.search(query_vec, max_k)

    retrieved = []

    for idx in I[0]:
        if idx == query_idx:
            continue
        retrieved.append(idx)

    retrieved = retrieved[:max(K_VALUES)]

    flags = []

    for idx in retrieved:
        cand_id = get_item_id(image_paths[idx])
        flags.append(cand_id == query_id)

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

# =====================================================
# RESULTS
# =====================================================
print("\n==============================")
print("FINE-TUNED CLIP RESULTS")
print("==============================")

for k in K_VALUES:

    recall = np.mean(metrics[k]["recall"])
    mAP = np.mean(metrics[k]["map"])
    ndcg = np.mean(metrics[k]["ndcg"])

    print(f"\n@{k}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"mAP@{k}:    {mAP:.4f}")
    print(f"NDCG@{k}:   {ndcg:.4f}")