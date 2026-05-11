import os
import json
import faiss
import numpy as np

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBED_FILE = os.path.join(BASE_PATH, "outputs", "embeddings_ft.npy")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths_ft.json")

INDEX_FILE = os.path.join(BASE_PATH, "outputs", "faiss_hnsw_ft.index")

# =====================================================
# SETTINGS
# =====================================================
DIM = 512
M = 32

# =====================================================
# LOAD DATA
# =====================================================
print("Loading embeddings...")

embeddings = np.load(EMBED_FILE).astype("float32")

with open(PATH_FILE, "r") as f:
    image_paths = json.load(f)

print("Embedding shape:", embeddings.shape)
print("Total items:", len(image_paths))

# =====================================================
# BUILD INDEX
# =====================================================
print("Creating HNSW index...")

index = faiss.IndexHNSWFlat(DIM, M)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 128

print("Adding vectors...")
index.add(embeddings)

print("Indexed vectors:", index.ntotal)

# =====================================================
# SAVE
# =====================================================
faiss.write_index(index, INDEX_FILE)

print("Saved index:")
print(INDEX_FILE)