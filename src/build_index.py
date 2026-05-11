import os
import numpy as np
import faiss
import json

# ---------------- PATHS ---------------- #
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBED_FILE = os.path.join(BASE_PATH, "outputs", "embeddings.npy")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths.json")
INDEX_FILE = os.path.join(BASE_PATH, "outputs", "faiss_hnsw.index")

# ---------------- LOAD DATA ---------------- #
print("Loading embeddings...")

embeddings = np.load(EMBED_FILE).astype("float32")

with open(PATH_FILE, "r") as f:
    paths = json.load(f)

print("Embedding shape:", embeddings.shape)
print("Total items:", len(paths))

# ---------------- BUILD HNSW INDEX ---------------- #
dim = embeddings.shape[1]

print("Creating HNSW index...")

index = faiss.IndexHNSWFlat(dim, 32)

index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64

print("Adding vectors to index...")

index.add(embeddings)

print("Total indexed vectors:", index.ntotal)

# ---------------- SAVE INDEX ---------------- #
faiss.write_index(index, INDEX_FILE)

print("Index saved at:", INDEX_FILE)