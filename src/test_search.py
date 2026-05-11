import os
import numpy as np
import faiss
import json

# ---------------- PATHS ---------------- #
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBED_FILE = os.path.join(BASE_PATH, "outputs", "embeddings.npy")
INDEX_FILE = os.path.join(BASE_PATH, "outputs", "faiss_hnsw.index")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths.json")

# ---------------- LOAD ---------------- #
print("Loading index...")
index = faiss.read_index(INDEX_FILE)

print("Loading embeddings...")
embeddings = np.load(EMBED_FILE).astype("float32")

with open(PATH_FILE) as f:
    paths = json.load(f)

print("Index size:", index.ntotal)

# ---------------- TEST SEARCH ---------------- #
print("\nTesting retrieval...")

# pick a random image
query_id = 100  # change this if you want
query_vector = embeddings[query_id].reshape(1, -1)

# search top 5
D, I = index.search(query_vector, 5)

print("\nQuery image:")
print(paths[query_id])

print("\nTop 5 results:")
for rank, idx in enumerate(I[0]):
    print(f"{rank+1}. {paths[idx]}  (score={D[0][rank]:.4f})")