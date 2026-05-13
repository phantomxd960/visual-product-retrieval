import os
import json
import argparse
import faiss
import numpy as np

# =====================================================
# ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=2023031,
    help="Seed used to identify embedding files"
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

EMBED_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"embeddings_seed{SEED}_alpha{ALPHA_TAG}.npy"
)

PATH_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"image_paths_seed{SEED}_alpha{ALPHA_TAG}.json"
)

INDEX_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"faiss_seed{SEED}_alpha{ALPHA_TAG}.index"
)

# =====================================================
# SETTINGS
# =====================================================
DIM = 512
M = 32

# =====================================================
# LOAD DATA
# =====================================================
print("Loading embeddings...")
print("Embedding file:", EMBED_FILE)

embeddings = np.load(EMBED_FILE).astype("float32")

with open(PATH_FILE, "r") as f:
    image_paths = json.load(f)

print("Embedding shape:", embeddings.shape)
print("Total items:", len(image_paths))
print("Seed:", SEED)
print("Alpha:", ALPHA)

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

print("\nSaved index:")
print(INDEX_FILE)