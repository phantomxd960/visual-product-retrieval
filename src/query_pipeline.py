import os
import json
import faiss
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_FILE = os.path.join(BASE_PATH, "outputs", "faiss_hnsw.index")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths.json")
EMBED_FILE = os.path.join(BASE_PATH, "outputs", "embeddings.npy")
IMAGE_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")

# =====================================================
# SETTINGS
# =====================================================
TOP_K = 5
SEARCH_K = 10          # retrieve more, rerank later
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD CLIP
# =====================================================
print("Loading CLIP...")

clip_name = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(clip_name)

clip_model = CLIPModel.from_pretrained(
    clip_name,
    use_safetensors=True
).to(device)

clip_model.eval()

# =====================================================
# LOAD YOLO
# =====================================================
print("Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")

# =====================================================
# LOAD INDEX + EMBEDDINGS
# =====================================================
print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

with open(PATH_FILE, "r") as f:
    image_paths = json.load(f)

embeddings = np.load(EMBED_FILE).astype("float32")

print("Index size:", index.ntotal)
print("Embeddings shape:", embeddings.shape)

# =====================================================
# YOLO CROP
# =====================================================
def yolo_crop(image):
    results = yolo_model(image)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return image

    # choose largest box
    box = max(
        boxes,
        key=lambda b: (
            (b.xyxy[0][2] - b.xyxy[0][0]) *
            (b.xyxy[0][3] - b.xyxy[0][1])
        )
    )

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return image.crop((x1, y1, x2, y2))

# =====================================================
# QUERY IMAGE
# =====================================================
query_input = input(
    "\nEnter image path (relative to cropped_images): "
).strip()

query_path = os.path.join(IMAGE_DIR, query_input)

if not os.path.exists(query_path):
    raise FileNotFoundError(f"Invalid path:\n{query_path}")

image = Image.open(query_path).convert("RGB")

# =====================================================
# STEP 1: YOLO CROP
# =====================================================
cropped = yolo_crop(image)

# =====================================================
# STEP 2: CLIP EMBEDDING
# =====================================================
inputs = clip_processor(
    images=cropped,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    vision_outputs = clip_model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    pooled = vision_outputs.pooler_output
    image_features = clip_model.visual_projection(pooled)

# normalize
image_features = image_features / image_features.norm(
    dim=-1,
    keepdim=True
)

query_vec = image_features.cpu().numpy().astype("float32")

# =====================================================
# STEP 3: ANN SEARCH
# =====================================================
D, I = index.search(query_vec, SEARCH_K)

# =====================================================
# STEP 4: COSINE RERANK
# =====================================================
results = []

for idx in I[0]:
    candidate_vec = embeddings[idx]

    # cosine similarity (all vectors normalized)
    score = float(np.dot(query_vec[0], candidate_vec))

    results.append(
        (
            image_paths[idx],
            score
        )
    )

# highest similarity first
results = sorted(
    results,
    key=lambda x: x[1],
    reverse=True
)[:TOP_K]

# =====================================================
# DISPLAY
# =====================================================
plt.figure(figsize=(16, 4))

# Original
plt.subplot(1, TOP_K + 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

# Crop
plt.subplot(1, TOP_K + 2, 2)
plt.imshow(cropped)
plt.title("YOLO Crop")
plt.axis("off")

# Results
for i, (path, score) in enumerate(results):

    img_path = os.path.join(IMAGE_DIR, path)

    try:
        result_img = Image.open(img_path).convert("RGB")
    except:
        continue

    plt.subplot(1, TOP_K + 2, i + 3)
    plt.imshow(result_img)
    plt.title(f"Rank {i+1}\n{score:.3f}")
    plt.axis("off")

plt.tight_layout()
plt.show()