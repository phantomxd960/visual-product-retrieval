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

INDEX_FILE = os.path.join(BASE_PATH, "outputs", "faiss_hnsw_ft.index")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths_ft.json")
EMBED_FILE = os.path.join(BASE_PATH, "outputs", "embeddings_ft.npy")

IMAGE_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")
MODEL_PATH = os.path.join(BASE_PATH, "models", "clip_finetuned.pth")

# =====================================================
# SETTINGS
# =====================================================
TOP_K = 5
SEARCH_K = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD CLIP
# =====================================================
print("Loading fine-tuned CLIP...")

clip_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(clip_name)

model = CLIPModel.from_pretrained(
    clip_name,
    use_safetensors=True
).to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

# =====================================================
# LOAD YOLO
# =====================================================
print("Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")

# =====================================================
# LOAD INDEX + DATA
# =====================================================
print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)

with open(PATH_FILE, "r") as f:
    image_paths = json.load(f)

embeddings = np.load(EMBED_FILE).astype("float32")

print("Index size:", index.ntotal)

# =====================================================
# YOLO CROP
# =====================================================
def yolo_crop(image):
    results = yolo_model(image)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return image

    # largest box
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
# QUERY INPUT
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
# STEP 2: QUERY EMBEDDING
# =====================================================
inputs = processor(
    images=cropped,
    return_tensors="pt"
).to(device)

with torch.no_grad():

    vision_outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    pooled = vision_outputs.pooler_output

    feats = model.visual_projection(pooled)

feats = feats / feats.norm(dim=-1, keepdim=True)

query_vec = feats.cpu().numpy().astype("float32")

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

    score = float(
        np.dot(query_vec[0], candidate_vec)
    )

    results.append(
        (
            image_paths[idx],
            score
        )
    )

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

# YOLO crop
plt.subplot(1, TOP_K + 2, 2)
plt.imshow(cropped)
plt.title("YOLO Crop")
plt.axis("off")

# Results
for i, (path, score) in enumerate(results):

    full_path = os.path.join(IMAGE_DIR, path)

    try:
        result_img = Image.open(full_path).convert("RGB")
    except:
        continue

    plt.subplot(1, TOP_K + 2, i + 3)
    plt.imshow(result_img)
    plt.title(f"Rank {i+1}\n{score:.3f}")
    plt.axis("off")

plt.tight_layout()
plt.show()