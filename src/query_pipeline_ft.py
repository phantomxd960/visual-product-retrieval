import os
import json
import faiss
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from transformers import (
    CLIPProcessor,
    CLIPModel,
    Blip2Processor,
    Blip2ForConditionalGeneration
)
from ultralytics import YOLO

# =====================================================
# PATHS (BEST CONFIGURATION)
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Best model from experiments:
# Seed = 2023032
# Alpha = 0.7
MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    "clip_finetuned_seed2023032.pth"
)

INDEX_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    "faiss_seed2023032_alpha07.index"
)

PATH_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    "image_paths_seed2023032_alpha07.json"
)

EMBED_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    "embeddings_seed2023032_alpha07.npy"
)

CAPTION_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    "captions.json"
)

IMAGE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "cropped_images"
)

# =====================================================
# SETTINGS
# =====================================================
TOP_K = 5
SEARCH_K = 15
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD CLIP
# =====================================================
print("Loading fine-tuned CLIP...")

clip_name = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(clip_name)

clip_model = CLIPModel.from_pretrained(
    clip_name,
    use_safetensors=True
).to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

clip_model.load_state_dict(state_dict)
clip_model.eval()

# =====================================================
# LOAD BLIP-2
# =====================================================
print("Loading BLIP-2... (this may take a minute)")

blip_processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-flan-t5-xl"
)

blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

blip_model.eval()

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

with open(CAPTION_FILE, "r") as f:
    captions = json.load(f)

print("Index size:", index.ntotal)

# =====================================================
# YOLO CROP
# =====================================================
def yolo_crop(image):
    results = yolo_model(image, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return image

    # Largest detected box
    box = max(
        boxes,
        key=lambda b: (
            (b.xyxy[0][2] - b.xyxy[0][0]) *
            (b.xyxy[0][3] - b.xyxy[0][1])
        )
    )

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    if x2 <= x1 or y2 <= y1:
        return image

    return image.crop((x1, y1, x2, y2))

# =====================================================
# ENCODE QUERY
# =====================================================
def encode_query(image):
    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled = vision_outputs.pooler_output
        feats = clip_model.visual_projection(pooled)

    feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu().numpy().astype("float32")

# =====================================================
# BLIP-2 SEMANTIC SCORE
# =====================================================
def blip_score(query_image, caption):
    """
    Uses BLIP-2 to score how well the caption matches the query image.
    Higher score = better semantic match.
    """
    if not caption:
        caption = "a clothing item"

    try:
        inputs = blip_processor(
            images=query_image,
            text=f"Does this image match: {caption}?",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = blip_model.generate(
                **inputs,
                max_new_tokens=3
            )

        answer = blip_processor.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).lower().strip()

        # Simple heuristic scoring
        if "yes" in answer:
            return 1.0
        elif "no" in answer:
            return 0.0
        else:
            return 0.5

    except Exception:
        return 0.5

# =====================================================
# RETRIEVAL + BLIP-2 RERANKING
# =====================================================
def retrieve(query_image):
    # Step 1: Encode query
    query_vec = encode_query(query_image)

    # Step 2: Retrieve top SEARCH_K using FAISS
    D, I = index.search(query_vec, SEARCH_K)

    # Step 3: BLIP-2 reranking
    results = []

    for idx in I[0]:
        path = image_paths[idx]
        caption = captions.get(path, "a clothing item")

        # CLIP similarity
        clip_score = float(
            np.dot(query_vec[0], embeddings[idx])
        )

        # BLIP-2 semantic score
        semantic_score = blip_score(query_image, caption)

        # Combined score
        final_score = 0.8 * clip_score + 0.2 * semantic_score

        results.append(
            (
                path,
                final_score,
                clip_score,
                semantic_score
            )
        )

    # Sort by final score
    results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )[:TOP_K]

    return results

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
# STEP 2: RETRIEVE
# =====================================================
print("Running retrieval + BLIP-2 reranking...")
results = retrieve(cropped)

# =====================================================
# DISPLAY
# =====================================================
plt.figure(figsize=(16, 4))

# Original image
plt.subplot(1, TOP_K + 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

# YOLO crop
plt.subplot(1, TOP_K + 2, 2)
plt.imshow(cropped)
plt.title("YOLO Crop")
plt.axis("off")

# Retrieved results
for i, (path, final_score, clip_score, semantic_score) in enumerate(results):
    full_path = os.path.join(IMAGE_DIR, path)

    try:
        result_img = Image.open(full_path).convert("RGB")
    except Exception:
        continue

    plt.subplot(1, TOP_K + 2, i + 3)
    plt.imshow(result_img)
    plt.title(
        f"Rank {i+1}\n"
        f"Final={final_score:.3f}\n"
        f"BLIP={semantic_score:.2f}"
    )
    plt.axis("off")

    print(
        f"Rank {i+1}: {path}\n"
        f"  Final Score : {final_score:.4f}\n"
        f"  CLIP Score  : {clip_score:.4f}\n"
        f"  BLIP Score  : {semantic_score:.4f}\n"
    )

plt.tight_layout()
plt.show()