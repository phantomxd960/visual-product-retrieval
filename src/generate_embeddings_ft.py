import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths.json")

MODEL_PATH = os.path.join(BASE_PATH, "models", "clip_finetuned.pth")

OUT_EMBED = os.path.join(BASE_PATH, "outputs", "embeddings_ft.npy")
OUT_PATHS = os.path.join(BASE_PATH, "outputs", "image_paths_ft.json")

# =====================================================
# SETTINGS
# =====================================================
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD IMAGE PATHS
# =====================================================
with open(PATH_FILE, "r") as f:
    image_paths = json.load(f)

print("Total images:", len(image_paths))

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading fine-tuned CLIP...")

model_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_name)

model = CLIPModel.from_pretrained(
    model_name,
    use_safetensors=True
).to(device)

# load fine-tuned weights
state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

print("Loaded weights from:")
print(MODEL_PATH)

# =====================================================
# ENCODE FUNCTION
# =====================================================
def encode_images(images):

    inputs = processor(
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled = vision_outputs.pooler_output

        feats = model.visual_projection(pooled)

    feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu().numpy()

# =====================================================
# GENERATE EMBEDDINGS
# =====================================================
all_embeddings = []

print("\nGenerating embeddings...\n")

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):

    batch_paths = image_paths[i:i+BATCH_SIZE]

    images = []
    valid_paths = []

    for p in batch_paths:

        full_path = os.path.join(IMAGE_DIR, p)

        try:
            img = Image.open(full_path).convert("RGB")
            images.append(img)
            valid_paths.append(p)

        except:
            continue

    if len(images) == 0:
        continue

    emb = encode_images(images)

    for e in emb:
        all_embeddings.append(e)

# =====================================================
# SAVE
# =====================================================
all_embeddings = np.array(all_embeddings).astype("float32")

np.save(OUT_EMBED, all_embeddings)

with open(OUT_PATHS, "w") as f:
    json.dump(image_paths, f)

print("\nSaved embeddings:")
print(OUT_EMBED)

print("\nSaved paths:")
print(OUT_PATHS)

print("\nEmbedding shape:", all_embeddings.shape)