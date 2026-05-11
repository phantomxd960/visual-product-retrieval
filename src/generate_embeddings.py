import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

# ---------------- PATHS ---------------- #
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")
CAPTION_FILE = os.path.join(BASE_PATH, "outputs", "captions.json")

OUTPUT_EMBEDDINGS = os.path.join(BASE_PATH, "outputs", "embeddings.npy")
OUTPUT_PATHS = os.path.join(BASE_PATH, "outputs", "image_paths.json")

# ---------------- SETTINGS ---------------- #
ALPHA = 0.7
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- LOAD CLIP ---------------- #
model_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(
    model_name,
    use_safetensors=True
)

model.to(device)
model.eval()

# ---------------- LOAD CAPTIONS ---------------- #
with open(CAPTION_FILE) as f:
    captions = json.load(f)

image_keys = list(captions.keys())

print("Total items:", len(image_keys))

embeddings = []
image_paths = []

# ---------------- GENERATE EMBEDDINGS ---------------- #
with torch.no_grad():

    for i in tqdm(range(0, len(image_keys), BATCH_SIZE)):

        batch_keys = image_keys[i:i+BATCH_SIZE]

        images = []
        texts = []

        for k in batch_keys:

            img_path = os.path.join(IMAGE_DIR, k)

            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                texts.append(captions[k])
                image_paths.append(k)
            except:
                continue

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = model(**inputs)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # normalize
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # fusion embedding
        fused = ALPHA * image_embeds + (1 - ALPHA) * text_embeds
        fused = fused / fused.norm(p=2, dim=-1, keepdim=True)

        embeddings.append(fused.cpu().numpy())

# ---------------- SAVE ---------------- #
embeddings = np.vstack(embeddings)

np.save(OUTPUT_EMBEDDINGS, embeddings)

with open(OUTPUT_PATHS, "w") as f:
    json.dump(image_paths, f)

print("Embeddings saved:", OUTPUT_EMBEDDINGS)
print("Paths saved:", OUTPUT_PATHS)