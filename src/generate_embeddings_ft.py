import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

# =====================================================
# ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=2023031,
    help="Seed used to select fine-tuned model"
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.7,
    help="Image-text fusion weight (0 to 1)"
)
args = parser.parse_args()

SEED = args.seed
ALPHA = args.alpha
ALPHA_TAG = str(ALPHA).replace(".", "")

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths.json")
CAPTION_FILE = os.path.join(BASE_PATH, "outputs", "captions.json")

MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    f"clip_finetuned_seed{SEED}.pth"
)

OUT_EMBED = os.path.join(
    BASE_PATH,
    "outputs",
    f"embeddings_seed{SEED}_alpha{ALPHA_TAG}.npy"
)

OUT_PATHS = os.path.join(
    BASE_PATH,
    "outputs",
    f"image_paths_seed{SEED}_alpha{ALPHA_TAG}.json"
)

# =====================================================
# SETTINGS
# =====================================================
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD INPUT FILES
# =====================================================
with open(PATH_FILE, "r") as f:
    image_paths = json.load(f)

with open(CAPTION_FILE, "r") as f:
    captions = json.load(f)

print("Total images:", len(image_paths))
print("Loaded captions:", len(captions))
print("Seed:", SEED)
print("Alpha:", ALPHA)

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

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("Loaded model from:")
print(MODEL_PATH)

# =====================================================
# ENCODE FUNCTION
# =====================================================
def encode_batch(images, texts):
    image_inputs = processor(
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)

    text_inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        # Image features
        vision_outputs = model.vision_model(
            pixel_values=image_inputs["pixel_values"]
        )

        image_pooled = vision_outputs.pooler_output
        image_features = model.visual_projection(image_pooled)

        # Text features
        text_outputs = model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )

        text_pooled = text_outputs.pooler_output
        text_features = model.text_projection(text_pooled)

    # Normalize
    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

    # Cross-modal fusion
    fused = ALPHA * image_features + (1.0 - ALPHA) * text_features

    # Final normalization
    fused = fused / fused.norm(
        dim=-1,
        keepdim=True
    )

    return fused.cpu().numpy()

# =====================================================
# GENERATE EMBEDDINGS
# =====================================================
all_embeddings = []
valid_paths = []

print("\nGenerating fused embeddings...\n")

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i + BATCH_SIZE]

    images = []
    texts = []
    batch_valid_paths = []

    for p in batch_paths:
        full_path = os.path.join(IMAGE_DIR, p)

        try:
            img = Image.open(full_path).convert("RGB")
        except Exception:
            continue

        caption = captions.get(p, "")
        if not caption:
            caption = "a clothing item"

        images.append(img)
        texts.append(caption)
        batch_valid_paths.append(p)

    if len(images) == 0:
        continue

    embeddings = encode_batch(images, texts)

    all_embeddings.extend(embeddings)
    valid_paths.extend(batch_valid_paths)

# =====================================================
# SAVE
# =====================================================
all_embeddings = np.array(all_embeddings).astype("float32")

np.save(OUT_EMBED, all_embeddings)

with open(OUT_PATHS, "w") as f:
    json.dump(valid_paths, f)

print("\nSaved embeddings:")
print(OUT_EMBED)

print("\nSaved paths:")
print(OUT_PATHS)

print("\nEmbedding shape:", all_embeddings.shape)
print("Valid paths:", len(valid_paths))