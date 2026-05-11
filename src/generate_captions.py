import os
import torch
from PIL import Image
from tqdm import tqdm
import json

from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ---------------- PATHS ---------------- #
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")
OUTPUT_FILE = os.path.join(BASE_PATH, "outputs", "captions.json")

# ---------------- DEVICE ---------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# RTX optimization
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------- SETTINGS ---------------- #
BATCH_SIZE = 8
PROMPT = "Describe the clothing item including color, type and style."

# ---------------- LOAD MODEL ---------------- #
model_name = "Salesforce/blip2-flan-t5-xl"

processor = Blip2Processor.from_pretrained(
    model_name,
    backend="torchvision"
)

model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.float16
)

model.to(device)
model.eval()

print("Model loaded on:", next(model.parameters()).device)

# ---------------- COLLECT IMAGES ---------------- #
image_paths = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, file))

image_paths.sort()

print(f"Total images found: {len(image_paths)}")

# ---------------- GENERATE CAPTIONS ---------------- #
captions = {}

with torch.no_grad():

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):

        batch_paths = image_paths[i:i + BATCH_SIZE]

        images = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"Skipping {p}: {e}")

        if len(images) == 0:
            continue

        inputs = processor(
            images=images,
            text=[PROMPT] * len(images),
            return_tensors="pt",
            padding=True
        )

        # Move tensors correctly to GPU
        inputs = {
            k: (v.to(device) if v.dtype == torch.long else v.to(device, torch.float16))
            for k, v in inputs.items()
        }

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30
        )

        batch_captions = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        for path, caption in zip(valid_paths, batch_captions):
            rel_path = os.path.relpath(path, IMAGE_DIR)
            captions[rel_path] = caption

# ---------------- SAVE ---------------- #
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(captions, f, indent=4)

print("Captions saved to:", OUTPUT_FILE)