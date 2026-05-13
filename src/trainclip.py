import os
import json
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel

# =====================================================
# ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=2023031,
    help="Random seed"
)
args = parser.parse_args()

SEED = args.seed

# =====================================================
# SET RANDOM SEEDS
# =====================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths.json")
IMAGE_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")

SAVE_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(
    SAVE_DIR,
    f"clip_finetuned_seed{SEED}.pth"
)

# =====================================================
# SETTINGS
# =====================================================
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-5
MARGIN = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using seed: {SEED}")
print(f"Device: {device}")

# =====================================================
# LOAD PATHS
# =====================================================
with open(PATH_FILE, "r") as f:
    image_paths = json.load(f)

# =====================================================
# HELPERS
# =====================================================
def get_item_id(path):
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("id_"):
            return p
    return None


# Group images by item ID
id_to_images = {}

for p in image_paths:
    item_id = get_item_id(p)

    if item_id is None:
        continue

    if item_id not in id_to_images:
        id_to_images[item_id] = []

    id_to_images[item_id].append(p)

# Only IDs with at least 2 images can form anchor-positive pairs
valid_ids = [
    item_id
    for item_id, imgs in id_to_images.items()
    if len(imgs) >= 2
]

print("Total valid IDs:", len(valid_ids))

# =====================================================
# DATASET
# =====================================================
class TripletFashionDataset(Dataset):
    def __init__(self, valid_ids):
        self.valid_ids = valid_ids

    def __len__(self):
        # Artificially increase number of triplets
        return len(self.valid_ids) * 10

    def __getitem__(self, idx):
        # -------- Anchor and Positive --------
        anchor_id = random.choice(self.valid_ids)
        imgs = id_to_images[anchor_id]

        anchor_path, positive_path = random.sample(imgs, 2)

        # -------- Negative --------
        negative_id = random.choice(self.valid_ids)

        while negative_id == anchor_id:
            negative_id = random.choice(self.valid_ids)

        negative_path = random.choice(
            id_to_images[negative_id]
        )

        # -------- Load Images --------
        anchor_img = Image.open(
            os.path.join(IMAGE_DIR, anchor_path)
        ).convert("RGB")

        positive_img = Image.open(
            os.path.join(IMAGE_DIR, positive_path)
        ).convert("RGB")

        negative_img = Image.open(
            os.path.join(IMAGE_DIR, negative_path)
        ).convert("RGB")

        return anchor_img, positive_img, negative_img

# =====================================================
# COLLATE FUNCTION
# =====================================================
def collate_fn(batch):
    anchors = []
    positives = []
    negatives = []

    for a, p, n in batch:
        anchors.append(a)
        positives.append(p)
        negatives.append(n)

    return anchors, positives, negatives

# =====================================================
# LOAD CLIP
# =====================================================
print("Loading CLIP...")

clip_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(clip_name)

model = CLIPModel.from_pretrained(
    clip_name,
    use_safetensors=True
).to(device)

# =====================================================
# FREEZE ALL PARAMETERS
# =====================================================
for param in model.parameters():
    param.requires_grad = False

# =====================================================
# UNFREEZE LAST 4 VISION BLOCKS
# =====================================================
vision_layers = model.vision_model.encoder.layers

for layer in vision_layers[-4:]:
    for param in layer.parameters():
        param.requires_grad = True

# Train visual projection layer
for param in model.visual_projection.parameters():
    param.requires_grad = True

# =====================================================
# LOSS + OPTIMIZER
# =====================================================
criterion = nn.TripletMarginLoss(
    margin=MARGIN,
    p=2
)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# =====================================================
# DATA LOADER
# =====================================================
dataset = TripletFashionDataset(valid_ids)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

# =====================================================
# EMBEDDING FUNCTION
# =====================================================
def encode_images(images):
    inputs = processor(
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Gradients enabled because we are training
    vision_outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    pooled = vision_outputs.pooler_output
    feats = model.visual_projection(pooled)

    feats = feats / feats.norm(
        dim=-1,
        keepdim=True
    )

    return feats

# =====================================================
# TRAIN LOOP
# =====================================================
print("\nStarting training...\n")

best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader)

    for anchor_imgs, pos_imgs, neg_imgs in loop:
        # -------- Encode --------
        anchor_feats = encode_images(anchor_imgs)
        pos_feats = encode_images(pos_imgs)
        neg_feats = encode_images(neg_imgs)

        # -------- Triplet Loss --------
        loss = criterion(
            anchor_feats,
            pos_feats,
            neg_feats
        )

        # -------- Optimize --------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )
        loop.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader)

    print(f"\nEpoch {epoch + 1} Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            SAVE_PATH
        )

        print("Saved best model.")

# =====================================================
# DONE
# =====================================================
print("\nTraining complete.")
print("Best model saved at:")
print(SAVE_PATH)