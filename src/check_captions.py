import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import random

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")
CAPTION_FILE = os.path.join(BASE_PATH, "outputs", "captions.json")
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs", "caption_check")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CAPTION_FILE) as f:
    captions = json.load(f)

keys = list(captions.keys())
samples = random.sample(keys, 20)

for i, k in enumerate(samples):

    img_path = os.path.join(IMAGE_DIR, k)
    caption = captions[k]

    img = Image.open(img_path)

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(caption)
    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, f"sample_{i}.png")
    plt.savefig(save_path)
    plt.close()

print("Saved caption check images to:", OUTPUT_DIR)