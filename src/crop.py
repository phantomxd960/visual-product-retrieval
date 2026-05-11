import os
from PIL import Image
from tqdm import tqdm
import logging

from dataloader import build_dataset

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("crop.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

# ---------------- PATHS ---------------- #
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")
SUBSET_FILE = os.path.join(BASE_PATH, "data", "subset.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- LOAD SUBSET ---------------- #
def load_subset_paths(subset_file):
    with open(subset_file, "r") as f:
        subset_paths = set(line.strip() for line in f)

    logger.info(f"Loaded {len(subset_paths)} subset paths")
    return subset_paths


# ---------------- MAIN FUNCTION ---------------- #
def crop_and_save(dataset, subset_paths):
    success = 0
    skipped = 0
    failed = 0

    logger.info("Starting cropping...")

    for data in tqdm(dataset):
        img_path = data["image_path"]

        if img_path not in subset_paths:
            continue

        x1, y1, x2, y2 = data["bbox"]

        try:
            # Fix relative path
            rel_path = img_path.split("data")[-1].lstrip("\\/")
            save_path = os.path.join(OUTPUT_DIR, rel_path)

            if os.path.exists(save_path):
                skipped += 1
                continue

            img = Image.open(img_path).convert("RGB")

            # Crop
            cropped = img.crop((x1, y1, x2, y2))

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            cropped.save(save_path)

            success += 1

            # Log progress every 2000 images
            if success % 2000 == 0:
                logger.info(f"Processed {success} images")

        except Exception as e:
            failed += 1
            logger.error(f"Error with {img_path}: {e}")

    logger.info("Cropping completed")
    logger.info(f"Successful: {success}")
    logger.info(f"Skipped (already exists): {skipped}")
    logger.info(f"Failed: {failed}")


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    dataset = build_dataset()
    logger.info(f"Total dataset size: {len(dataset)}")

    subset_paths = load_subset_paths(SUBSET_FILE)

    crop_and_save(dataset, subset_paths)