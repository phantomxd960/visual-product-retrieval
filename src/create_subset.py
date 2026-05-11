import random
import logging
from collections import defaultdict
import os

from dataloader import build_dataset

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ---------------- CONFIG ---------------- #
SUBSET_SIZE = 20000
RANDOM_SEED = 42   # for reproducibility

# ---------------- PATHS ---------------- #
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET_PATH = os.path.join(BASE_PATH, "data", "subset.txt")


# ---------------- FUNCTION ---------------- #
def create_subset(dataset, subset_size):
    random.seed(RANDOM_SEED)

    # Group by split
    split_groups = defaultdict(list)
    for data in dataset:
        split_groups[data["split"]].append(data)

    total = len(dataset)
    logger.info(f"Total dataset: {total}")

    subset = []

    # Step 1: proportional sampling
    for split, items in split_groups.items():
        ratio = len(items) / total
        sample_size = int(subset_size * ratio)

        logger.info(f"{split}: selecting {sample_size} samples (ratio={ratio:.2f})")

        sampled = random.sample(items, sample_size)
        subset.extend(sampled)

    # Step 2: Fix rounding issue (if not exact size)
    remaining = subset_size - len(subset)

    if remaining > 0:
        logger.info(f"Adding {remaining} extra samples to match exact subset size")

        all_items = dataset.copy()
        random.shuffle(all_items)

        subset.extend(all_items[:remaining])

    logger.info(f"Final subset size: {len(subset)}")

    return subset


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    dataset = build_dataset()

    logger.info("Dataset loaded successfully")

    subset = create_subset(dataset, SUBSET_SIZE)

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBSET_PATH), exist_ok=True)

    # Save subset
    with open(SUBSET_PATH, "w") as f:
        for item in subset:
            f.write(item["image_path"] + "\n")

    logger.info(f"Subset saved at: {SUBSET_PATH}")