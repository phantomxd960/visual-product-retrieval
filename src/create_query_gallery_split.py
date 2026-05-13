import os
import json
from collections import defaultdict

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    "image_paths.json"
)

QUERY_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    "query_paths.json"
)

GALLERY_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    "gallery_paths.json"
)

# =====================================================
# HELPER
# =====================================================
def get_item_id(path):
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("id_"):
            return p
    return None

# =====================================================
# LOAD ALL PATHS
# =====================================================
with open(INPUT_FILE, "r") as f:
    image_paths = json.load(f)

print("Total images:", len(image_paths))

# =====================================================
# GROUP BY ITEM ID
# =====================================================
groups = defaultdict(list)

for path in image_paths:
    item_id = get_item_id(path)
    if item_id is not None:
        groups[item_id].append(path)

print("Total unique IDs:", len(groups))

# =====================================================
# CREATE SPLIT
# =====================================================
query_paths = []
gallery_paths = []

for item_id, paths in groups.items():
    paths = sorted(paths)

    if len(paths) == 1:
        # Single image item: use in both query and gallery
        query_paths.append(paths[0])
        gallery_paths.append(paths[0])
    else:
        # First image as query
        query_paths.append(paths[0])

        # Remaining images as gallery
        gallery_paths.extend(paths[1:])

print("Query images:", len(query_paths))
print("Gallery images:", len(gallery_paths))

# =====================================================
# SAVE
# =====================================================
with open(QUERY_FILE, "w") as f:
    json.dump(query_paths, f, indent=2)

with open(GALLERY_FILE, "w") as f:
    json.dump(gallery_paths, f, indent=2)

print("\nSaved:")
print(QUERY_FILE)
print(GALLERY_FILE)