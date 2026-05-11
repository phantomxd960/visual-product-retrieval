import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_PATH = os.path.join(BASE_PATH, "data", "img")
BBOX_FILE = os.path.join(BASE_PATH, "data", "annotations", "list_bbox_inshop.txt")
SPLIT_FILE = os.path.join(BASE_PATH, "data", "annotations", "list_eval_partition.txt")

def load_bbox(file_path):
    bbox_dict = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()[2:]  # skip header

        for line in lines:
            parts = line.strip().split()

            img_path = parts[0]

            x1, y1, x2, y2 = map(int, parts[3:7])

            bbox_dict[img_path] = (x1, y1, x2, y2)

    return bbox_dict


def load_splits(file_path):
    split_dict = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()[2:]

        for line in lines:
            parts = line.strip().split()
            img_path = parts[0]
            split = parts[-1]

            split_dict[img_path] = split

    return split_dict


def build_dataset():
    bbox_data = load_bbox(BBOX_FILE)
    split_data = load_splits(SPLIT_FILE)

    dataset = []

    for img_path in bbox_data:
        full_img_path = os.path.join(BASE_PATH, "data", img_path)

        if not os.path.exists(full_img_path):
            continue

        data = {
            "image_path": full_img_path,
            "bbox": bbox_data[img_path],
            "split": split_data.get(img_path, "unknown")
        }

        dataset.append(data)

    return dataset


if __name__ == "__main__":
    dataset = build_dataset()

    print("Total samples:", len(dataset))
    print("\nSample data:\n")

    for i in range(3):
        print(dataset[i])