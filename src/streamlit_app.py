# streamlit_app.py

import os
import json
import numpy as np
import faiss
import torch
from PIL import Image

import streamlit as st

from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Visual Product Search",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Visual Product Search Engine")
st.write(
    "Upload a fashion image, choose Upper Body / Lower Body / Full Body, "
    "confirm the crop, and retrieve similar products."
)

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_FILE = os.path.join(BASE_PATH, "outputs", "faiss_hnsw_ft.index")
PATH_FILE = os.path.join(BASE_PATH, "outputs", "image_paths_ft.json")
EMBED_FILE = os.path.join(BASE_PATH, "outputs", "embeddings_ft.npy")
MODEL_PATH = os.path.join(BASE_PATH, "models", "clip_finetuned.pth")
IMAGE_DIR = os.path.join(BASE_PATH, "outputs", "cropped_images")

SEARCH_K = 15
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    clip_name = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(clip_name)

    model = CLIPModel.from_pretrained(
        clip_name,
        use_safetensors=True
    ).to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    yolo_model = YOLO("yolov8n.pt")

    return processor, model, yolo_model


@st.cache_resource
def load_index():
    index = faiss.read_index(INDEX_FILE)

    with open(PATH_FILE, "r") as f:
        image_paths = json.load(f)

    embeddings = np.load(EMBED_FILE).astype("float32")

    return index, image_paths, embeddings


# =====================================================
# LOAD EVERYTHING
# =====================================================
with st.spinner("Loading models and index..."):
    clip_processor, clip_model, yolo_model = load_models()
    index, image_paths, embeddings = load_index()

st.success(f"System loaded successfully. Indexed {len(image_paths):,} products.")

# =====================================================
# FUNCTIONS
# =====================================================
def detect_largest_box(image):
    """
    Run YOLO and return the largest detected bounding box.
    Returns (x1, y1, x2, y2) or None.
    """
    results = yolo_model(image, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    best_box = None
    best_area = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = max(0, x2 - x1) * max(0, y2 - y1)

        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    return best_box


def crop_by_preference(image, box, preference):
    """
    preference:
        - Upper Body
        - Lower Body
        - Full Body
    """
    if box is None:
        return image

    x1, y1, x2, y2 = box

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    if x2 <= x1 or y2 <= y1:
        return image

    if preference == "Full Body":
        crop_box = (x1, y1, x2, y2)

    elif preference == "Upper Body":
        mid_y = y1 + (y2 - y1) // 2
        crop_box = (x1, y1, x2, mid_y)

    elif preference == "Lower Body":
        mid_y = y1 + (y2 - y1) // 2
        crop_box = (x1, mid_y, x2, y2)

    else:
        crop_box = (x1, y1, x2, y2)

    cropped = image.crop(crop_box)

    # Avoid invalid crops
    if cropped.width < 5 or cropped.height < 5:
        return image

    return cropped


def encode_query(image):
    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled = vision_outputs.pooler_output
        feats = clip_model.visual_projection(pooled)

    feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu().numpy().astype("float32")


def search(query_vec, top_k):
    D, I = index.search(query_vec, SEARCH_K)

    results = []
    seen = set()

    for idx in I[0]:
        path = image_paths[idx]

        # Avoid duplicates
        if path in seen:
            continue
        seen.add(path)

        score = float(np.dot(query_vec[0], embeddings[idx]))
        results.append((path, score))

        if len(results) >= top_k:
            break

    return results


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Settings")

crop_preference = st.sidebar.radio(
    "Select clothing region",
    ["Upper Body", "Lower Body", "Full Body"],
    index=2
)

top_k = st.sidebar.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=5
)

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded = st.file_uploader(
    "Upload a query image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, width=350)

    # ---------------------------------------------
    # Detect crop
    # ---------------------------------------------
    if st.button("Detect Clothing Region"):
        with st.spinner("Running YOLO detection..."):
            box = detect_largest_box(image)
            cropped = crop_by_preference(image, box, crop_preference)

        st.session_state["cropped_image"] = cropped

        # Remove previous results if any
        if "results" in st.session_state:
            del st.session_state["results"]

    # ---------------------------------------------
    # Show crop preview
    # ---------------------------------------------
    if "cropped_image" in st.session_state:
        st.subheader("Detected Crop Preview")
        st.image(st.session_state["cropped_image"], width=350)

        col1, col2 = st.columns(2)

        # Confirm Crop
        with col1:
            if st.button("Confirm Crop and Search"):
                cropped = st.session_state["cropped_image"]

                with st.spinner("Encoding query..."):
                    query_vec = encode_query(cropped)

                with st.spinner("Searching similar products..."):
                    results = search(query_vec, top_k)

                st.session_state["results"] = results

        # Re-crop
        with col2:
            if st.button("Re-crop"):
                if "cropped_image" in st.session_state:
                    del st.session_state["cropped_image"]
                if "results" in st.session_state:
                    del st.session_state["results"]
                st.rerun()

# =====================================================
# DISPLAY RESULTS
# =====================================================
if "results" in st.session_state:
    st.subheader("Top Retrieved Results")

    results = st.session_state["results"]
    cols = st.columns(len(results))

    for i, (path, score) in enumerate(results):
        full_path = os.path.join(IMAGE_DIR, path)

        try:
            result_img = Image.open(full_path).convert("RGB")
        except Exception:
            continue

        with cols[i]:
            st.image(result_img, use_container_width=True)
            st.markdown(f"**Rank {i+1}**")
            st.markdown(f"**Score:** {score:.3f}")
            st.caption(path)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "Pipeline: Upload Image → YOLO Detection → Region Selection → "
    "Crop Confirmation → Fine-Tuned CLIP Embedding → HNSW Retrieval"
)