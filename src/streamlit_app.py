# streamlit_app.py

import os
import json
import numpy as np
import faiss
import torch
from PIL import Image

import streamlit as st

from transformers import (
    CLIPProcessor,
    CLIPModel,
    Blip2Processor,
    Blip2ForConditionalGeneration
)
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
    "confirm the crop, and retrieve similar products using "
    "Fine-Tuned CLIP + BLIP-2 Semantic Re-ranking."
)

# =====================================================
# BEST CONFIGURATION
# =====================================================
BEST_SEED = 2023031
BEST_ALPHA_TAG = "09"

# =====================================================
# PATHS
# =====================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    f"clip_finetuned_seed{BEST_SEED}.pth"
)

INDEX_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"faiss_seed{BEST_SEED}_alpha{BEST_ALPHA_TAG}.index"
)

PATH_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"image_paths_seed{BEST_SEED}_alpha{BEST_ALPHA_TAG}.json"
)

EMBED_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    f"embeddings_seed{BEST_SEED}_alpha{BEST_ALPHA_TAG}.npy"
)

CAPTION_FILE = os.path.join(
    BASE_PATH,
    "outputs",
    "captions.json"
)

IMAGE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "cropped_images"
)

SEARCH_K = 15
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    # -------- CLIP --------
    clip_name = "openai/clip-vit-base-patch32"

    clip_processor = CLIPProcessor.from_pretrained(clip_name)

    clip_model = CLIPModel.from_pretrained(
        clip_name,
        use_safetensors=True
    ).to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    clip_model.load_state_dict(state_dict)
    clip_model.eval()

    # -------- BLIP-2 --------
    blip_processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-flan-t5-xl"
    )

    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    blip_model.eval()

    # -------- YOLO --------
    yolo_model = YOLO("yolov8n.pt")

    return (
        clip_processor,
        clip_model,
        blip_processor,
        blip_model,
        yolo_model
    )


@st.cache_resource
def load_index():
    index = faiss.read_index(INDEX_FILE)

    with open(PATH_FILE, "r") as f:
        image_paths = json.load(f)

    embeddings = np.load(EMBED_FILE).astype("float32")

    with open(CAPTION_FILE, "r") as f:
        captions = json.load(f)

    return index, image_paths, embeddings, captions


# =====================================================
# LOAD EVERYTHING
# =====================================================
with st.spinner("Loading models and index... (BLIP-2 may take a minute)"):
    (
        clip_processor,
        clip_model,
        blip_processor,
        blip_model,
        yolo_model
    ) = load_models()

    (
        index,
        image_paths,
        embeddings,
        captions
    ) = load_index()

st.success(
    f"System loaded successfully. Indexed {len(image_paths):,} products."
)

# =====================================================
# FUNCTIONS
# =====================================================
def detect_largest_box(image):
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
    if box is None:
        return image

    x1, y1, x2, y2 = box

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    if x2 <= x1 or y2 <= y1:
        return image

    if preference == "Upper Body":
        mid_y = y1 + (y2 - y1) // 2
        crop_box = (x1, y1, x2, mid_y)

    elif preference == "Lower Body":
        mid_y = y1 + (y2 - y1) // 2
        crop_box = (x1, mid_y, x2, y2)

    else:  # Full Body
        crop_box = (x1, y1, x2, y2)

    cropped = image.crop(crop_box)

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


def blip_score(query_image, caption):
    if not caption:
        caption = "a clothing item"

    try:
        inputs = blip_processor(
            images=query_image,
            text=f"Does this image match: {caption}?",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = blip_model.generate(
                **inputs,
                max_new_tokens=3
            )

        answer = blip_processor.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).lower().strip()

        if "yes" in answer:
            return 1.0
        elif "no" in answer:
            return 0.0
        else:
            return 0.5

    except Exception:
        return 0.5


def search(query_image, top_k):
    query_vec = encode_query(query_image)

    # Stage 1: FAISS retrieval
    _, I = index.search(query_vec, SEARCH_K)

    candidates = []

    for idx in I[0]:
        path = image_paths[idx]
        caption = captions.get(path, "a clothing item")

        clip_score = float(
            np.dot(query_vec[0], embeddings[idx])
        )

        # Stage 2: BLIP-2 semantic reranking
        semantic_score = blip_score(
            query_image,
            caption
        )

        final_score = 0.8 * clip_score + 0.2 * semantic_score

        candidates.append(
            (
                path,
                final_score,
                clip_score,
                semantic_score
            )
        )

    # Sort by combined score
    candidates = sorted(
        candidates,
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[:top_k]


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
            cropped = crop_by_preference(
                image,
                box,
                crop_preference
            )

        st.session_state["cropped_image"] = cropped

        if "results" in st.session_state:
            del st.session_state["results"]

    # ---------------------------------------------
    # Show crop preview
    # ---------------------------------------------
    if "cropped_image" in st.session_state:
        st.subheader("Detected Crop Preview")
        st.image(
            st.session_state["cropped_image"],
            width=350
        )

        col1, col2 = st.columns(2)

        # Confirm Crop
        with col1:
            if st.button("Confirm Crop and Search"):
                cropped = st.session_state["cropped_image"]

                with st.spinner(
                    "Searching with CLIP + BLIP-2 reranking..."
                ):
                    results = search(
                        cropped,
                        top_k
                    )

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

    for i, (
        path,
        final_score,
        clip_score,
        semantic_score
    ) in enumerate(results):

        full_path = os.path.join(
            IMAGE_DIR,
            path
        )

        try:
            result_img = Image.open(
                full_path
            ).convert("RGB")
        except Exception:
            continue

        with cols[i]:
            st.image(
                result_img,
                use_container_width=True
            )
            st.markdown(f"**Rank {i+1}**")
            st.markdown(
                f"**Final Score:** {final_score:.3f}"
            )
            st.markdown(
                f"CLIP: {clip_score:.3f}"
            )
            st.markdown(
                f"BLIP-2: {semantic_score:.2f}"
            )
            st.caption(path)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "Pipeline: Upload Image → YOLO Detection → Region Selection → "
    "Fine-Tuned CLIP Retrieval → BLIP-2 Semantic Re-ranking"
)