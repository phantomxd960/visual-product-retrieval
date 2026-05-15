# Visual Product Retrieval System for Fashion Search

This project implements an end-to-end visual product retrieval system for fashion items using YOLO, BLIP-2, CLIP, FAISS HNSW, and Streamlit.

The system allows a user to upload a full-body image, select whether to search using the **Upper Body**, **Lower Body**, or **Full Body** clothing region, and retrieve visually similar products from the DeepFashion In-Shop Clothes dataset.

---

## Features

- YOLO-based clothing localization
- Upper Body / Lower Body / Full Body region selection
- BLIP-2 automatic caption generation
- CLIP multimodal embeddings
- Fine-tuned CLIP vision encoder
- FAISS HNSW approximate nearest-neighbor search
- Retrieval evaluation using Recall@K, mAP@K, and NDCG@K
- Streamlit interactive demo application

---

## Dataset

**DeepFashion In-Shop Clothes Retrieval Dataset**

Ground truth labels are based on `item_id`. Two images are considered relevant if and only if they share the same `item_id`.

---

## System Architecture

### Offline Pipeline

1. Crop product regions using dataset bounding box annotations
2. Generate captions using BLIP-2
3. Generate multimodal CLIP embeddings
4. Fine-tune the CLIP vision encoder
5. Regenerate embeddings using the fine-tuned model
6. Build the FAISS HNSW index

### Online Query Pipeline

1. Upload query image
2. Detect clothing region using YOLOv8
3. Select Upper Body / Lower Body / Full Body
4. Confirm crop
5. Encode using fine-tuned CLIP
6. Search the FAISS HNSW index
7. Display Top-K similar products

---

## Fine-Tuning Strategy

### Pretrained Models

- CLIP (OpenAI ViT-B/32)
- BLIP-2 (Salesforce BLIP2 FLAN-T5 XL)
- YOLOv8n

### Fine-Tuned Component

- CLIP vision encoder

### Frozen Components

- CLIP text encoder
- BLIP-2
- YOLO

### Objective

Bring images of the same item closer in embedding space while pushing different items farther apart.

---

## Evaluation Metrics

The following metrics are reported for **K = {5, 10, 15}**:

- Recall@K
- mAP@K
- NDCG@K

---

## Final Fine-Tuned Results

| Metric |     @5 |    @10 |    @15 |
| -----: | -----: | -----: | -----: |
| Recall | 0.5887 | 0.6638 | 0.7045 |
|    mAP | 0.4488 | 0.4360 | 0.4232 |
|   NDCG | 0.4326 | 0.4806 | 0.5086 |

---

## Project Structure

```text
Project/
├── data/
├── models/
│   └── clip_finetuned.pth
├── outputs/
│   ├── captions.json
│   ├── embeddings.npy
│   ├── embeddings_ft.npy
│   ├── faiss_hnsw.index
│   ├── faiss_hnsw_ft.index
│   └── image_paths_ft.json
├── src/
│   ├── crop.py
│   ├── generate_captions.py
│   ├── generate_embeddings.py
│   ├── generate_embeddings_ft.py
│   ├── build_index.py
│   ├── build_index_ft.py
│   ├── trainclip.py
│   ├── evaluate.py
│   ├── evaluate_ft.py
│   ├── query_pipeline.py
│   ├── query_pipeline_ft.py
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Project
```

### 2. Create a Virtual Environment

```bash
python -m venv vr_env
```

### 3. Activate the Environment

#### Windows (PowerShell)

```powershell
vr_env\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source vr_env/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Files

Place the following files in the appropriate directories.

### Model Checkpoint

```text
models/clip_finetuned.pth
```

### Generated Outputs

```text
outputs/embeddings_ft.npy
outputs/image_paths_ft.json
outputs/faiss_hnsw_ft.index
```

If these files are missing, regenerate them using the scripts below.

---

## Training

### Fine-Tune CLIP

```bash
python src/trainclip.py
```

### Generate Fine-Tuned Embeddings

```bash
python src/generate_embeddings_ft.py
```

### Build FAISS Index

```bash
python src/build_index_ft.py
```

---

## Evaluation

```bash
python src/evaluate_ft.py
```

---

## Streamlit Demo

```bash
streamlit run src/streamlit_app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## Demo Workflow

1. Upload an image
2. Choose Upper Body / Lower Body / Full Body
3. Detect clothing region using YOLO
4. Confirm crop
5. View Top-K retrieved results

---

## Baseline Pipeline

Run the pretrained (non-fine-tuned) version:

```bash
python src/query_pipeline.py
```

---

## Fine-Tuned Query Pipeline

```bash
python src/query_pipeline_ft.py
```

---

## Technologies Used

- Python
- PyTorch
- Transformers
- FAISS
- Ultralytics YOLO
- Streamlit
- Matplotlib

## Multi-Seed Training and Robustness Analysis

The fine-tuned CLIP model was trained using three different random seeds:

- 2023031
- 2023032
- 2023033

These seeds affect data shuffling and optimization, leading to slightly different model checkpoints. For each seed, the full retrieval pipeline was executed:

1. Fine-tune CLIP
2. Generate fused embeddings
3. Build FAISS index
4. Evaluate retrieval performance

To study the effect of semantic fusion, two image-text fusion weights were tested:

- α = 0.7
- α = 0.9

This produced a total of six experimental runs. Final results are reported as **mean ± standard deviation** across all runs.

---

## BLIP-2 Semantic Re-ranking

After the initial FAISS retrieval, the top candidates are re-ranked using BLIP-2.

For each retrieved item:

1. The stored caption is loaded.
2. BLIP-2 estimates whether the query image matches the caption.
3. A semantic compatibility score is generated.
4. The final score is computed as:

```text
Final Score = 0.8 × CLIP Similarity + 0.2 × BLIP-2 Semantic Score

Final Results (Explicit Query/Gallery Split)
Mean ± Standard Deviation (6 Runs)
Metric	@5	@10	@15
Recall	1.0000 ± 0.0000	1.0000 ± 0.0000	1.0000 ± 0.0000
mAP	0.9682 ± 0.0008	0.9337 ± 0.0019	0.9103 ± 0.0026
NDCG	0.9213 ± 0.0026	0.9446 ± 0.0017	0.9564 ± 0.0013
Best Configuration
Seed: 2023031
Alpha: 0.9
Recall@5: 1.0000
mAP@5: 0.9694
NDCG@5: 0.9233
Batch Evaluation Script

The project includes a batch evaluation script that runs the retrieval pipeline over a set of query images and computes all required metrics.

python src/evaluate_ft_query_gallery.py --seed 2023031 --alpha 0.9

This script computes:

Recall@5, Recall@10, Recall@15
mAP@5, mAP@10, mAP@15
NDCG@5, NDCG@10, NDCG@15

using the best-performing model.
```

## Additional Folder Notes

- data/README.md — Dataset download instructions
- models/README.md — Information about fine-tuned checkpoints
- outputs/README.md — Description of generated outputs and results
