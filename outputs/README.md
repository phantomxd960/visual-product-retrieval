This directory contains generated metadata and evaluation results.

Included files:

- captions.json
- query_paths.json
- gallery_paths.json
- results_seed\*.json
- results_query_gallery_seed\*.json

Large binary files such as embeddings and FAISS indexes are excluded and can be regenerated using:

python src/generate_embeddings_ft.py
python src/build_index_ft.py
