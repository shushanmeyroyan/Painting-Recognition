# Painting Recognition Project Documentation

## 1. Project Purpose

This project recognizes known indexed paintings from user uploads. The user upload may include wall, frame, shadows, blur, perspective distortion, camera noise, or phone EXIF rotation. Most dataset paintings are clean no-frame images, but some Armenian dataset images may already include frames or wall context, so the query path tries both full-image and crop-normalized variants.

The new recognition logic is deliberately split into stages:

1. Find and normalize the painting crop.
2. Retrieve likely indexed candidates with DINOv2 image embeddings.
3. Confirm the candidate with geometric matching before claiming recognition.
4. Return `not_found` when either semantic retrieval or geometric verification is not strong enough.

## 2. Main Workflow

1. Keep the Armenian metadata columns as they are.
2. Load Armenian records from `data/datasets/armenian/metadata/Book1.xlsx` and `data/datasets/armenian/images`.
3. Generate synthetic wall/frame/perspective images from clean Armenian paintings.
4. Train a YOLO11-seg or YOLOv8-seg cropper on those synthetic images.
5. For every dataset painting, create augmented variants. The first stored variant is always the original image, so framed/wall Armenian records remain searchable as they exist in the dataset.
6. Extract `facebook/dinov2-base` embeddings from each variant.
7. Store all augmented embeddings in FAISS, but map them back to the original painting row.
8. For uploads, respect EXIF orientation, then try full-image, rotated, border-removed, and crop-normalized variants before choosing the best result.
9. Search top-k embeddings, group hits by original painting, and rank by aggregated score.
10. Verify the top candidates geometrically.
11. Return a recognized painting only when both embedding score and geometric inliers pass thresholds.
12. WikiArt can be used either as unknown-painting validation/style data or included in the identity index with `--include-wikiart`. When included, WikiArt paintings are treated equivalently to Armenian paintings for exact recognition.

## 3. Current Stack

- Painting crop detection: YOLO11-seg or YOLOv8-seg through Ultralytics.
- Crop fallback: existing OpenCV contour-based gallery detector when YOLO weights are missing.
- Identity embeddings: `facebook/dinov2-base`.
- Vector search: FAISS `IndexFlatIP` with L2-normalized vectors.
- Result aggregation: top-k augmented hits grouped by `painting_id`, regardless of whether the source is Armenian or WikiArt.
- Exact-image fallback: perceptual hash comparison before DINOv2, source-neutral for Armenian and WikiArt.
- Geometric verification: LightGlue + SuperPoint when available, with ORB homography as a local fallback. LoFTR can be added as an alternative verifier for weak texture or blur.
- Style recognition: kept separate from identity recognition. CLIP remains useful for zero-shot style fallback, not exact identity.

## 4. Important Commands

Generate synthetic YOLO segmentation data:

```bash
python main.py generate-yolo --samples-per-image 12
```

Generate synthetic YOLO data from Armenian + WikiArt paintings:

```bash
python main.py generate-yolo --samples-per-image 8 --include-wikiart --wikiart-limit 4500
```

Train the cropper:

```bash
python main.py train-cropper --model yolo11n-seg.pt
```

After training, copy the best weights to:

```text
data/models/painting_yolo_seg.pt
```

Build the Armenian-only DINOv2 identity index:

```bash
python main.py build-index --embedding-model facebook/dinov2-base --augmentations 16
```

Build Armenian + WikiArt identity index:

```bash
python main.py build-index --embedding-model facebook/dinov2-base --augmentations 8 --include-wikiart --wikiart-limit 4500
```

Query an uploaded image:

```bash
python main.py query path/to/upload.jpg --embedding-model facebook/dinov2-base --top-k 20
```

Export a manifest of indexed paintings and copy test samples:

```bash
python main.py export-manifest --samples-per-source 20
```

This creates:

```text
data/index_manifest.csv
data/test_samples/indexed/
```

Use `data/index_manifest.csv` to see exactly which Armenian and WikiArt paintings are in the current FAISS index. Use the copied sample images for quick Streamlit tests.

Sync visible processed-image copies from the current index:

```bash
python main.py sync-processed --source wikiart --clean --dataset-copy data/datasets/wikiart --manifest data/processed_images/wikiart_processed_manifest.csv
```

This copies the currently indexed WikiArt paintings into both `data/datasets/wikiart/` and `data/processed_images/wikiart/` while preserving their style subfolders. The two folders then contain the same 4500 files, and duplicate filenames from different WikiArt folders do not overwrite each other. The CSV manifest shows which raw dataset image maps to which visible dataset and processed copy.

Run the website:

```bash
python -m streamlit run frontend_app.py
```

## 5. Core Files

### `art_recognition/cropping.py`

Contains `PaintingCropper`, which tries to use `data/models/painting_yolo_seg.pt`. If YOLO weights are not available, it falls back to the existing OpenCV contour detector. It also performs perspective correction when a polygon is available and removes a small border from the crop.

### `art_recognition/synthetic_yolo.py`

Generates synthetic training samples by placing indexed paintings on random wall backgrounds, adding artificial frames, perspective distortion, brightness/contrast changes, blur, JPEG compression, and polygon labels for the true painting area. It can generate from Armenian paintings only or Armenian + WikiArt with `--include-wikiart`.

### `art_recognition/identity.py`

Contains DINOv2 embedding extraction, painting augmentations, perceptual hashing, FAISS result aggregation, and geometric verification with LightGlue/SuperPoint first and ORB fallback. Current thresholds:

```text
DINOv2 similarity threshold: 0.82
Geometric inlier threshold: 35
Perceptual hash distance threshold: 8
```

These are starting values and should be tuned with validation data.

### `art_recognition/pipeline.py`

Connects the cropper, DINOv2 identity index, FAISS search, aggregation, and geometric verification. Identity indexing uses Armenian paintings by default and can include WikiArt with `--include-wikiart`. Once WikiArt is included, Streamlit should recognize those indexed WikiArt paintings the same way it recognizes indexed Armenian paintings.

### `scripts/generate_synthetic_yolo.py`

Creates YOLO segmentation images and labels under `data/synthetic_yolo_paintings`.

### `scripts/train_cropper.py`

Runs Ultralytics training for YOLO segmentation.

### `scripts/build_index.py`

Builds the Armenian-only DINOv2 FAISS identity index. Each painting contributes multiple augmented embeddings.

### `scripts/query_index.py`

Runs the upload query flow: crop, normalize, embed, search, aggregate, verify.

### `scripts/export_index_manifest.py`

Exports one row per indexed painting to `data/index_manifest.csv` and optionally copies sample source images into `data/test_samples/indexed/`.

## 5.1 When To Rebuild Embeddings

You do not need to rebuild embeddings every time you run Streamlit. The files below are persistent artifacts:

```text
data/faiss_index.idx
data/embeddings.npy
data/index_mapping.json
data/build_report.json
```

Rebuild only when you change one of these:

- indexed sources, for example Armenian-only vs Armenian + WikiArt
- `--wikiart-limit`
- `--augmentations`
- DINOv2 model name
- dataset images or metadata

If none of those changed, restart Streamlit and use the existing index.

## 6. Metadata Mapping

The identity index stores generated IDs internally without changing the metadata file:

```python
painting_id = index_number
filename = record.filename
painter_name = record.artist
painting_name = record.title
year = record.year
```

All augmented embeddings from the same painting map back to the same `painting_id`.

## 7. Validation Plan

Do not evaluate by searching the exact same clean image that was indexed. That gives fake accuracy.

Use three validation sets:

1. Synthetic wall/frame photos generated from indexed paintings.
2. Real phone photos of indexed Armenian or WikiArt paintings when available.
3. Unknown paintings that are not in the identity index.

Expected behavior:

- Known indexed Armenian and indexed WikiArt paintings should pass DINOv2 retrieval and geometric verification.
- Unknown paintings outside the current index should return `not_found`.
- Visually similar but different paintings should fail geometric verification.

Recommended future metrics:

- recognition precision/recall/F1
- false positive rate on unknown paintings
- top-1/top-5 retrieval before verification
- geometric inlier distributions
- cropper segmentation IoU on labeled validation images

## 8. Limitations

- YOLO crop quality depends on synthetic training realism and should be validated with real phone photos.
- ORB verification is only a local fallback. LoFTR should be tested next for blur, weak texture, and repetitive regions.
- DINOv2 thresholds are starting values and must be tuned on validation sets.
- Style classification is not the identity system and should be trained/evaluated separately.
