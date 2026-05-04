# Capstone Project Report Notes: AI Painting Recognition System

## 1. Project Overview

This project is an AI-powered painting recognition and visual analysis system. The goal is to let a user upload a painting image and receive the best available information about it:

- whether the uploaded image matches a known indexed painting
- painting title, painter, and year when a match is reliable
- predicted artistic style
- predicted or inferred genre
- visually similar paintings
- diagnostic confidence values for similarity and geometric verification

The project started as a simpler image-similarity application and evolved into a multi-stage recognition pipeline. The final design separates the problem into several smaller tasks:

1. Prepare the uploaded image and handle borders, frames, wall context, and rotation.
2. Extract robust visual embeddings with DINOv2.
3. Search a FAISS vector database.
4. Aggregate multiple augmented matches back to the original painting.
5. Confirm likely matches with geometric verification.
6. Predict style and genre even when the exact artwork is not recognized.

This separation is important because painting recognition from user uploads is harder than matching clean dataset images. Users may upload photos with frames, walls, perspective distortion, shadows, blur, compression, or phone orientation problems. A single embedding comparison is often not enough.

## 2. Datasets

### Armenian Dataset

The Armenian dataset is the local curated dataset used for Armenian painting recognition. It is loaded from:

```text
data/datasets/armenian/images
data/datasets/armenian/metadata/Book1.xlsx
```

The metadata is loaded without changing the original columns. Internally, the system creates a generated identity record:

```python
painting_id = index_number
filename = record.filename
painter_name = record.artist
painting_name = record.title
year = record.year
```

Some Armenian images are clean painting images, while others may already include frames or wall context. This became important because an overly aggressive cropper can damage an already valid dataset image. The final query flow therefore tries both full-image and cropped/normalized variants.

### WikiArt Dataset

WikiArt is used in two ways:

1. As part of the identity index when `--include-wikiart` is used.
2. As style/genre training data because WikiArt folder names provide style labels.

The full raw WikiArt dataset is stored in:

```text
data/datasets/wikiart_raw
```

This raw folder contains many more paintings than the current identity index. The selected visible 4500-image subset is stored in:

```text
data/datasets/wikiart
data/processed_images/wikiart
```

These two folders were synchronized so they contain the exact same 4500 indexed WikiArt paintings. The raw folder remains the larger source library.

Current identity index summary from `data/build_report.json`:

```text
identity model: facebook/dinov2-base
total indexed paintings: 4709
Armenian paintings: 209
WikiArt paintings: 4500
total embeddings: 18836
augmentations per painting: 4
include WikiArt: true
```

## 3. Earlier System Versions

### First Approach: ResNet50

The earliest supported embedding model was ResNet50 with ImageNet pretrained weights. The final classification layer was removed, producing a 2048-dimensional feature vector.

Why it was useful:

- easy to load with `torchvision`
- fast and well-known
- provided a baseline for image similarity

Why it was not enough:

- ImageNet pretraining is optimized for object categories, not fine-grained artwork identity
- it was not strong enough for visually similar paintings
- it did not handle artistic style and composition as well as newer vision foundation models

### Second Approach: CLIP

The next major approach used CLIP, specifically:

```text
openai/clip-vit-base-patch32
```

CLIP was useful because it maps images and text into a shared embedding space. It can support both image retrieval and text-based style prompts.

What worked:

- better semantic understanding than ResNet50
- useful for zero-shot style labels
- good for broad visual similarity
- convenient fallback for style descriptions

What did not work well:

- CLIP is semantic, not always instance-specific
- two paintings with similar subjects or styles could be close even when they are not the same artwork
- exact recognition of a specific painting requires stronger instance retrieval
- user-uploaded images with frames or wall context could confuse the embedding

CLIP-based evaluation reports showed high accuracy when testing clean images already present in the index, but this was not a realistic test. Searching the same clean image that was indexed gives artificially high scores. This helped reveal that evaluation needed to be redesigned.

### Third Approach: WikiArt-SigLIP

The project also experimented with a WikiArt fine-tuned SigLIP-style model:

```text
prithivMLmods/WikiArt-Style
```

The idea was to use a model closer to paintings and trained on WikiArt styles.

What helped:

- better style awareness than generic ImageNet features
- more directly connected to painting categories

What did not solve the main problem:

- style classification and exact painting identity are different tasks
- a model trained to recognize style may group paintings by movement or visual language, not by exact artwork identity
- user photos with frames, borders, or perspective changes still required preprocessing and verification

The lesson was that style-aware embeddings are useful, but identity recognition needs an instance-retrieval model and a second verification stage.

## 4. Current Model Stack

### Identity Embedding Model: DINOv2

The current main identity model is:

```text
facebook/dinov2-base
```

DINOv2 is used to extract image embeddings for painting identity retrieval. Each embedding is L2-normalized before being stored or queried.

Why DINOv2 helped:

- stronger visual representation for image retrieval
- less dependent on text semantics than CLIP
- better suited for instance retrieval and visual structure
- works well with FAISS cosine-like search

The system can later test:

```text
facebook/dinov2-large
```

but the current practical model is `facebook/dinov2-base`.

### Vector Search: FAISS

The vector database uses FAISS:

```text
IndexFlatIP
```

Embeddings are normalized, so inner product behaves like cosine similarity.

Saved artifacts:

```text
data/faiss_index.idx
data/embeddings.npy
data/index_mapping.json
```

### Multiple Embeddings Per Painting

Each indexed painting contributes multiple embeddings. The first variant is always the original image. Additional variants use small augmentations such as:

- small border crop
- brightness change
- contrast change
- blur
- perspective jitter
- JPEG compression

This helps the model recognize uploads that are not pixel-identical to the dataset image.

Current build:

```text
4 augmentations per painting
4709 paintings
18836 embeddings
```

### Match Aggregation

Because every painting has multiple augmented embeddings, FAISS returns embedding-level hits, not painting-level hits. The system groups hits by `painting_id` and aggregates them. This avoids treating each augmentation as a separate painting.

The aggregated score gives strongest weight to the best hit and adds a small contribution from the average of the best hits for the same painting.

### Geometric Verification

Embedding search alone is not enough. DINOv2 may retrieve paintings that are visually similar but not the same. The second confirmation stage checks whether the query and candidate image share local geometric structure.

The intended stack is:

- LightGlue + SuperPoint when available
- ORB homography fallback locally
- LoFTR as a recommended future alternative

The current local fallback is ORB homography. It estimates feature matches and counts homography inliers.

Current threshold:

```text
geometric inlier threshold: 35
```

A painting is recognized only if:

```python
embedding_score >= 0.82
geometric_inliers >= 35
```

or if the exact-image perceptual hash fallback succeeds.

### Perceptual Hash Fallback

A perceptual hash stage was added before DINOv2 retrieval. This helps exact or near-exact copies of indexed files be recognized quickly.

Current threshold:

```text
perceptual hash distance threshold: 8
```

Why this was useful:

- fast exact-image recognition
- source-neutral: works for Armenian and WikiArt
- avoids unnecessary DINOv2/geometric work for identical test images

Important limitation:

- perceptual hashing is not a replacement for recognition from real camera photos
- it works best when the uploaded image is close to an indexed file

## 5. Cropping and Upload Preprocessing

The project originally attempted to crop user uploads before recognition. This helped for framed or wall images but created a new problem: when the user uploaded an already-clean dataset image, the cropper sometimes damaged the image and caused recognition failure.

The current solution is to try multiple query variants:

- original full image
- rotated versions: 0, 90, 180, and 270 degrees
- small border crops: 3% and 5%
- cropper result
- cropper result with 3% and 8% border removal

This solved two important problems:

1. Some uploaded images had phone EXIF rotation or appeared rotated in Streamlit.
2. Some dataset images already included useful frame/wall context, so full-image matching had to remain available.

### YOLO Cropper Plan

The code supports a YOLO segmentation cropper through Ultralytics:

```text
YOLO11-seg or YOLOv8-seg
```

Expected weight path:

```text
data/models/painting_yolo_seg.pt
```

If the YOLO model is missing, the system falls back to the older OpenCV contour-based gallery detector.

### Synthetic YOLO Data

The project can generate synthetic cropper training data by placing paintings on artificial walls and adding:

- frames
- perspective distortion
- brightness/contrast changes
- blur
- JPEG compression

The generated labels are YOLO segmentation polygons for the true painting area.

Command:

```bash
python main.py generate-yolo --samples-per-image 8 --include-wikiart --wikiart-limit 4500
```

This is designed to teach YOLO to answer a narrow question:

> Where is the painting inside this uploaded photo?

This is separate from identity recognition.

## 6. Style and Genre Prediction

Style and genre are treated separately from identity recognition. This is important because a system can fail to identify the exact artwork but still give useful visual analysis.

Current style/genre artifact:

```text
data/style_genre_classifier.pkl
```

Training command:

```bash
python main.py train-style-genre --source wikiart
```

Training result:

```text
style examples: 18000
genre examples: 7484
style classes: 27
genre classes: 9
```

### Style Model

The style classifier uses DINOv2 embeddings and WikiArt folder labels. The classifier is a scikit-learn pipeline:

```text
StandardScaler + LogisticRegression
```

It predicts a style label and probability.

If a recognized WikiArt painting already has a metadata style, the app uses the metadata style with confidence 1.0. If style is missing, or if the painting is not recognized, it predicts the style from the query embedding.

### Genre Model

The genre model also uses DINOv2 embeddings, but the labels are weaker. WikiArt does not always provide clean genre labels, so the project infers genre labels using metadata and title/filename keywords.

Example keyword-derived genres:

- portrait
- landscape
- seascape
- cityscape
- still life
- religious
- mythological
- interior
- abstract

Because these labels are heuristic, genre confidence should be interpreted carefully.

### Why Style/Genre Prediction Matters

When an uploaded painting is outside the index, the identity system correctly returns `not_found`. However, the style/genre classifier can still provide useful information.

Example tested behavior:

```text
query: Botticelli Madonna of the Sea
identity result: not_found
predicted style: Early Renaissance
style confidence: 0.97
predicted genre: Religious
genre confidence: 0.99
```

This is desirable because the system avoids falsely claiming identity while still offering art-historical interpretation.

## 7. Current Recognition Decision Logic

The final decision is conservative.

For normal embedding/geometric recognition:

```python
if embedding_score >= 0.82 and geometric_inliers >= 35:
    recognized
else:
    not_found
```

For exact or near-exact indexed images:

```python
if perceptual_hash_distance <= 8:
    recognized
```

This means the app may show a visually similar candidate but still return `not_found` if geometric verification fails. This is intentional. It reduces false positives.

## 8. Logs and Saved Artifacts

### Build Report

Path:

```text
data/build_report.json
```

Purpose:

- records the current index configuration
- stores model name, number of paintings, number of embeddings, and whether WikiArt was included

Current values:

```text
identity_model: facebook/dinov2-base
total_paintings: 4709
armenian_paintings: 209
wikiart_paintings: 4500
total_embeddings: 18836
augmentations_per_painting: 4
```

### Index Mapping

Path:

```text
data/index_mapping.json
```

Purpose:

- maps every FAISS embedding row back to painting metadata
- includes source, filename, painter, title, year, style, genre, image path, augmentation index, and embedding model

### Embeddings

Path:

```text
data/embeddings.npy
```

Purpose:

- stores the DINOv2 embedding matrix used for FAISS search and style/genre training

### FAISS Index

Path:

```text
data/faiss_index.idx
```

Purpose:

- stores the searchable vector index

### Index Manifest

Path:

```text
data/index_manifest.csv
```

Purpose:

- one row per unique indexed painting
- useful for checking which paintings are currently searchable

### Processed WikiArt Manifest

Path:

```text
data/processed_images/wikiart_processed_manifest.csv
```

Purpose:

- maps raw WikiArt source images to the selected visible dataset and processed copy
- confirms that `data/datasets/wikiart` and `data/processed_images/wikiart` contain the same selected 4500 images

### Recognition Debug Log

Path:

```text
data/recognition_debug_log.jsonl
```

Purpose:

- logs failed or unrecognized Streamlit queries
- stores top candidates, scores, query variant, threshold, and near-threshold status

This is useful for error analysis. For example, many `not_found` entries still have high visual similarity to religious paintings, portraits, landscapes, or abstract works, but geometric verification prevents the system from claiming a false identity.

### Streamlit Error Log

Path:

```text
data/last_streamlit_error.log
```

Purpose:

- stores the latest Streamlit exception traceback
- useful when the web interface shows `Analysis failed`

### Evaluation Reports

Paths:

```text
data/evaluation_report.json
data/evaluation_report_recomputed.json
data/evaluation_report_debug.json
```

Important caveat:

Some older evaluation reports were produced during the CLIP-based stage and tested clean indexed images. These results are useful historically but should not be presented as final real-world accuracy.

Reported older CLIP-style numbers included:

```text
top-1 recognition accuracy: about 0.995 to 1.0
top-5 recognition accuracy: 1.0
style classifier accuracy: about 0.53 to 0.57
```

These numbers are high because clean indexed images are easy to retrieve. The project documentation should explain that realistic validation requires framed photos, rotated photos, phone photos, and unknown paintings outside the index.

## 9. Why Some Things Worked

### DINOv2 Worked Better Than CLIP for Identity

DINOv2 helped because exact painting identity is closer to visual instance retrieval than text-image semantic matching. CLIP can understand that two images are both religious Renaissance paintings, but that does not mean they are the same painting. DINOv2 provided stronger visual features for image-to-image retrieval.

### Augmentations Helped

Multiple embeddings per painting helped because user uploads are not always identical to dataset images. Small crops, brightness changes, blur, compression, and perspective changes make the index more tolerant.

### Full-Image Query Variants Helped

At one point, cropping first caused exact dataset images to fail. The system now tries the original full image before relying on cropper variants. This fixed cases where the cropper removed useful content or changed the image too much.

### Rotation Variants Helped

Some uploaded images appeared rotated. Querying 0, 90, 180, and 270 degree variants helped handle phone orientation and Streamlit/PIL/OpenCV orientation differences.

### Geometric Verification Helped Reduce False Positives

DINOv2 sometimes retrieves visually similar paintings. Geometric verification checks whether the candidate has matching local structure. This prevents the app from claiming a match just because the subject and style are similar.

### Perceptual Hash Helped for Exact Test Images

When the user uploads the exact indexed file, perceptual hash can recognize it quickly. This is useful for controlled testing from `data/datasets/wikiart` or `data/test_samples/indexed`.

## 10. Why Some Things Did Not Work

### Clean-Image Evaluation Was Misleading

Testing with the exact same clean image that exists in the index gives almost perfect accuracy. This does not prove the system works on real user uploads. It only proves the vector index can retrieve an identical file.

### CLIP Was Too Semantic for Exact Identity

CLIP often recognizes high-level meaning but can confuse different paintings with similar subjects. For example, several Madonna and Child paintings may be close in CLIP space even though they are different artworks.

### Style Models Do Not Solve Identity

A model trained on style labels can identify a movement such as Impressionism or Baroque, but it cannot reliably identify the exact painting. This is why style recognition and identity recognition are separate.

### Cropping Can Help or Hurt

Cropping is necessary for wall/frame photos, but it can hurt if the image is already clean or if the cropper chooses the wrong region. The current multi-variant query strategy is a practical compromise.

### ORB Is Only a Fallback

ORB homography is local and available without large dependencies, but it is not always robust for low-texture, blurry, repetitive, or painterly images. LightGlue/SuperPoint or LoFTR should be tested more deeply for future improvement.

## 11. Current Performance Interpretation

The current system performs well for:

- exact images from the indexed Armenian and WikiArt sets
- many framed Armenian images
- uploads where the artwork is close to an indexed image
- images where geometric verification finds enough local structure
- style/genre analysis even when identity is unknown

The system is intentionally conservative. It may return `not_found` even when the top visual candidate looks plausible, because it requires both high embedding similarity and enough geometric evidence.

This is a good design choice for a capstone project because false identification is more serious than saying “not found.”

Current observed behavior:

- Armenian indexed paintings work, including framed cases.
- WikiArt indexed paintings work when selected from `data/datasets/wikiart`.
- WikiArt paintings from `data/datasets/wikiart_raw` may not be recognized if they are not part of the selected 4500-image index.
- Unknown paintings still receive style and genre predictions.

## 12. Evaluation Plan for the Paper

The final paper should not rely only on old clean-image retrieval accuracy. A stronger evaluation section should define three test sets:

### 1. Indexed Clean Images

Purpose:

- confirms that the FAISS index and mapping are correct
- tests exact retrieval

Limitation:

- not realistic
- should not be treated as real-world recognition accuracy

### 2. Synthetic Wall/Frame Images

Purpose:

- tests robustness to frame, wall, crop, compression, blur, perspective, and lighting changes
- can be generated automatically from indexed paintings

Useful metrics:

- top-1 accuracy before verification
- top-5 accuracy before verification
- final recognition accuracy after verification
- false rejection rate
- cropper success rate

### 3. Unknown Paintings

Purpose:

- tests whether the system correctly returns `not_found`
- can use WikiArt images outside the selected index

Useful metrics:

- false positive rate
- near-threshold count
- style/genre prediction quality

### Recommended Metrics

For identity:

- top-1 retrieval accuracy
- top-5 retrieval accuracy
- recognition precision
- recognition recall
- F1 score
- false positive rate on unknown images
- false negative rate on known images
- mean reciprocal rank

For geometric verification:

- inlier distribution for true matches
- inlier distribution for false matches
- threshold sensitivity

For style:

- style accuracy
- macro F1
- confusion matrix

For genre:

- genre accuracy if labeled validation data exists
- otherwise report it as heuristic/pseudo-labeled

## 13. Commands Used in the Current System

Build Armenian + WikiArt identity index:

```bash
python main.py build-index --embedding-model facebook/dinov2-base --augmentations 4 --include-wikiart --wikiart-limit 4500 --progress-interval 100
```

Train style and genre:

```bash
python main.py train-style-genre --source wikiart
```

Sync visible selected WikiArt dataset with processed images:

```bash
python main.py sync-processed --source wikiart --clean --dataset-copy data/datasets/wikiart --manifest data/processed_images/wikiart_processed_manifest.csv
```

Run Streamlit:

```bash
python -m streamlit run frontend_app.py
```

Query one image:

```bash
python main.py query path/to/image.jpg --embedding-model facebook/dinov2-base --top-k 20
```

## 14. Main Files and Their Roles

### `frontend_app.py`

Streamlit web application. It lets the user upload an image and displays recognition, style, genre, confidence, similar works, and curator-style chat responses.

### `main.py`

Command dispatcher. It runs project scripts such as build, query, train-style-genre, generate-yolo, train-cropper, sync-processed, and export-manifest.

### `art_recognition/pipeline.py`

Connects the full recognition flow:

- load records
- build index
- generate query variants
- perceptual hash matching
- DINOv2 embedding extraction
- FAISS search
- aggregation
- geometric verification
- style/genre prediction
- final response construction

### `art_recognition/identity.py`

Contains DINOv2 embedding extraction, normalization, augmentations, aggregation, perceptual hashing, and geometric verification.

### `art_recognition/cropping.py`

Contains the painting cropper. It uses YOLO segmentation if trained weights exist, otherwise it falls back to OpenCV contour detection.

### `art_recognition/synthetic_yolo.py`

Generates synthetic wall/frame images and YOLO segmentation labels.

### `art_recognition/style_genre.py`

Trains and runs DINOv2-based style and genre classifiers.

### `art_recognition/datasets.py`

Loads Armenian and WikiArt records and converts filenames/folder structure into metadata.

### `art_recognition/database.py`

Builds and loads the FAISS vector database.

### `scripts/build_index.py`

Builds the DINOv2 identity index.

### `scripts/query_index.py`

Runs one command-line query.

### `scripts/train_style_genre.py`

Trains style and genre classifiers from indexed embeddings.

### `scripts/sync_processed_images.py`

Creates matching visible copies of selected WikiArt paintings in `data/datasets/wikiart` and `data/processed_images/wikiart`.

## 15. How to Explain the Final System in the Paper

A concise description:

> The final system uses a multi-stage recognition pipeline. First, it prepares several normalized versions of the uploaded image to handle rotation, borders, and possible frame/wall context. It then extracts DINOv2 visual embeddings and searches a normalized FAISS inner-product index built from Armenian and selected WikiArt paintings. Each painting is represented by several augmented embeddings, and results are grouped by original painting identity. Candidate matches are accepted only when they pass both embedding similarity and geometric verification. If the exact painting is not recognized, the system still predicts style and genre using DINOv2-based classifiers trained on WikiArt labels.

## 16. Honest Limitations

- The current YOLO cropper support exists in code, but final crop quality depends on having trained YOLO weights.
- The current geometric fallback uses ORB, which is not ideal for every painting.
- The genre classifier uses pseudo-labels from titles/filenames when explicit labels are missing.
- Old evaluation reports are useful historically but should not be presented as final real-world accuracy.
- The selected WikiArt identity index contains 4500 paintings, not the entire raw WikiArt dataset.
- A painting in `wikiart_raw` is not recognizable unless it is also in the current identity index.
- Real phone-photo validation is still needed for a scientifically strong evaluation.

## 17. Best Future Improvements

1. Train and validate YOLO11-seg or YOLOv8-seg on synthetic and real framed painting photos.
2. Add a labeled validation set of real phone photos.
3. Evaluate unknown paintings outside the index to measure false positives.
4. Replace or supplement ORB with LightGlue/SuperPoint and LoFTR.
5. Calibrate DINOv2 and geometric thresholds using validation distributions.
6. Add precision, recall, F1, and mean reciprocal rank reports.
7. Expand the WikiArt identity index if more coverage is needed.
8. Improve genre labels with a real labeled genre dataset.
9. Save query preview/crop images for easier debugging.
10. Add a user-facing explanation of why a result was accepted or rejected.

## 18. Suggested Paper Structure

1. Introduction
2. Problem Statement
3. Dataset Description
4. Initial Baselines: ResNet50 and CLIP
5. Limitations of Baselines
6. Final Architecture
7. Image Preprocessing and Crop Handling
8. DINOv2 Embedding Retrieval
9. FAISS Index and Augmented Embeddings
10. Geometric Verification
11. Style and Genre Prediction
12. Web Application
13. Evaluation Methodology
14. Results and Discussion
15. Limitations
16. Future Work
17. Conclusion

## 19. Key Takeaway

The most important lesson from this project is that painting recognition should not be treated as a single image-classification task. A practical system needs separate stages for image preparation, visual retrieval, identity verification, and fallback interpretation. DINOv2 improved retrieval, augmentations improved robustness, geometric verification reduced false positives, and the style/genre classifier kept the application useful even when exact identity recognition failed.
