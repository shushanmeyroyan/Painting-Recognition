# Painting Recognition Project Documentation

## 1. Project Purpose

This project is a painting recognition and painting analysis system. Its goal is to let a user upload a painting image and receive the best available information about it:

- whether the exact painting is recognized
- title, painter, and year when the match is strong
- likely artistic style
- likely genre, when enough visual or metadata evidence exists
- related visually similar paintings
- a possible painter only when the evidence is strong enough

The system works like a small "Shazam for paintings". It converts paintings into numerical image embeddings, compares those embeddings with saved artwork embeddings, and uses a trained style classifier to describe the uploaded painting.

## 2. Main Workflow

1. Local Armenian paintings are loaded from `data/datasets/armenian/images`.
2. Armenian metadata is loaded from `data/datasets/armenian/metadata/Book1.xlsx`.
3. WikiArt paintings are loaded from `data/datasets/wikiart_raw` or downloaded with KaggleHub.
4. Each image is preprocessed.
5. A pretrained image model creates an embedding vector for every painting.
6. Embeddings are saved in `data/embeddings.npy`.
7. A FAISS similarity index is saved in `data/faiss_index.idx`.
8. Metadata for every indexed painting is saved in `data/index_mapping.json`.
9. A style classifier is trained using WikiArt style labels and saved in `data/style_classifier.pkl`.
10. When a user uploads a painting, the same preprocessing and embedding steps are applied.
11. The uploaded image is prepared with the same model preprocessing path.
12. The app compares a prepared full-image version and, when gallery detection finds a framed painting, a detected crop.
13. The uploaded embedding is compared with indexed embeddings.
14. The app returns a recognized painting only if the similarity score is high enough. Otherwise it gives fallback analysis: style, genre, related works, and possible painter if confidence is high.

## 3. Folder Structure

```text
.
├── art_recognition/
├── data/
├── docs/
├── preprocessing_output/
├── scripts/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── frontend_app.py
├── main.py
├── requirements.txt
└── Untitled-1.ipynb
```

## 4. Important Folders

### `art_recognition/`

This is the main Python package. It contains the core logic for loading data, preprocessing images, extracting embeddings, training style classifiers, building the index, and querying uploaded paintings.

### `scripts/`

Command-line scripts for building the recognition index and querying it.

### `data/`

Stores datasets and generated model artifacts:

- `data/datasets/armenian/`: Armenian painting images and Excel metadata.
- `data/datasets/wikiart_raw/`: WikiArt painting images grouped by style.
- `data/processed_images/`: preprocessed image copies used during indexing.
- `data/faiss_index.idx`: FAISS similarity-search index.
- `data/index_mapping.json`: metadata mapping for indexed embeddings.
- `data/embeddings.npy`: saved embedding matrix.
- `data/style_classifier.pkl`: trained style classifier.
- `data/armenian_style_predictions.csv`: predicted styles for Armenian paintings.
- `data/build_report.json`: summary of the last index build.
- `data/art_database.db`: SQLite database artifact, currently not the main query path.
- `data/user_uploads/`: temporary user-uploaded images from the web app.

### `docs/`

Contains the original README documentation.

### `tests/`

Contains tests for the preprocessing pipeline.

### `preprocessing_output/`

Stores debug images from preprocessing, such as segmentation masks and annotated candidates.

## 5. File-by-File Explanation

### `main.py`

Small command dispatcher. It supports:

- `python main.py build-index`
- `python main.py query <image_path>`

It forwards those commands to scripts in the `scripts/` folder.

### `frontend_app.py`

The Streamlit web application. It is now designed for end users, not developers.

Current user-facing flow:

- Renaissance/Baroque visual style with Medici red background.
- Short explanation of what the website does.
- Upload area for a painting image.
- Result area showing recognized artwork or fallback analysis.
- Style, genre, and confidence fields.
- A visible local curator-style chat panel with a text field and Send button.
- Related paintings shown in a visual, readable way.

The public page hides developer details such as database size, build reports, model selectors, and internal project progress.

### `requirements.txt`

Lists Python dependencies:

- `opencv-python`
- `numpy`
- `matplotlib`
- `torch`
- `torchvision`
- `faiss-cpu`
- `pillow`
- `scikit-learn`
- `transformers`
- `requests`
- `pandas`
- `openpyxl`
- `kagglehub[pandas-datasets]`
- `streamlit`

### `Dockerfile`

Builds a Python 3.11 container with system libraries needed by OpenCV, PyTorch, FAISS, and the rest of the project.

### `docker-compose.yml`

Defines a service named `art-recognition`. It mounts the project and `data/` folder into Docker and includes Kaggle/Hugging Face cache configuration.

### `Untitled-1.ipynb`

Notebook file. It is not part of the production command-line or website flow.

## 6. Core Package Files

### `art_recognition/config.py`

Defines `ProjectPaths`, a dataclass that centralizes important project paths.

Important properties:

- `data_dir`
- `datasets_dir`
- `armenian_images_dir`
- `armenian_metadata_path`
- `wikiart_raw_dir`
- `processed_dir`
- `faiss_index_path`
- `mapping_path`
- `embeddings_path`
- `classifier_path`
- `build_report_path`

### `art_recognition/datasets.py`

Loads Armenian and WikiArt painting records.

Important class:

- `ArtworkRecord`: stores source, image path, filename, title, artist, year, style, and genre.

Important functions:

- `load_armenian_records()`: reads Armenian metadata and connects rows to image files.
- `load_wikiart_records()`: loads or downloads WikiArt records.
- `_read_excel_with_zip_fallback()`: reads Excel metadata even if normal Excel support is unavailable.
- `_normalize_columns()`: standardizes metadata column names.
- `_discover_local_images()`: finds local image files.
- `_build_image_lookup()`: creates lookup keys for matching filenames.
- `_infer_wikiart_metadata_from_path()`: extracts artist/title/style from WikiArt folder and filename structure.
- `_sample_diverse_artists()`: samples WikiArt records while avoiding over-representing one artist.

### `art_recognition/preprocessing.py`

Handles image preprocessing and gallery-image painting detection.

Important class:

- `PaintingCandidate`: stores bounding box, contour data, corners, crop, and mask for a detected painting candidate.

Important functions:

- `mean_shift_segmentation()`: smooths image regions.
- `get_mask_of_largest_segment()`: finds the largest segment, usually the wall/background.
- `dilate_image()`, `erode_image()`, `invert_image()`, `median_filter()`: image morphology helpers.
- `canny_edge_detection()`: detects edges.
- `get_possible_painting_contours()`: filters contours that could be paintings.
- `order_corners()`: orders four detected corners.
- `_extract_painting_shape()`: tries to detect painting boundaries.
- `preprocess_gallery_image()`: full pipeline for detecting paintings inside a photo.
- `draw_preprocessing_result()`: draws rectangles around detected painting candidates.
- `load_image()`: reads image from disk.
- `save_debug_images()`: saves intermediate preprocessing outputs.

### `art_recognition/ml_models.py`

Contains the machine learning model wrappers.

Important classes:

- `EmbeddingExtractor`
- `StyleClassifier`
- `ClipZeroShotStylePredictor`

`EmbeddingExtractor` supports:

- `resnet50`
- `clip`

For `resnet50`, it uses `torchvision.models.resnet50` with default pretrained weights and removes the final classification layer. The embedding dimension is 2048.

For `clip`, it uses Hugging Face `openai/clip-vit-base-patch32`. The embedding dimension is the CLIP projection dimension.

Important functions/methods:

- `EmbeddingExtractor.extract()`: converts an RGB image into a normalized embedding vector.
- `EmbeddingExtractor.extract_texts()`: converts CLIP text prompts into normalized text embeddings.
- `StyleClassifier.fit()`: trains the non-linear style classifier.
- `StyleClassifier.predict()`: returns predicted style and probability.
- `StyleClassifier.save()`: saves the classifier with pickle.
- `StyleClassifier.load()`: loads the classifier from disk.
- `predict_style_with_fallback()`: uses XGBoost when confident and CLIP zero-shot prompts as fallback/complement.

The style classifier uses:

- `XGBoost` `XGBClassifier` as the preferred backend
- scikit-learn `HistGradientBoostingClassifier` as a fallback if XGBoost is not installed
- CLIP zero-shot prompts for fallback style prediction

### `art_recognition/database.py`

Handles vector storage and similarity search.

Important class:

- `ArtVectorDatabase`

Important functions/methods:

- `build()`: normalizes embeddings, builds a FAISS index, saves embeddings and metadata.
- `load()`: loads the FAISS index and metadata mapping.
- `load_embeddings()`: loads the saved embedding matrix.
- `search()`: searches with FAISS.
- `export_matches()`: returns search results as dictionaries.
- `export_matches_with_numpy()`: computes similarity with NumPy matrix multiplication.

Similarity method:

- Embeddings are L2-normalized.
- FAISS uses `IndexFlatIP`, which means inner product.
- Because vectors are normalized, inner product behaves like cosine similarity.

### `art_recognition/pipeline.py`

Connects the whole system.

Important functions:

- `crop_border()`: removes a small border from images.
- `preprocess_painting_image()`: crops border and converts BGR to RGB.
- `preprocess_query_image()`: detects the largest painting candidate and prepares it.
- `preprocess_query_image_variants()`: prepares multiple query versions, including the full image and detected crop.
- `_fit_style_classifier()`: trains the style classifier using WikiArt labels.
- `infer_genre_from_matches()`: estimates genre from explicit genre metadata or title/filename keywords.
- `infer_artist_from_matches()`: estimates painter only when similar works agree strongly.
- `build_query_response()`: creates the final response with recognition/fallback logic.

Important class:

- `ArtRecognitionPipeline`

Important methods:

- `build_index()`: builds embeddings, FAISS index, mapping file, style classifier, and build report.
- `_save_armenian_style_predictions()`: stores predicted styles for Armenian paintings.
- `query()`: analyzes a new image and returns JSON-style results.

Current recognition threshold:

- `RECOGNITION_SCORE_THRESHOLD = 0.96`

If the best match score is below this threshold, the system does not claim the exact painting is recognized.

Painter fallback thresholds:

- `ARTIST_SCORE_THRESHOLD = 0.55`
- `ARTIST_CONFIDENCE_THRESHOLD = 0.72`

This means painter attribution is intentionally conservative.

## 7. Scripts

### `scripts/build_index.py`

Builds the full project index.

Example:

```bash
python main.py build-index --wikiart-limit 4500 --embedding-model clip
```

Important arguments:

- `--project-root`
- `--wikiart-limit`
- `--wikiart-metadata-path`
- `--embedding-model`
- `--skip-wikiart`

### `scripts/query_index.py`

Queries the built index with one image.

Example:

```bash
python main.py query path/to/image.jpg --embedding-model clip --top-k 5
```

Returns JSON with:

- `is_recognized`
- `recognition_status`
- `recognition_score`
- `recognized_painting`
- `artist`
- `year`
- `predicted_style`
- `predicted_style_confidence`
- `predicted_style_source`
- `inferred_genre`
- `possible_artist`
- `similar_paintings`

### `scripts/evaluate_models.py`

Evaluates the current artifacts and saves a JSON report.

Example:

```bash
python main.py evaluate --max-query 500 --output data/evaluation_report.json
```

The report includes:

- top-1 and top-5 recognition accuracy for known indexed artworks
- split-based style classifier accuracy
- style confusion matrix
- CLIP zero-shot style accuracy
- positive and negative similarity score distributions
- a suggested threshold based on score separation

## 8. Models Used

### CLIP

Default image embedding extractor.

Source:

- Hugging Face model `openai/clip-vit-base-patch32`

How it is used:

- The CLIP image encoder creates a 512-dimensional feature vector.
- The vector is L2-normalized before saving and querying.
- The same CLIP model also creates text embeddings for zero-shot style prompts such as `a painting in the style of impressionism`.
- CLIP features are used for FAISS similarity search and style classification.

Current project status:

- The existing saved index was rebuilt with `clip`.
- This is the default model used by the CLI and website.

### ResNet50

Optional legacy image embedding extractor.

Source:

- `torchvision.models.resnet50`
- default pretrained ImageNet weights

How it is used:

- The final classification layer is replaced with `torch.nn.Identity()`.
- The model outputs a 2048-dimensional feature vector.
- The vector is normalized.
- It can still be used by passing `--embedding-model resnet50`, but the index must be rebuilt with the same model before querying.

Current project status:

- ResNet50 is supported for comparison or legacy experiments.
- It is not the current default.

### Style Classifier

The style classifier is trained on embeddings and WikiArt style labels.

Model:

- `XGBoost` `XGBClassifier`
- fallback to scikit-learn `HistGradientBoostingClassifier` if XGBoost is unavailable

Output:

- predicted style
- predicted style probability/confidence
- style prediction source, such as `classifier` or `clip_zero_shot`

The classifier is saved to:

```text
data/style_classifier.pkl
```

## 9. Current Artifacts

The current `data/build_report.json` reports:

- total indexed records: 4709
- Armenian records: 209
- WikiArt records: 4500
- style classes: 27
- style classifier trained: yes
- style classifier backend: XGBoost
- embedding model in mapping: CLIP

These numbers describe the current local build, not a user-facing website feature.

## 10. How Good The Models Are

The project now saves an evaluation report in `data/evaluation_report.json`.

What exists now:

- embedding similarity score for painting recognition
- conservative exact-match threshold
- XGBoost style classifier probability
- CLIP zero-shot style fallback
- possible-artist confidence from agreement among top matches
- top-1 and top-5 recognition accuracy for known indexed artworks
- style confusion matrix
- positive and negative similarity score distributions
- preprocessing test for candidate painting detection

What was observed during testing after the CLIP rebuild:

- A known indexed image, `our-lady-cathedral-of-ani-190.jpg`, returned the correct painting with a score of 1.0.
- A recomputed evaluation run over 30 query samples reported top-1 recognition accuracy of 1.0 and top-5 recognition accuracy of 1.0.
- That diagnostic run had no threshold rejections at the current 0.96 threshold when the inference-style full-image variant was used correctly.
- Split-based XGBoost style accuracy in that compact run was about 0.567.
- CLIP zero-shot style accuracy in that compact run was about 0.333, so it is best used as a fallback/complement rather than the primary style model.

This is good behavior for safety because the system avoids overclaiming when the exact painting is not found.

## 11. Metrics Currently Used

### Similarity Score

Used for recognition.

The score is computed from normalized embeddings:

```text
similarity = embedding_database @ query_embedding
```

Because embeddings are normalized, this behaves like cosine similarity.

### Recognition Threshold

Used to decide whether to claim an exact painting match.

Current threshold:

```text
0.96
```

If the best score is below this value, the system says the exact painting was not found.

### Style Confidence

Used for style prediction.

This is the maximum predicted probability from the XGBoost style classifier, or the CLIP prompt probability when the zero-shot fallback is used.

### Possible Artist Confidence

Used only when the exact painting is not recognized.

The system looks at top similar paintings and checks whether one artist dominates the weighted similarity scores. It only returns a possible artist if the evidence passes conservative thresholds.

## 12. Evaluation Report

The project includes `scripts/evaluate_models.py`, available through:

```bash
python main.py evaluate --max-query 500 --output data/evaluation_report.json
```

The report currently saves:

- known-artwork top-1 recognition accuracy
- known-artwork top-5 recognition accuracy
- split-based XGBoost style accuracy
- style confusion matrix
- CLIP zero-shot style accuracy
- positive and negative similarity score distributions
- suggested recognition threshold from the measured distributions

Metrics that are still recommended for deeper scientific evaluation:

- precision, recall, and F1-score for each style
- mean reciprocal rank
- preprocessing detection precision/recall on labeled framed-photo examples
- threshold calibration using more real phone photos and gallery-frame photos

## 13. Tests

### `tests/test_preprocessing.py`

Creates a synthetic gallery scene with rectangular paintings and runs preprocessing.

It verifies that the preprocessing pipeline can run without crashing and save debug output.

Current test command:

```bash
python -m pytest tests/
```

Current verified result:

```text
1 passed
```

## 14. Website Behavior

The website is in:

```text
frontend_app.py
```

Run it with:

```bash
python -m streamlit run frontend_app.py
```

The website now:

- opens with a Medici red Renaissance/Baroque visual design
- explains what the website can do for users
- provides a simple upload area
- analyzes the uploaded painting
- shows exact match information only when confidence is high
- otherwise shows fallback analysis
- includes a small curator chat with a visible input field that can answer short questions about the painting, painter, style, genre, and confidence
- shows artwork information in readable title-style display text, for example `Contemporary Realism` instead of raw labels like `Contemporary_Realism`
- avoids developer-only sections such as database size, build status, and model controls

The curator chat is local and template-based. It does not call an external AI service. It uses the recognition result, predicted style, inferred genre, and top similar painting metadata to produce short explanations.

## 15. Limitations

- Exact recognition only works well for paintings similar to the indexed collection.
- Style prediction depends on WikiArt labels and the embedding model.
- Genre prediction is currently heuristic and uses metadata/title clues from similar paintings.
- Painter attribution is intentionally conservative.
- Formal accuracy metrics are not yet implemented.
- The project can be improved with a labeled validation set and systematic evaluation.

## 16. Best Next Improvements

Recommended next steps:

1. Add precision, recall, and F1-score to the evaluation report.
2. Add better genre labels or train a genre classifier.
3. Add a curated test set of framed phone photos.
4. Calibrate the recognition threshold on real uploaded-photo examples.
5. Add a user-friendly confidence explanation to the website.
6. Clean temporary uploads periodically.
