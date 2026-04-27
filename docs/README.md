# Art Recognition System

This capstone project implements a full "Shazam for paintings" pipeline:

- preprocess paintings by simulating crop-based frame removal
- load Armenian paintings from the project root and metadata from `Book1.xlsx`
- download and sample 4,000-5,000 WikiArt images with `kagglehub`
- extract image embeddings with CLIP or ResNet50
- build a FAISS similarity index
- train a style classifier on WikiArt embeddings
- query a new image and return the recognized painting, artist, year, predicted style, and top matches
- reuse the same embedding space so WikiArt style labels can predict Armenian painting styles with confidence scores

## Clean Folder Layout

The local Armenian dataset is now organized like this:

```text
data/
├── datasets/
│   ├── armenian/
│   │   ├── images/
│   │   └── metadata/Book1.xlsx
│   └── wikiart_raw/
├── processed_images/
├── faiss_index.idx
├── index_mapping.json
├── embeddings.npy
└── style_classifier.pkl
```

## Project Structure

```text
art_recognition/
├── database.py       # FAISS index persistence and similarity search
├── datasets.py       # Armenian + WikiArt dataset loaders
├── ml_models.py      # Embedding extractor and style classifier
├── pipeline.py       # End-to-end build/query pipeline
└── preprocessing.py  # Gallery-image painting detection already present

scripts/
├── build_index.py    # Build the full index
└── query_index.py    # Query a new image
```

## Setup

```bash
pip install -r requirements.txt
```

If you use WikiArt, make sure Kaggle credentials are configured for `kagglehub`.

## Docker

Docker is the cleanest way to keep this project reproducible because it bundles:

- Python and system dependencies for `opencv`, `faiss`, and `torch`
- a mounted `data/` directory so generated files stay on your machine
- mounted Kaggle credentials for WikiArt downloads
- a persistent Hugging Face cache for model weights

Build the image:

```bash
docker compose build
```

Run an Armenian-only build:

```bash
docker compose run --rm art-recognition python main.py build-index --skip-wikiart --embedding-model clip
```

Run the full WikiArt build:

```bash
docker compose run --rm art-recognition python main.py build-index --wikiart-limit 4500 --embedding-model clip
```

Query an image:

```bash
docker compose run --rm art-recognition python main.py query path/to/query_image.jpg --top-k 3 --embedding-model clip
```

Notes:

- Kaggle credentials should exist on your machine at `~/.kaggle/kaggle.json`
- the first WikiArt run is large because `steubk/wikiart` is about 31.4 GB
- the first model run may also download CLIP weights into the mounted Hugging Face cache
- if you do not want Docker, the local `.venv` workflow still works

## Build The Index

```bash
python main.py build-index --wikiart-limit 4500 --embedding-model clip
```

Useful options:

- `--wikiart-limit 4500` keeps the WikiArt subset in the requested 4,000-5,000 range
- `--wikiart-metadata-path <file>` uses a specific metadata file inside the Kaggle dataset if one exists
- `--skip-wikiart` builds an Armenian-only index
- `--embedding-model clip` or `--embedding-model resnet50`

Build outputs are saved into `data/`:

- `faiss_index.idx`
- `index_mapping.json`
- `embeddings.npy`
- `style_classifier.pkl` when style training succeeds
- `armenian_style_predictions.csv` when WikiArt-based style prediction is available
- `processed_images/`

## Query An Image

```bash
python main.py query path/to/query_image.jpg --top-k 3 --embedding-model clip
```

The query response is returned as JSON and includes:

- recognized painting only when the best embedding score is strong enough
- artist and year for recognized paintings
- possible artist only when similar-painting evidence is high confidence
- predicted style
- predicted style confidence
- inferred genre when enough title or metadata cues are available
- top similar paintings with metadata and similarity score

## Frontend

A Streamlit frontend is available in [frontend_app.py](/Users/shushann/Desktop/Capstone%20Project/frontend_app.py:1).

Run it with:

```bash
streamlit run frontend_app.py
```

The frontend includes:

- an intro page with a Birth of Venus background
- a project-progress page showing dataset and artifact status
- a mobile-friendly upload page for image analysis and prediction

Upload handling covers:

- invalid non-image uploads
- images that are too small
- images that are too large
- no detected painting candidate
- multiple detected painting candidates, where the largest one is selected automatically
- exact painting not found by embedding, where the app falls back to style, genre, and possible painter evidence

## Style Classification Flow

The style component uses the same image embeddings as the recognition component:

1. WikiArt paintings are embedded with the same model as the Armenian paintings.
2. WikiArt style labels are used to train a classifier on top of those embeddings.
3. The trained classifier predicts a style for each Armenian painting and saves:
   - the predicted style
   - the confidence score
4. When a user uploads a new painting image, the same classifier predicts the uploaded image style too.

That means the product uses one shared visual pipeline:

- embedding model for image similarity
- WikiArt labels for learning artistic style
- Armenian style prediction with confidence
- uploaded image style prediction at query time

## Armenian Metadata Notes

The loader normalizes `Book1.xlsx` column names automatically. Your current workbook uses headers equivalent to:

- `filename`
- `title`
- `painter`
- `year`

## Testing

```bash
python -m pytest tests/
```
