# Painting-Recognition

Identity recognition now follows a crop-first pipeline:

1. Generate synthetic wall/frame/perspective training images for a YOLO segmentation cropper.
2. Train a YOLO11-seg or YOLOv8-seg cropper.
3. Build a DINOv2 FAISS identity index with augmented embeddings. Armenian paintings are included by default; WikiArt can be added with `--include-wikiart`.
4. Query uploads by respecting EXIF orientation, trying full-image/rotated/crop variants, embedding with DINOv2, aggregating top-k hits by painting, then confirming with geometric verification.

When WikiArt is included in the identity index, indexed WikiArt paintings are recognized the same way as indexed Armenian paintings.

Generate YOLO training data:

```bash
python main.py generate-yolo --samples-per-image 12
```

Generate YOLO data from Armenian + WikiArt:

```bash
python main.py generate-yolo --samples-per-image 8 --include-wikiart --wikiart-limit 4500
```

Train cropper:

```bash
python main.py train-cropper --model yolo11n-seg.pt
```

Copy the best trained weights to:

```text
data/models/painting_yolo_seg.pt
```

Build Armenian identity index:

```bash
python main.py build-index --embedding-model facebook/dinov2-base --augmentations 16
```

Build Armenian + WikiArt identity index:

```bash
python main.py build-index --embedding-model facebook/dinov2-base --augmentations 8 --include-wikiart --wikiart-limit 4500
```

Query:

```bash
python main.py query path/to/upload.jpg --embedding-model facebook/dinov2-base --top-k 20
```

See exactly which paintings are indexed and copy sample test images:

```bash
python main.py export-manifest --samples-per-source 20
```

This writes:

```text
data/index_manifest.csv
data/test_samples/indexed/
```

Sync the current indexed WikiArt paintings into `processed_images`:

```bash
python main.py sync-processed --source wikiart --clean --dataset-copy data/datasets/wikiart --manifest data/processed_images/wikiart_processed_manifest.csv
```

This makes `data/datasets/wikiart/` and `data/processed_images/wikiart/` contain the same indexed 4500 WikiArt files, preserving style folders so files with the same filename from different styles do not overwrite each other.

You do not need to rebuild embeddings every time. Rebuild only when you change the indexed dataset, model name, augmentation count, or source mix.
