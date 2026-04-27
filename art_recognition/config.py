from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def armenian_dir(self) -> Path:
        return self.datasets_dir / "armenian"

    @property
    def armenian_images_dir(self) -> Path:
        return self.armenian_dir / "images"

    @property
    def armenian_metadata_path(self) -> Path:
        return self.armenian_dir / "metadata" / "Book1.xlsx"

    @property
    def wikiart_raw_dir(self) -> Path:
        return self.datasets_dir / "wikiart_raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed_images"

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "faiss_index.idx"

    @property
    def mapping_path(self) -> Path:
        return self.data_dir / "index_mapping.json"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.npy"

    @property
    def classifier_path(self) -> Path:
        return self.data_dir / "style_classifier.pkl"

    @property
    def build_report_path(self) -> Path:
        return self.data_dir / "build_report.json"
