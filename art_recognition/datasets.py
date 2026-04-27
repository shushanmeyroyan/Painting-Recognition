from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd

from art_recognition.config import ProjectPaths


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "art_recognition",
    "data",
    "docs",
    "preprocessing_output",
    "scripts",
    "tests",
}


@dataclass
class ArtworkRecord:
    source: str
    image_path: str
    filename: str
    title: str | None = None
    artist: str | None = None
    year: str | None = None
    style: str | None = None
    genre: str | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "source": self.source,
            "image_path": self.image_path,
            "filename": self.filename,
            "title": self.title,
            "artist": self.artist,
            "year": self.year,
            "style": self.style,
            "genre": self.genre,
        }


def _canonicalize_name(value: str | None) -> str:
    if value is None:
        return ""

    cleaned = str(value).strip().lower().replace("\\", "/")
    last_part = cleaned.split("/")[-1]

    previous = None
    while previous != last_part:
        previous = last_part
        last_part = Path(last_part).stem.lower()

    return "".join(ch for ch in last_part if ch.isalnum())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: str(column).strip().lower().replace(" ", "_") for column in df.columns}
    return df.rename(columns=renamed)


def _read_excel_with_zip_fallback(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except ImportError:
        pass

    shared_strings: list[str] = []
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with ZipFile(path) as workbook:
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", namespace):
                shared_strings.append("".join(node.text or "" for node in item.findall(".//a:t", namespace)))

        sheet_root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []

        for row in sheet_root.findall(".//a:row", namespace):
            values_by_index: dict[int, str] = {}
            max_index = -1
            for cell in row.findall("a:c", namespace):
                cell_type = cell.attrib.get("t")
                reference = cell.attrib.get("r", "")
                value_node = cell.find("a:v", namespace)
                value = "" if value_node is None or value_node.text is None else value_node.text
                if cell_type == "s" and value != "":
                    value = shared_strings[int(value)]
                column_letters = "".join(ch for ch in reference if ch.isalpha())
                column_index = 0
                for letter in column_letters:
                    column_index = column_index * 26 + (ord(letter.upper()) - ord("A") + 1)
                column_index -= 1
                if column_index < 0:
                    column_index = max_index + 1
                values_by_index[column_index] = value
                max_index = max(max_index, column_index)
            values = [values_by_index.get(index, "") for index in range(max_index + 1)]
            rows.append(values)

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    body = rows[1:]
    normalized_rows = [row + [""] * (len(header) - len(row)) for row in body]
    return pd.DataFrame(normalized_rows, columns=header)


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _discover_local_images(images_root: str | Path) -> list[Path]:
    root = Path(images_root)
    image_paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_paths.append(path)
    return sorted(image_paths)


def _build_image_lookup(paths: Iterable[Path]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in paths:
        for key in {
            _canonicalize_name(path.name),
            _canonicalize_name(path.stem),
            _canonicalize_name(str(path.relative_to(path.parents[0]))),
        }:
            if key and key not in lookup:
                lookup[key] = path
    return lookup


def load_armenian_records(
    project_root: str | Path,
    metadata_path: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> list[ArtworkRecord]:
    paths = ProjectPaths(Path(project_root))
    metadata_file = Path(metadata_path) if metadata_path is not None else paths.armenian_metadata_path
    local_images_root = Path(images_dir) if images_dir is not None else paths.armenian_images_dir
    df = _normalize_columns(_read_excel_with_zip_fallback(metadata_file))

    filename_column = _pick_column(df, ["filename", "file_name", "file"])
    title_column = _pick_column(df, ["painting_name", "title", "painting"])
    artist_column = _pick_column(df, ["artist", "painter"])
    year_column = _pick_column(df, ["year"])

    if filename_column is None:
        raise ValueError("metadata file is missing a filename column")

    image_lookup = _build_image_lookup(_discover_local_images(local_images_root))
    records: list[ArtworkRecord] = []
    matched_paths: set[str] = set()

    for row in df.to_dict(orient="records"):
        raw_filename = row.get(filename_column)
        if pd.isna(raw_filename):
            continue

        image_path = image_lookup.get(_canonicalize_name(str(raw_filename)))
        if image_path is None:
            continue
        matched_paths.add(str(image_path))

        title = row.get(title_column) if title_column else None
        artist = row.get(artist_column) if artist_column else None
        year = row.get(year_column) if year_column else None

        records.append(
            ArtworkRecord(
                source="armenian_local",
                image_path=str(image_path),
                filename=image_path.name,
                title=None if pd.isna(title) else str(title),
                artist=None if pd.isna(artist) else str(artist).strip(),
                year=None if pd.isna(year) else str(year),
            )
        )

    for image_path in _discover_local_images(local_images_root):
        if str(image_path) in matched_paths:
            continue
        records.append(
            ArtworkRecord(
                source="armenian_local",
                image_path=str(image_path),
                filename=image_path.name,
                title=image_path.stem.replace("_", " ").replace("-", " "),
            )
        )

    return records


def _load_optional_metadata_from_kaggle(handle: str, metadata_path: str | None) -> pd.DataFrame | None:
    if metadata_path == "":
        return None

    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError:
        return None

    try:
        if metadata_path:
            df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, handle, metadata_path)
            return _normalize_columns(df)
    except Exception:
        return None

    return None


def _download_kaggle_dataset(handle: str, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if output_path.exists():
        has_images = any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in output_path.rglob("*")
        )
        if has_images:
            return output_path

    import kagglehub

    dataset_path = kagglehub.dataset_download(handle, output_dir=str(output_path))
    return Path(dataset_path)


def _infer_wikiart_metadata_from_path(dataset_root: Path, image_path: Path) -> ArtworkRecord:
    relative_parts = image_path.relative_to(dataset_root).parts
    style = image_path.parent.name if len(relative_parts) >= 2 else None

    stem = image_path.stem
    artist = None
    title = None

    if "_" in stem:
        artist_part, title_part = stem.split("_", 1)
        artist = artist_part.replace("-", " ").strip() or None
        title = title_part.replace("-", " ").replace("_", " ").strip() or None
    else:
        title = stem.replace("-", " ").replace("_", " ").strip() or None

    return ArtworkRecord(
        source="wikiart",
        image_path=str(image_path),
        filename=image_path.name,
        title=title,
        artist=artist,
        style=style,
        genre=None,
    )


def _resolve_image_path(dataset_root: Path, raw_value: str, lookup: dict[str, Path]) -> Path | None:
    candidates = [
        raw_value,
        Path(raw_value).name,
        Path(raw_value).stem,
    ]
    for candidate in candidates:
        key = _canonicalize_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def _records_from_metadata(dataset_root: Path, df: pd.DataFrame) -> list[ArtworkRecord]:
    df = _normalize_columns(df)
    image_column = _pick_column(df, ["image_path", "image", "path", "filename", "file", "name"])
    artist_column = _pick_column(df, ["artist", "artist_name", "painter"])
    style_column = _pick_column(df, ["style", "art_style"])
    genre_column = _pick_column(df, ["genre"])
    title_column = _pick_column(df, ["title", "painting_name", "name"])
    year_column = _pick_column(df, ["year", "year_created"])

    if image_column is None:
        return []

    image_paths = [path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    lookup = _build_image_lookup(image_paths)
    records: list[ArtworkRecord] = []

    for row in df.to_dict(orient="records"):
        raw_path = row.get(image_column)
        if raw_path is None or pd.isna(raw_path):
            continue

        resolved = _resolve_image_path(dataset_root, str(raw_path), lookup)
        if resolved is None:
            continue

        artist = row.get(artist_column) if artist_column else None
        style = row.get(style_column) if style_column else None
        genre = row.get(genre_column) if genre_column else None
        title = row.get(title_column) if title_column else None
        year = row.get(year_column) if year_column else None

        records.append(
            ArtworkRecord(
                source="wikiart",
                image_path=str(resolved),
                filename=resolved.name,
                title=None if pd.isna(title) else str(title),
                artist=None if pd.isna(artist) else str(artist),
                style=None if pd.isna(style) else str(style),
                genre=None if pd.isna(genre) else str(genre),
                year=None if pd.isna(year) else str(year),
            )
        )

    return records


def _records_from_folder_structure(dataset_root: Path) -> list[ArtworkRecord]:
    image_paths = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(dataset_root).parts)
    ]
    return [_infer_wikiart_metadata_from_path(dataset_root, path) for path in sorted(image_paths)]


def _sample_diverse_artists(records: list[ArtworkRecord], sample_size: int, seed: int) -> list[ArtworkRecord]:
    if len(records) <= sample_size:
        return records

    rng = random.Random(seed)
    by_artist: dict[str, list[ArtworkRecord]] = {}
    for record in records:
        artist = (record.artist or "unknown_artist").strip() or "unknown_artist"
        by_artist.setdefault(artist, []).append(record)

    for artist_records in by_artist.values():
        rng.shuffle(artist_records)

    artist_keys = list(by_artist.keys())
    rng.shuffle(artist_keys)

    selected: list[ArtworkRecord] = []
    per_artist_cap = max(1, math.ceil(sample_size / max(len(artist_keys), 1)))

    for artist in artist_keys:
        selected.extend(by_artist[artist][:per_artist_cap])
        if len(selected) >= sample_size:
            return selected[:sample_size]

    remaining = [record for artist in artist_keys for record in by_artist[artist][per_artist_cap:]]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    return selected[:sample_size]


def load_wikiart_records(
    sample_size: int = 4500,
    kaggle_handle: str = "steubk/wikiart",
    metadata_path: str | None = "",
    output_dir: str | Path = "data/wikiart_raw",
    seed: int = 42,
) -> list[ArtworkRecord]:
    dataset_root = _download_kaggle_dataset(kaggle_handle, output_dir)
    metadata_df = _load_optional_metadata_from_kaggle(kaggle_handle, metadata_path)

    if metadata_df is not None:
        records = _records_from_metadata(dataset_root, metadata_df)
    else:
        records = _records_from_folder_structure(dataset_root)

    records = [record for record in records if Path(record.image_path).exists()]
    return _sample_diverse_artists(records, sample_size=sample_size, seed=seed)
