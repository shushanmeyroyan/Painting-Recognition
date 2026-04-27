from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class SearchMatch:
    rank: int
    score: float
    metadata: dict[str, object]


class ArtVectorDatabase:
    def __init__(
        self,
        index_path: str | Path = "data/faiss_index.idx",
        mapping_path: str | Path = "data/index_mapping.json",
        embeddings_path: str | Path = "data/embeddings.npy",
    ) -> None:
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.embeddings_path = Path(embeddings_path)
        self.index: faiss.Index | None = None
        self.mapping: list[dict[str, object]] = []

    @property
    def exists(self) -> bool:
        return self.index_path.exists() and self.mapping_path.exists()

    def build(self, embeddings: np.ndarray, mapping: list[dict[str, object]]) -> None:
        if len(embeddings) != len(mapping):
            raise ValueError("embeddings and mapping must have the same length")
        if len(embeddings) == 0:
            raise ValueError("cannot build an empty FAISS index")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        np.save(self.embeddings_path, embeddings)
        self.mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

        self.index = index
        self.mapping = mapping

    def load(self) -> "ArtVectorDatabase":
        if not self.exists:
            raise FileNotFoundError("FAISS index or mapping file is missing; build the index first")

        self.index = faiss.read_index(str(self.index_path))
        self.mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        return self

    def load_embeddings(self) -> np.ndarray:
        if not self.embeddings_path.exists():
            raise FileNotFoundError("saved embeddings are missing")
        embeddings = np.load(self.embeddings_path).astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[SearchMatch]:
        if self.index is None:
            self.load()
        if self.index is None or self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        matches: list[SearchMatch] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            matches.append(
                SearchMatch(
                    rank=rank,
                    score=float(score),
                    metadata=self.mapping[idx],
                )
            )
        return matches

    def export_matches(self, query_embedding: np.ndarray, k: int = 3) -> list[dict[str, object]]:
        return [asdict(match) for match in self.search(query_embedding, k=k)]

    def export_matches_with_numpy(self, query_embedding: np.ndarray, k: int = 3) -> list[dict[str, object]]:
        embeddings = self.load_embeddings().astype(np.float32)
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = np.linalg.norm(query)
        if query_norm != 0:
            query = query / query_norm

        scores = embeddings @ query
        top_indices = np.argsort(scores)[::-1][:k]

        matches: list[dict[str, object]] = []
        for rank, idx in enumerate(top_indices, start=1):
            matches.append(
                asdict(
                    SearchMatch(
                        rank=rank,
                        score=float(scores[idx]),
                        metadata=self.mapping[int(idx)],
                    )
                )
            )
        return matches
