from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_GSM_SOURCE_RE = re.compile(r"(?:^|[^a-z0-9])gsm[\s_-]*8k(?:$|[^a-z0-9])", re.I)


def normalize_text(text: str) -> str:
    """Normalize text conservatively for contamination checks."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _PUNCT_RE.sub(" ", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def character_ngrams(text: str, size: int = 5) -> set[str]:
    normalized = normalize_text(text)
    padded = f" {normalized} "
    if len(padded) <= size:
        return {padded} if padded else set()
    return {padded[index : index + size] for index in range(len(padded) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _minhash(tokens: set[str], num_perm: int) -> MinHash:
    signature = MinHash(num_perm=num_perm)
    for token in sorted(tokens):
        signature.update(token.encode("utf-8"))
    return signature


@dataclass(frozen=True)
class ReferenceText:
    key: str
    text: str


@dataclass(frozen=True)
class RemovalRecord:
    dataset_index: int
    reason: str
    train_hash: str
    reference_key: str | None = None
    similarity: float | None = None
    source: str | None = None


class ReferenceIndex:
    """Exact-hash and MinHash-LSH index with exact Jaccard confirmation."""

    def __init__(
        self,
        references: Iterable[ReferenceText],
        *,
        approximate_threshold: float = 0.90,
        candidate_threshold: float = 0.75,
        num_perm: int = 64,
        ngram_size: int = 5,
    ) -> None:
        if candidate_threshold > approximate_threshold:
            raise ValueError("candidate_threshold must not exceed approximate_threshold")
        self.approximate_threshold = approximate_threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.exact: dict[str, str] = {}
        self.ngrams: dict[str, set[str]] = {}
        self.lsh = MinHashLSH(threshold=candidate_threshold, num_perm=num_perm)
        for reference in references:
            digest = text_hash(reference.text)
            self.exact.setdefault(digest, reference.key)
            grams = character_ngrams(reference.text, ngram_size)
            self.ngrams[reference.key] = grams
            self.lsh.insert(reference.key, _minhash(grams, num_perm))

    def match(self, text: str) -> tuple[str, str | None, float | None]:
        digest = text_hash(text)
        if digest in self.exact:
            return "benchmark_exact", self.exact[digest], 1.0
        grams = character_ngrams(text, self.ngram_size)
        best_key: str | None = None
        best_score = 0.0
        for key in self.lsh.query(_minhash(grams, self.num_perm)):
            score = jaccard(grams, self.ngrams[key])
            if score > best_score:
                best_key, best_score = key, score
        if best_key is not None and best_score >= self.approximate_threshold:
            return "benchmark_approximate", best_key, best_score
        return "clean", None, None


class Decontaminator:
    def __init__(self, reference_index: ReferenceIndex, *, remove_internal_duplicates: bool = True):
        self.reference_index = reference_index
        self.remove_internal_duplicates = remove_internal_duplicates
        self.seen_hashes: set[str] = set()
        self.removals: list[RemovalRecord] = []

    def keep(
        self,
        text: str,
        *,
        dataset_index: int,
        source: str | None = None,
        reject_gsm_source: bool = False,
    ) -> bool:
        digest = text_hash(text)
        if reject_gsm_source and source and _GSM_SOURCE_RE.search(source):
            self.removals.append(
                RemovalRecord(dataset_index, "source_benchmark", digest, source=source)
            )
            return False
        reason, reference_key, similarity = self.reference_index.match(text)
        if reason != "clean":
            self.removals.append(
                RemovalRecord(
                    dataset_index,
                    reason,
                    digest,
                    reference_key=reference_key,
                    similarity=similarity,
                    source=source,
                )
            )
            return False
        if self.remove_internal_duplicates and digest in self.seen_hashes:
            self.removals.append(
                RemovalRecord(dataset_index, "internal_duplicate", digest, source=source)
            )
            return False
        self.seen_hashes.add(digest)
        return True

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for removal in self.removals:
            counts[removal.reason] = counts.get(removal.reason, 0) + 1
        return counts

    def write_audit(self, directory: str | Path, metadata: dict[str, Any]) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "removed.jsonl").open("w", encoding="utf-8") as handle:
            for removal in self.removals:
                handle.write(json.dumps(asdict(removal), sort_keys=True) + "\n")
        manifest = dict(metadata)
        manifest["counts"] = self.counts()
        manifest["total_removed"] = len(self.removals)
        with (target / "decontamination_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
