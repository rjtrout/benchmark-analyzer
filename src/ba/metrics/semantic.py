"""Semantic error: how far a transcript's meaning is from its reference's.

Both texts are normalized, embedded with a sentence encoder, and compared: a test case's
semantic error is one minus the cosine similarity of the two embeddings. Two identical texts
are at 0, and an empty transcript of a test case where something was said is at 1. It is a
rate per test case, like a test case's WER, so it has the same mean, median and overall
figures: the overall one weights each test case by its reference's words.

There is one encoder, named below. It is behind a small interface, so everything here is
tested with a stand-in.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

MODEL = "sentence-transformers/all-mpnet-base-v2"  # the one sentence encoder


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self) -> None:
        # Slow to import, and only computing semantic error needs it.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(MODEL)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, batch_size=64, show_progress_bar=False))


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    """One minus the cosine similarity of two embeddings. Never below 0."""
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm == 0:
        return 1.0
    cosine = float(np.dot(first, second)) / norm
    return max(0.0, 1.0 - cosine)


def distances(
    pairs: list[tuple[str, str]],
    embedder: Embedder,
    references: dict[str, np.ndarray] | None = None,
) -> list[float]:
    """One minus cosine similarity for each (reference, transcript) pair.

    An empty transcript is 1.0, as far as a transcript of something said can be, unless
    the reference is empty too, which is 0.0: nothing was said and nothing was written.
    Identical texts are exactly 0.0, and no distance is below 0.

    Pass the same ``references`` dictionary across model runs to reuse the embeddings of
    any repeated references and transcripts.
    """
    embeddings = {} if references is None else references

    # Only a pair of two different texts, neither of them empty, needs the encoder.
    unknown = set()
    for reference, transcript in pairs:
        if not reference.strip() or not transcript.strip() or reference == transcript:
            continue
        if reference not in embeddings:
            unknown.add(reference)
        if transcript not in embeddings:
            unknown.add(transcript)
    if unknown:
        texts = sorted(unknown)
        vectors = embedder.encode(texts)
        embeddings.update(dict(zip(texts, vectors, strict=True)))

    out = []
    for reference, transcript in pairs:
        nothing_said = not reference.strip()
        nothing_written = not transcript.strip()
        if nothing_said and nothing_written:
            out.append(0.0)
        elif nothing_said or nothing_written:
            out.append(1.0)
        elif reference == transcript:
            out.append(0.0)
        else:
            out.append(_distance(embeddings[reference], embeddings[transcript]))
    return out
