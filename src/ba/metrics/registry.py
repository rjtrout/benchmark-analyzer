"""The metrics, by id. Each one turns a normalized reference and transcript into one test case's row.

WER and CER give counts with the same four fields (reference units, substitutions,
deletions, insertions), and each is registered with the function that counts them.
Semantic error is no count: it is a rate, from a sentence encoder, with the reference's
word count as its weight (see ``ba.metrics.semantic``). The aggregations read all three alike.
"""

from __future__ import annotations

from ba.metrics import cer, wer
from ba.utils.registry import Registry

METRICS = Registry("metrics")
METRICS.register("wer", wer.counts)
METRICS.register("cer", cer.counts)
METRICS.register("semantic", None)
