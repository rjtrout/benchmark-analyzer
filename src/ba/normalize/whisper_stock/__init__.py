"""OpenAI Whisper's English text normalizer, as shipped.

The three files beside this one (basic.py, english.py, english.json) are copied unchanged
from https://github.com/openai/whisper at commit 86098128c0b4f24f0e2aa2994de830614b474227
(whisper/normalizers/), under the MIT licence in LICENSE. This is the stock normalizer, not
the Open ASR Leaderboard's fork of it, so the two can be compared.
"""

from .english import EnglishTextNormalizer

__all__ = ["EnglishTextNormalizer"]
