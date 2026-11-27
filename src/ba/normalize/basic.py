"""The two plainest normalizers: the text as written, and the floor that only drops case and punctuation."""

from __future__ import annotations

import unicodedata


def verbatim(text: str) -> str:
    """No normalization: the text as written."""
    return text


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})  # the curly and the modifier forms of the same mark


def case_punct(text: str) -> str:
    """Lowercase, and turn every punctuation mark or symbol into a space, except an apostrophe inside a word.

    Inside a word the apostrophe is spelling: without it "we'll" is "well" and "it's" is
    "its". At the edge of a word it is a quotation mark, or a plural's possessive, and
    goes as punctuation does.
    """
    lowered = text.lower().translate(_APOSTROPHES)

    # Every punctuation mark or symbol becomes a space, except the apostrophe.
    characters = []
    for character in lowered:
        is_mark_or_symbol = unicodedata.category(character)[0] in ("P", "S")
        if is_mark_or_symbol and character != "'":
            characters.append(" ")
        else:
            characters.append(character)

    # An apostrophe at a word's edge goes too, and a word that was nothing else goes with it.
    words = []
    for word in "".join(characters).split():
        word = word.strip("'")
        if word:
            words.append(word)
    return " ".join(words)
