"""How things are printed: tables that fit a terminal, and the short names in them."""

from __future__ import annotations

from ba.utils import display_utils


def test_a_models_short_name_is_the_first_words_of_its_name_that_tell_it_apart():
    board = [
        "Musci-research/Musci-ASR-qwen-1019-hf", "salesforce-sovereign-intelligence/SOV-Transcribe-1", "oruk/orukqwen-asr-1.7b",
        "FluidVoice-Labs/fluid-speech-pro", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8",
    ]  # fmt: skip
    assert display_utils.brief_names(board) == dict(zip(board, ["Musci", "SOV", "orukqwen", "fluid", "NVIDIA"], strict=True))
    # Two models that begin alike keep as many words as it takes, and no other model's name grows for it.
    assert display_utils.brief_names(["openai/whisper-large-v3", "openai/whisper-large-v3-turbo", "nvidia/parakeet_tdt-1.1b"]) == {
        "openai/whisper-large-v3": "whisper-large-v3", "openai/whisper-large-v3-turbo": "whisper-large-v3-turbo", "nvidia/parakeet_tdt-1.1b": "parakeet",
    }  # fmt: skip
    # The same name under two organisations: only the whole id tells them apart.
    assert display_utils.brief_names(["a/model", "b/model"]) == {"a/model": "a/model", "b/model": "b/model"}


def test_a_table_can_set_its_columns_of_figures_to_the_right():
    rows = [["all families on", "9.44", "10.54"], ["without case", "+4.39", "+3.12"]]
    assert display_utils.table(["variant", "orukqwen", "SOV"], rows, right=[1, 2]).splitlines() == [
        "variant          orukqwen    SOV",
        "---------------  --------  -----",
        "all families on      9.44  10.54",
        "without case        +4.39  +3.12",
    ]
    # Left alone, every column starts at the left, as text does.
    assert display_utils.table(["variant", "orukqwen"], [row[:2] for row in rows]).splitlines()[2:] == ["all families on  9.44", "without case     +4.39"]


def test_an_errors_message_is_what_it_says_without_the_quotes_python_puts_round_a_missing_key():
    # str() of a KeyError is its key in quotes: "'No such thing.'". The message is the key itself.
    assert display_utils.error_message(KeyError("No system 'x'.")) == "No system 'x'."
    assert display_utils.error_message(ValueError("Choose one of: a, b")) == "Choose one of: a, b"
    assert display_utils.error_message(KeyError()) == ""
