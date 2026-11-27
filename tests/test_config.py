from __future__ import annotations

import pytest

from ba.config import BenchmarkConfig, SystemConfig, load_config
from ba.paths import Paths, find_root


def test_find_root_walks_up_to_the_project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "benchmark-analyzer"\n')
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.delenv("BA_ROOT", raising=False)
    monkeypatch.chdir(nested)
    assert find_root() == tmp_path


def test_ba_root_env_overrides_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("BA_ROOT", str(tmp_path))
    assert find_root() == tmp_path
    assert Paths.discover().artifacts == tmp_path / "artifacts"


def test_the_two_clones_are_under_third_party_unless_they_are_given(tmp_path):
    paths = Paths(root=tmp_path)
    assert paths.clone == tmp_path / "third_party" / "open_asr_leaderboard"
    assert paths.urgent_clone == tmp_path / "third_party" / "urgent2024_analysis"
    elsewhere = Paths(root=tmp_path, clone=tmp_path / "a", urgent_clone=tmp_path / "b")
    assert (elsewhere.clone, elsewhere.urgent_clone) == (tmp_path / "a", tmp_path / "b")


def test_an_unknown_benchmark_names_the_ones_there_are(paths):
    cfg = load_config(paths)
    with pytest.raises(KeyError, match="ami-cleaned"):
        cfg.benchmark("ami")


def test_a_key_the_config_does_not_know_is_refused():
    # A misspelt key in a config file is an error, and is not passed over.
    with pytest.raises(ValueError, match="rank"):
        SystemConfig(id="lab/alpha", rank=3)


def test_a_benchmark_knows_the_names_its_public_transcripts_go_by():
    plain = BenchmarkConfig(id="ami-cleaned", loader="hf", dataset_key="ami_cleaned", source={"repo": "hf-audio/open-asr-leaderboard", "config": "ami_cleaned"})
    assert plain.manifest_dataset_ids == ["hf-audio-open-asr-leaderboard_ami_cleaned_test"]
    older = BenchmarkConfig(id="e22", loader="hf", dataset_key="e22", source={"repo": "AA/E22"}, manifest_aliases=["AA-E22_e22_test"])
    assert older.manifest_dataset_ids == ["AA-E22__test", "AA-E22_e22_test"]  # no config: an empty part


