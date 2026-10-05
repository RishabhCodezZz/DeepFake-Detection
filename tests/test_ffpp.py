"""Prevent silent corpus changes and identity leakage in FF++ discovery."""
import json
from pathlib import Path

import pytest

from ffpp_v5 import build_ffpp_manifest, discover_ffpp_videos, load_ffpp_official_splits, split_ffpp_identity_disjoint


def test_discovery_does_not_fall_back_to_explicitly_wrong_compression(tmp_path):
    wrong = tmp_path / "original_sequences" / "youtube" / "c40" / "videos" / "001.mp4"
    wrong.parent.mkdir(parents=True)
    wrong.touch()
    found = discover_ffpp_videos((tmp_path.as_posix(),), compression="c23", verbose=False)
    assert found["original"] == []


def test_discovery_accepts_unlabelled_flat_mirrors(tmp_path):
    path = tmp_path / "original" / "001.mp4"
    path.parent.mkdir()
    path.touch()
    found = discover_ffpp_videos((tmp_path.as_posix(),), verbose=False)
    assert [Path(p) for p in found["original"]] == [path]


def test_discovery_checks_fallback_independently_for_each_root(tmp_path):
    canonical = tmp_path / "first" / "original_sequences" / "youtube" / "c23" / "001.mp4"
    flat = tmp_path / "second" / "original" / "002.mp4"
    for p in (canonical, flat):
        p.parent.mkdir(parents=True)
        p.touch()
    found = discover_ffpp_videos(tuple((tmp_path / name).as_posix() for name in ("first", "second")), verbose=False)
    assert {Path(p) for p in found["original"]} == {canonical, flat}


def test_duplicate_mirrors_are_rejected_before_cache_files_are_overwritten():
    with pytest.raises(ValueError, match="Duplicate FF"):
        build_ffpp_manifest({"original": ["/mirror1/001.mp4", "/mirror2/001.mp4"]}, verbose=False)


def test_split_loader_ignores_unrelated_json_and_uses_one_directory(tmp_path):
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    for name in ("train", "val", "test"):
        (unrelated / (name + ".json")).write_text(json.dumps({"images": []}))
    official = tmp_path / "ffpp"
    official.mkdir()
    data = {"train": [["001", "002"]], "val": [["003", "004"]], "test": [["005", "006"]]}
    for name, pairs in data.items():
        (official / (name + ".json")).write_text(json.dumps(pairs))
    splits = load_ffpp_official_splits((tmp_path.as_posix(),))
    assert splits == {name: set(pairs[0]) for name, pairs in data.items()}


def test_split_loader_does_not_combine_separate_datasets(tmp_path):
    for name, pair in (("train", ["001", "002"]), ("val", ["003", "004"]), ("test", ["005", "006"])):
        folder = tmp_path / name
        folder.mkdir()
        (folder / (name + ".json")).write_text(json.dumps([pair]))
    assert load_ffpp_official_splits((tmp_path.as_posix(),)) is None


@pytest.mark.parametrize("source,target", [("003", "001"), ("001", "003"), ("", "999")])
def test_official_split_rejects_cross_partition_pairs_and_unknown_ids(source, target):
    rows = [{"target_id": target, "source_id": source, "label": 1}]
    splits = {"train": {"001", "002"}, "val": {"003", "004"}, "test": {"005", "006"}}
    with pytest.raises(ValueError):
        split_ffpp_identity_disjoint(rows, official_splits=splits, verbose=False)
