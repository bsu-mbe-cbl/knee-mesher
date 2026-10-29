"""Verify local data release assets and GitHub dataset downloads."""

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from kneemesher import datasets

_BUILDER_PATH = Path(__file__).parents[1] / "tools/build_dataset_release.py"
_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "kneemesher_dataset_release_builder", _BUILDER_PATH
)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(_BUILDER_SPEC)
sys.modules[_BUILDER_SPEC.name] = builder
_BUILDER_SPEC.loader.exec_module(builder)


def _write_config(path: Path, subjects: list[str]) -> None:
    """Write a small reconstruction config for archive-builder tests."""
    path.write_text(
        "[subjects]\ninclude = ["
        + ", ".join(json.dumps(subject) for subject in subjects)
        + "]\n",
        encoding="utf-8",
    )


def test_build_release_contains_cohort_and_full_dataset(tmp_path: Path) -> None:
    """Build deterministic example and full-dataset ZIP assets."""
    source = tmp_path / "source"
    source.mkdir()
    subjects = ["1001", "1002", "1003", "1004", "1005"]
    filenames = [f"{subject}_M00.h5" for subject in subjects] + [
        "2001_M00.h5",
        "2002_M00.h5",
    ]
    for index, name in enumerate(filenames):
        (source / name).write_bytes(bytes([index]) * 300)
    config = tmp_path / "config.toml"
    _write_config(config, subjects)

    output = tmp_path / "release"
    manifest_path = builder.build_release(
        source,
        config,
        output,
        version="v1",
        max_archive_bytes=700,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["release_tag"] == "data-manual-segmentations-2023-v1"
    assert manifest["datasets"]["example"]["files"] == sorted(filenames[:5])
    assert manifest["datasets"]["all"]["files"] == sorted(filenames)

    for dataset in manifest["datasets"].values():
        archived = []
        for archive in dataset["archives"]:
            archive_path = output / archive["name"]
            assert archive_path.stat().st_size == archive["size_bytes"]
            assert archive_path.stat().st_size < 2 * builder.GIB
            assert builder._sha256_file(archive_path) == archive["sha256"]
            with zipfile.ZipFile(archive_path) as zipped:
                archived.extend(zipped.namelist())
        assert sorted(archived) == dataset["files"]

    repeat_output = tmp_path / "repeat"
    builder.build_release(
        source,
        config,
        repeat_output,
        version="v1",
        max_archive_bytes=700,
    )
    for archive in output.glob("*.zip"):
        assert builder._sha256_file(archive) == builder._sha256_file(
            repeat_output / archive.name
        )


def test_split_files_rejects_limit_at_or_above_two_gibibytes() -> None:
    """Enforce a strict upper bound below GitHub's 2 GiB asset ceiling."""
    with pytest.raises(ValueError, match="below 2 GiB"):
        builder._split_files([], 2 * builder.GIB)


def test_download_dataset_verifies_and_extracts_release_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download a mocked public release and verify the extracted HDF5 file."""
    tag = "data-manual-segmentations-2023-v-test"
    filename = "1234_M00.h5"
    content = b"synthetic hdf5 fixture"
    archive_name = "manual-segmentations-2023-test-example-001.zip"
    archive_bytes_io = io.BytesIO()
    with zipfile.ZipFile(archive_bytes_io, "w", zipfile.ZIP_STORED) as zipped:
        zipped.writestr(filename, content)
    archive_bytes = archive_bytes_io.getvalue()
    manifest = {
        "schema_version": 1,
        "release_tag": tag,
        "repository": "example/project",
        "files": {
            filename: {
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        },
        "datasets": {
            "example": {
                "files": [filename],
                "archives": [
                    {
                        "name": archive_name,
                        "size_bytes": len(archive_bytes),
                        "sha256": hashlib.sha256(archive_bytes).hexdigest(),
                        "files": [filename],
                    }
                ],
            }
        },
    }
    manifest_bytes = json.dumps(manifest).encode()
    urls = {
        "https://api.github.com/repos/example/project/releases/tags/" + tag: json.dumps(
            {
                "assets": [
                    {
                        "name": datasets.MANIFEST_NAME,
                        "size": len(manifest_bytes),
                        "browser_download_url": (
                            "https://github.com/example/project/releases/download/"
                            f"{tag}/manifest"
                        ),
                    },
                    {
                        "name": archive_name,
                        "size": len(archive_bytes),
                        "browser_download_url": (
                            "https://github.com/example/project/releases/download/"
                            f"{tag}/{archive_name}"
                        ),
                    },
                ]
            }
        ).encode(),
        f"https://github.com/example/project/releases/download/{tag}/manifest": manifest_bytes,
        (
            f"https://github.com/example/project/releases/download/{tag}/{archive_name}"
        ): archive_bytes,
    }

    def fake_urlopen(request, timeout=60):
        """Return fixture responses for the GitHub API and release URLs."""
        url = request.full_url
        return io.BytesIO(urls[url])

    monkeypatch.setattr(datasets.urllib.request, "urlopen", fake_urlopen)
    output = tmp_path / "output"
    extracted = datasets.download_dataset(
        "example",
        output,
        repository="example/project",
        release_tag=tag,
        cache_dir=tmp_path / "cache",
    )
    assert extracted == [output / filename]
    assert (output / filename).read_bytes() == content


@pytest.mark.parametrize(
    "filename", ["../outside.h5", "nested/file.h5", "bad\\file.h5"]
)
def test_validate_filename_rejects_paths(filename: str) -> None:
    """Reject archive members that could escape the flat dataset directory."""
    with pytest.raises(datasets.DatasetDownloadError, match="Invalid dataset filename"):
        datasets._validate_filename(filename)
