"""Build reproducible ZIP assets for a GitHub dataset release."""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/segmentations/manual-segmentations-2023"
DEFAULT_CONFIG = ROOT / "configs/00-kg-reconstruct-knee-geometries.toml"
DEFAULT_OUTPUT = ROOT / "dist/dataset-releases"
GIB = 1024**3
DEFAULT_ARCHIVE_LIMIT = 1_900_000_000
RELEASE_TAG_PREFIX = "data-manual-segmentations-2023-"
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class FileRecord:
    """Describe one source HDF5 file."""

    path: Path
    name: str
    size_bytes: int
    sha256: str


def _sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _zip_size_overhead(name: str) -> int:
    """Return ZIP header bytes for one flat, non-ZIP64 file."""
    name_size = len(name.encode("utf-8"))
    return 76 + 2 * name_size


def _split_files(
    files: list[FileRecord], max_archive_bytes: int
) -> list[list[FileRecord]]:
    """Group whole files into ZIPs whose estimated size stays below the limit."""
    if not 0 < max_archive_bytes < 2 * GIB:
        raise ValueError("max_archive_bytes must be greater than zero and below 2 GiB")
    groups: list[list[FileRecord]] = []
    current: list[FileRecord] = []
    current_size = 22
    for record in files:
        file_size = record.size_bytes + _zip_size_overhead(record.name)
        if file_size + 22 > max_archive_bytes:
            raise ValueError(f"{record.name} cannot fit below the archive size limit")
        if current and current_size + file_size > max_archive_bytes:
            groups.append(current)
            current = []
            current_size = 22
        current.append(record)
        current_size += file_size
    if current:
        groups.append(current)
    return groups


def _write_archive(
    output_dir: Path,
    name: str,
    files: list[FileRecord],
    max_archive_bytes: int,
    overwrite: bool,
) -> dict[str, object]:
    """Write one deterministic, uncompressed ZIP and return its metadata."""
    destination = output_dir / name
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; use --overwrite")
    temporary_handle = tempfile.NamedTemporaryFile(
        dir=output_dir,
        prefix=f".{name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=False,
        ) as zipped:
            for record in files:
                info = zipfile.ZipInfo(record.name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.file_size = record.size_bytes
                with record.path.open("rb") as source, zipped.open(info, "w") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        size, digest = temporary.stat().st_size, _sha256_file(temporary)
        if size >= 2 * GIB or size > max_archive_bytes:
            raise ValueError(
                f"{name} is {size} bytes, above the configured limit "
                f"of {max_archive_bytes}"
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "name": name,
        "size_bytes": size,
        "sha256": digest,
        "files": [record.name for record in files],
    }


def _collect_files(source_dir: Path) -> list[FileRecord]:
    """Read and hash all immediate HDF5 files in the source directory."""
    paths = sorted(source_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No HDF5 files found in {source_dir}")
    records = [
        FileRecord(
            path=path,
            name=path.name,
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in paths
    ]
    if len({record.name for record in records}) != len(records):
        raise ValueError("Source contains duplicate HDF5 filenames")
    return records


def _load_example_subjects(config_path: Path) -> list[str]:
    """Read the five-subject example cohort from the reconstruction config."""
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    subjects = config.get("subjects", {}).get("include")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError(f"No example cohort is configured in {config_path}")
    if not all(isinstance(subject, str) and subject for subject in subjects):
        raise ValueError(f"Invalid example cohort in {config_path}")
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"Duplicate example subjects in {config_path}")
    return sorted(subjects)


def _write_manifest(path: Path, manifest: dict[str, object], overwrite: bool) -> None:
    """Write the release manifest atomically."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; use --overwrite")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def build_release(
    source_dir: Path,
    config_path: Path,
    output_dir: Path,
    version: str,
    max_archive_bytes: int = DEFAULT_ARCHIVE_LIMIT,
    overwrite: bool = False,
) -> Path:
    """Build the example archive, full-dataset chunks, and their manifest."""
    if _VERSION_RE.fullmatch(version) is None:
        raise ValueError("version must contain only letters, digits, '.', '_' or '-'")
    if not 0 < max_archive_bytes < 2 * GIB:
        raise ValueError("max_archive_bytes must be greater than zero and below 2 GiB")
    source_dir = Path(source_dir)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    records = _collect_files(source_dir)
    example_subjects = _load_example_subjects(config_path)
    example_files = [
        record
        for record in records
        if record.name.split("_", maxsplit=1)[0] in example_subjects
    ]
    found_subjects = {record.name.split("_", maxsplit=1)[0] for record in example_files}
    missing_subjects = set(example_subjects) - found_subjects
    if missing_subjects:
        raise ValueError(
            f"No HDF5 files found for subjects: {sorted(missing_subjects)}"
        )
    if not example_files:
        raise ValueError("The configured example cohort has no HDF5 files")

    release_tag = RELEASE_TAG_PREFIX + version
    example_groups = _split_files(example_files, max_archive_bytes)
    example_names = [
        f"manual-segmentations-2023-{version}-example-{index:03d}.zip"
        for index in range(1, len(example_groups) + 1)
    ]
    full_groups = _split_files(records, max_archive_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_names = [
        f"manual-segmentations-2023-{version}-full-{index:03d}.zip"
        for index in range(1, len(full_groups) + 1)
    ]
    output_names = [*example_names, *full_names, "dataset-manifest.json"]
    existing = {path.name for path in output_dir.iterdir()}
    unexpected = sorted(existing - set(output_names))
    if unexpected:
        raise ValueError(
            "Output directory contains files not generated for this release: "
            + ", ".join(unexpected)
        )
    if existing and not overwrite:
        raise FileExistsError(
            f"Output assets already exist: {', '.join(sorted(existing))}; use --overwrite"
        )

    example_archives = [
        _write_archive(
            output_dir,
            name,
            group,
            max_archive_bytes,
            overwrite,
        )
        for name, group in zip(example_names, example_groups, strict=True)
    ]
    full_archives = [
        _write_archive(
            output_dir,
            name,
            group,
            max_archive_bytes,
            overwrite,
        )
        for name, group in zip(full_names, full_groups, strict=True)
    ]
    file_records = {
        record.name: {"size_bytes": record.size_bytes, "sha256": record.sha256}
        for record in records
    }
    manifest = {
        "schema_version": 1,
        "release_tag": release_tag,
        "repository": "bsu-mbe-cbl/knee-mesher",
        "dataset_version": version,
        "files": file_records,
        "datasets": {
            "example": {
                "subjects": example_subjects,
                "files": [record.name for record in example_files],
                "archives": example_archives,
            },
            "all": {
                "files": [record.name for record in records],
                "archives": full_archives,
            },
        },
    }
    manifest_path = output_dir / "dataset-manifest.json"
    _write_manifest(manifest_path, manifest, overwrite)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """Build local release assets without uploading them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--max-archive-bytes", type=int, default=DEFAULT_ARCHIVE_LIMIT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output_dir = args.output or DEFAULT_OUTPUT / (RELEASE_TAG_PREFIX + args.version)
    try:
        manifest_path = build_release(
            args.source,
            args.config,
            output_dir,
            args.version,
            args.max_archive_bytes,
            args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Created local dataset assets and manifest at {manifest_path}")
    print("No files were uploaded to GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
