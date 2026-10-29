"""Download public Kneemesher example and segmentation datasets."""

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_REPOSITORY = "bsu-mbe-cbl/knee-mesher"
DEFAULT_RELEASE_TAG = "data-manual-segmentations-2023-v1"
DEFAULT_OUTPUT = Path("data/segmentations/manual-segmentations-2023")
MANIFEST_NAME = "dataset-manifest.json"
_DOWNLOAD_BLOCK_SIZE = 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class DatasetDownloadError(RuntimeError):
    """Report a dataset release or download validation error."""


def _read_json_url(url: str) -> dict[str, Any]:
    """Read a JSON object from a public HTTPS URL."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "kneemesher-datasets",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise DatasetDownloadError(
            f"Could not read release metadata at {url}"
        ) from error
    if not isinstance(data, dict):
        raise DatasetDownloadError(f"Expected a JSON object at {url}")
    return data


def _load_release_assets(repository: str, tag: str) -> dict[str, dict[str, Any]]:
    """Fetch release assets indexed by filename."""
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise DatasetDownloadError("Repository must have the form owner/name")
    release_url = (
        f"https://api.github.com/repos/{repository}/releases/tags/"
        f"{urllib.parse.quote(tag, safe='')}"
    )
    release = _read_json_url(release_url)
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise DatasetDownloadError(f"Release {tag!r} has no asset list")
    result = {}
    for asset in assets:
        if not isinstance(asset, dict):
            raise DatasetDownloadError(f"Release {tag!r} contains invalid asset data")
        name = asset.get("name")
        url = asset.get("browser_download_url")
        size = asset.get("size")
        if (
            not isinstance(name, str)
            or not isinstance(url, str)
            or type(size) is not int
            or size < 0
            or urllib.parse.urlsplit(url).scheme != "https"
            or urllib.parse.urlsplit(url).hostname != "github.com"
        ):
            raise DatasetDownloadError(f"Release {tag!r} contains invalid asset data")
        if name in result:
            raise DatasetDownloadError(f"Release {tag!r} has duplicate asset {name!r}")
        result[name] = {"url": url, "size_bytes": size}
    return result


def _download_asset(
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
) -> Path:
    """Download an asset and verify its exact size and SHA-256 digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        size, digest = _hash_file(destination)
        if size == expected_size and digest == expected_sha256:
            return destination
        raise DatasetDownloadError(
            f"Cached asset {destination} does not match the release manifest; "
            "remove it and retry"
        )

    temporary = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "kneemesher"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                while block := response.read(_DOWNLOAD_BLOCK_SIZE):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
        if size != expected_size:
            raise DatasetDownloadError(
                f"Downloaded {destination.name} has size {size}; "
                f"expected {expected_size}"
            )
        if digest.hexdigest() != expected_sha256:
            raise DatasetDownloadError(
                f"Downloaded {destination.name} failed SHA-256 verification"
            )
        temporary.replace(destination)
    except (OSError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise DatasetDownloadError(f"Could not download {destination.name}") from error
    except DatasetDownloadError:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _hash_file(path: Path) -> tuple[int, str]:
    """Return a file's byte count and SHA-256 digest."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        while block := file.read(_DOWNLOAD_BLOCK_SIZE):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _validate_filename(filename: object) -> str:
    """Validate a flat HDF5 filename before using it as a destination."""
    if (
        not isinstance(filename, str)
        or PurePosixPath(filename).name != filename
        or "\\" in filename
        or not filename.endswith(".h5")
        or filename in {"", ".", ".."}
    ):
        raise DatasetDownloadError(f"Invalid dataset filename: {filename!r}")
    return filename


def _load_manifest(
    assets: dict[str, dict[str, Any]], release_tag: str, repository: str
) -> dict[str, Any]:
    """Load the manifest from the release's declared asset."""
    asset = assets.get(MANIFEST_NAME)
    if asset is None:
        raise DatasetDownloadError(f"Release has no {MANIFEST_NAME} asset")
    manifest = _read_json_url(asset["url"])
    if manifest.get("schema_version") != 1:
        raise DatasetDownloadError("Unsupported dataset manifest schema")
    if not isinstance(manifest.get("release_tag"), str):
        raise DatasetDownloadError("Dataset manifest has no release tag")
    if manifest.get("release_tag") != release_tag:
        raise DatasetDownloadError("Manifest release tag does not match the request")
    if manifest.get("repository") != repository:
        raise DatasetDownloadError("Manifest repository does not match the request")
    return manifest


def _dataset_archives(
    manifest: dict[str, Any],
    dataset: str,
    release_assets: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate and return the selected dataset's archives and file records."""
    datasets = manifest.get("datasets")
    file_records = manifest.get("files")
    if not isinstance(datasets, dict) or not isinstance(file_records, dict):
        raise DatasetDownloadError("Dataset manifest is missing datasets or files")
    dataset_record = datasets.get(dataset)
    if not isinstance(dataset_record, dict):
        raise DatasetDownloadError(f"Release does not contain dataset {dataset!r}")
    archives = dataset_record.get("archives")
    filenames = dataset_record.get("files")
    if not isinstance(archives, list) or not isinstance(filenames, list):
        raise DatasetDownloadError(f"Dataset {dataset!r} has invalid archive records")

    validated_files = {}
    for item in filenames:
        filename = _validate_filename(item)
        record = file_records.get(filename)
        if not isinstance(record, dict):
            raise DatasetDownloadError(f"Manifest has no record for {filename}")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise DatasetDownloadError(f"Invalid file record for {filename}")
        if filename in validated_files:
            raise DatasetDownloadError(f"Duplicate file record for {filename}")
        validated_files[filename] = {"size_bytes": size, "sha256": digest}

    archive_names = set()
    archived_files = []
    for archive in archives:
        if not isinstance(archive, dict):
            raise DatasetDownloadError(f"Dataset {dataset!r} has an invalid archive")
        name = archive.get("name")
        size = archive.get("size_bytes")
        digest = archive.get("sha256")
        members = archive.get("files")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.zip", name) is None
            or type(size) is not int
            or not 0 < size < 2 * 1024**3
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or not isinstance(members, list)
        ):
            raise DatasetDownloadError(
                f"Dataset {dataset!r} has invalid archive metadata"
            )
        asset = release_assets.get(name)
        if asset is None or asset["size_bytes"] != size:
            raise DatasetDownloadError(
                f"Release asset {name!r} is missing or has wrong size"
            )
        if name in archive_names:
            raise DatasetDownloadError(f"Duplicate archive record for {name}")
        archive_names.add(name)
        for member in members:
            filename = _validate_filename(member)
            if filename not in validated_files:
                raise DatasetDownloadError(
                    f"Archive references unknown file {filename}"
                )
            archived_files.append(filename)
        archive["url"] = asset["url"]
    if len(archived_files) != len(set(archived_files)):
        raise DatasetDownloadError(f"Dataset {dataset!r} archives repeat files")
    if set(archived_files) != set(validated_files):
        raise DatasetDownloadError(
            f"Dataset {dataset!r} archive file list is incomplete"
        )
    return archives, validated_files


def _extract_archive(
    archive_path: Path,
    archive: dict[str, Any],
    file_records: dict[str, dict[str, Any]],
    output_dir: Path,
    overwrite: bool,
) -> list[Path]:
    """Verify and extract flat HDF5 files without trusting archive paths."""
    expected_members = set(archive["files"])
    extracted = []
    try:
        with zipfile.ZipFile(archive_path) as zipped:
            infos = zipped.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != expected_members:
                raise DatasetDownloadError(
                    f"Archive {archive['name']} has an unexpected file list"
                )
            for info in infos:
                filename = _validate_filename(info.filename)
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_ISLNK(mode):
                    raise DatasetDownloadError(
                        f"Archive {archive['name']} contains a non-file member"
                    )
                record = file_records[filename]
                if info.file_size != record["size_bytes"]:
                    raise DatasetDownloadError(
                        f"Archive size for {filename} does not match the manifest"
                    )
                destination = output_dir / filename
                if destination.exists():
                    size, digest = _hash_file(destination)
                    if size == record["size_bytes"] and digest == record["sha256"]:
                        extracted.append(destination)
                        continue
                    if not overwrite:
                        raise DatasetDownloadError(
                            f"{destination} exists and differs from the dataset; "
                            "use --overwrite to replace it"
                        )
                temporary = destination.with_name(destination.name + ".part")
                digest = hashlib.sha256()
                size = 0
                try:
                    with zipped.open(info) as source, temporary.open("wb") as target:
                        while block := source.read(_DOWNLOAD_BLOCK_SIZE):
                            if size + len(block) > record["size_bytes"]:
                                raise DatasetDownloadError(
                                    f"Extracted {filename} exceeds its manifest size"
                                )
                            target.write(block)
                            digest.update(block)
                            size += len(block)
                    if (
                        size != record["size_bytes"]
                        or digest.hexdigest() != record["sha256"]
                    ):
                        raise DatasetDownloadError(
                            f"Extracted {filename} failed size or SHA-256 verification"
                        )
                    temporary.replace(destination)
                except (OSError, zipfile.BadZipFile) as error:
                    temporary.unlink(missing_ok=True)
                    raise DatasetDownloadError(
                        f"Could not extract {filename} from {archive['name']}"
                    ) from error
                except DatasetDownloadError:
                    temporary.unlink(missing_ok=True)
                    raise
                extracted.append(destination)
    except (OSError, zipfile.BadZipFile) as error:
        raise DatasetDownloadError(
            f"Could not read archive {archive['name']}"
        ) from error
    return extracted


def download_dataset(
    dataset: str,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    repository: str = DEFAULT_REPOSITORY,
    release_tag: str = DEFAULT_RELEASE_TAG,
    cache_dir: Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download and verify one dataset from a public GitHub release."""
    if dataset not in {"example", "all"}:
        raise ValueError("dataset must be 'example' or 'all'")
    if cache_dir is None:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = cache_root / "kneemesher" / "datasets" / repository / release_tag
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    release_assets = _load_release_assets(repository, release_tag)
    manifest = _load_manifest(release_assets, release_tag, repository)
    archives, file_records = _dataset_archives(manifest, dataset, release_assets)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    for archive in archives:
        filename = archive["name"]
        digest = archive["sha256"]
        if _SHA256_RE.fullmatch(digest) is None:
            raise DatasetDownloadError(f"Invalid SHA-256 for archive {filename}")
        archive_path = _download_asset(
            archive["url"],
            cache_dir / filename,
            archive["size_bytes"],
            digest,
        )
        extracted.extend(
            _extract_archive(
                archive_path,
                archive,
                file_records,
                output_dir,
                overwrite,
            )
        )
    return extracted


def main(argv: list[str] | None = None) -> int:
    """Run the dataset download command-line interface."""
    parser = argparse.ArgumentParser(
        description="Download and verify Kneemesher datasets from GitHub Releases."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, dataset in (
        ("download-example", "example"),
        ("download-all", "all"),
    ):
        subparser = subparsers.add_parser(command)
        subparser.set_defaults(dataset=dataset)
        subparser.add_argument("--repository", default=DEFAULT_REPOSITORY)
        subparser.add_argument("--tag", default=DEFAULT_RELEASE_TAG)
        subparser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        subparser.add_argument("--cache-dir", type=Path)
        subparser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = download_dataset(
            args.dataset,
            args.output,
            repository=args.repository,
            release_tag=args.tag,
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
        )
    except (DatasetDownloadError, OSError) as error:
        print(f"Dataset download failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified {len(paths)} HDF5 files in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
