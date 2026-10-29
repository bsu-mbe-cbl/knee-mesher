# Dataset downloads and releases

Kneemesher distributes HDF5 segmentation datasets as assets on public GitHub
Releases, separately from the PyPI package and source repository. The planned
release `data-manual-segmentations-2023-v1` contains:

- The five-subject example cohort selected by
  `configs/00-kg-reconstruct-knee-geometries.toml`.
- All available HDF5 files, divided into ZIP archives that are each strictly
  smaller than 2 GiB.
- `dataset-manifest.json`, which records file and archive sizes and SHA-256
  digests.

The release is not published yet. Download commands work after its assets are
attached to the public repository release.

## Download data

Install Kneemesher, then run one of these commands from the directory where the
data should be stored:

```shell
kneemesher-data download-example
kneemesher-data download-all
```

The example command downloads the configured five-subject cohort. The full
dataset command downloads and extracts all archives. Both commands use
`data/segmentations/manual-segmentations-2023` as the default output path and
cache verified ZIP archives under the user's cache directory.

Use `--output PATH` to select another destination, `--tag TAG` to select a data
release, or `--repository OWNER/NAME` to use another public repository. Existing
matching files are kept. The command rejects mismatched existing files unless
`--overwrite` is provided.

## Prepare a data release

The source directory and the example cohort are taken from the repository's
segmentation folder and reconstruction config. Build assets locally with:

```shell
uv run --locked python tools/build_dataset_release.py --version v1
```

The command writes the example ZIP, the full-dataset ZIP chunks, and
`dataset-manifest.json` under
`dist/dataset-releases/data-manual-segmentations-2023-v1/`. Each version uses a
separate output directory. The default maximum archive size is 1,900,000,000
bytes, below GitHub's per-asset limit of 2 GiB.
The builder hashes each input and archive, uses stable file order and ZIP
timestamps, and does not upload or publish assets. Review all generated files
and confirm that the manifest lists the expected cohort and complete dataset.

After review, attach the generated files to a GitHub Release using the matching
tag. For example, from the repository root:

```shell
gh release create data-manual-segmentations-2023-v1 \
  dist/dataset-releases/data-manual-segmentations-2023-v1/* \
  --repo bsu-mbe-cbl/knee-mesher \
  --title "Manual segmentations 2023, v1" \
  --notes "Versioned HDF5 segmentation datasets and SHA-256 manifest."
```

The `--repo` option selects the public repository explicitly, regardless of the
local Git remotes or GitHub CLI default repository.

Do not reuse a published data tag for changed files. Increment the data version,
rebuild the assets, and publish a new release. The downloader verifies archive
hashes before extraction and verifies each HDF5 file's size and hash after
extraction. Test a new release with a clean download directory before announcing
it.

GitHub Release assets have a per-file size limit and no total release-size
limit; keep every uploaded archive strictly below 2 GiB. The data manifest is
itself a release asset. Publicly hosted datasets are available to anyone with
access to this repository.
