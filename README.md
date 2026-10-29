# Knee Mesher

Reconstruct knee tissue surfaces from HDF5 segmentations and export bone,
articular cartilage, and meniscus meshes to Abaqus INP files.

## Installation

Install Python 3.14 and sync the project environment from the repository root:

```shell
uv python install 3.14
uv sync --locked --python 3.14 --managed-python
```

uv creates `.venv` and installs the project in editable mode. Select that
interpreter in VS Code. uv is the supported installation method; keep `uv.lock`
under version control. MeshFix remains optional:

```shell
uv sync --locked --extra meshfix
```

Run commands in this environment without activating it:

```shell
uv run --locked python --version
uv run --locked pytest
```

Include `--extra meshfix` in `uv run` commands when using that backend, for
example `uv run --locked --extra meshfix pytest`.

`--managed-python` selects uv's Python distribution, including
the headers needed to build native dependencies. Source builds
also require a C++ compiler and OpenMP support; on Debian/Ubuntu, install these
with `sudo apt install build-essential libgomp1`. Other platforms need their
corresponding compiler and OpenMP packages.

Use `uv sync --locked --python 3.14 --managed-python` to install the committed
lockfile without updating it. Add `--extra meshfix` when needed. A sync without
that extra removes MeshFix from the uv-managed environment.

The base repair backend uses Trimesh and Open3D. See the
[repair guide](docs/user-guide.md#mesh-repair) for limitations, explicit backend
selection, and licensing considerations.

## Licensing

### Core software

Project-authored code is licensed under the
**[PolyForm Noncommercial License 1.0.0](docs/license.md)**. Copyright (c) 2025
Kalin Gibbons. Third-party code and dependencies retain their own licenses.

The license permits noncommercial purposes and specified personal uses without
anticipated commercial application. It also expressly permits use by educational
institutions, public research organizations, and the other organizations listed
in its Noncommercial Organizations section, regardless of funding source.
The full license terms control; this summary does not add restrictions or grants.

Commercial uses outside those permissions, including commercial corporate R&D,
product prototyping, or consulting, require a separate agreement. Contact
[Kalin Gibbons](mailto:jointmodels@protonmail.com) to discuss commercial
licensing. Any agreement covers only rights the project licensor can grant,
not rights in third-party software. PolyForm Noncommercial is source-available,
not an OSI-approved open-source license.

### Optional third-party dependencies

The optional `meshfix` extra installs **PyMeshFix** for surface repair and hole
filling. Its implementation is not vendored in this project's source package;
it is installed separately and imported on demand. With `repair_backend="auto"`,
an installed PyMeshFix is selected automatically.

[PyMeshFix's license](https://github.com/pyvista/pymeshfix/blob/v0.18.1/LICENSE)
is GNU GPLv3. Its bundled MeshFix code has
[upstream licensing statements](https://github.com/pyvista/pymeshfix/blob/v0.18.1/README.rst#copyright)
describing GPLv3-or-later and a separate commercial licensing route involving
the authors and IMATI-GE/CNR. The upstream README also states that commercial
use requires a licensing contract; review these statements with the upstream
rights holders rather than treating this summary as permission.

GPLv3 itself permits commercial use and does not impose a noncommercial-only
restriction. Its source, notice, and licensing obligations depend on how covered
software is modified and conveyed. Conditional imports and separate installation
are technical boundaries, not a finding of license compatibility or an exemption
from copyleft obligations. A combined work subject to GPLv3 cannot simply carry
PolyForm's additional noncommercial restriction.

Users and redistributors, including holders of a separate project commercial
license, must assess and satisfy the licenses of every enabled dependency.
This project does not grant third-party permissions or guarantee that a
particular combination or distribution is compliant. Seek appropriate upstream
permissions and legal advice where needed. A MeshFix agreement does not
automatically cover the PyMeshFix wrapper or its other components.

To avoid selecting MeshFix, omit the extra and explicitly choose
`repair_backend="trimesh_open3d"`; those dependencies still have their own
licenses. See the [license documentation](docs/license.md) and
[repair guide](docs/user-guide.md#mesh-repair).

## Sample Pipeline

Both configurations select subjects `9056363`, `9068453`, `9087863`, `9094865`,
and `9102858`, including every available side and timepoint. Subject IDs select
the cohort; a file-count cap does not limit it.

Provide local HDF5 inputs under
`data/segmentations/manual-segmentations-2023`. Each file contains a `masks`
group with tissue arrays; `images` is optional. Inputs are not distributed with
the repository. Review voxel spacing, mask names, and orientation in the
[reconstruction config](https://github.com/bsu-mbe-cbl/knee-mesher/blob/main/configs/00-kg-reconstruct-knee-geometries.toml).

```shell
uv run --locked python notebooks/meshing_pipeline/mesh_00_kg_reconstruct_knee_geometries.py
uv run --locked python notebooks/meshing_pipeline/mesh_01_kg_export_knee_meshes.py
```

Reconstruction writes STLs and screenshots to
`data/reconstructions/manual-segmentations-2023`. Reconstruction logs and Python
warnings append to `logging/reconstruction.log`. Export reads that directory and
writes INP files under `exports/sample`, with logs under `logging`.
Review tissue choices and meshing settings in the
[export config](https://github.com/bsu-mbe-cbl/knee-mesher/blob/main/configs/01-kg-export-knee-meshes.toml).
Keep the same subject list in both configurations.

The sample is not a validated success cohort. Inspect per-tissue results and geometry
quality; a closed surface or a completed process does not establish a valid mesh.

## Downloading Segmentation Data

The planned public GitHub release
`data-manual-segmentations-2023-v1` will provide the five-subject example cohort
and the full HDF5 dataset once published. Install Kneemesher, then download the
example cohort:

```shell
kneemesher-data download-example
```

By default, the command verifies and extracts HDF5 files under
`data/segmentations/manual-segmentations-2023`. To download the complete dataset,
use `kneemesher-data download-all`. The full dataset downloads as multiple
archives; each archive is strictly smaller than 2 GiB. Downloaded archives are
cached separately from the package.

See the [dataset release guide](docs/datasets.md) for download options and
maintainer release instructions.

## Project Organization

```shell
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE                 <- PolyForm Noncommercial License 1.0.0.
├── README.md               <- The top-level README for developers.
├── configs                 <- Reconstruction and export configurations.
├── data
│   ├── segmentations       <- HDF5 segmentation inputs.
│   └── reconstructions     <- Reconstructed tissue surfaces.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── exports                 <- Exported knee meshes
├── notebooks/meshing_pipeline <- The two supported pipeline scripts.
├── pyproject.toml          <- Metadata, dependencies, Hatchling, pytest, coverage, and Ruff.
├── .python-version         <- Python version used by uv.
├── uv.lock                 <- Locked uv dependencies.
├── references              <- Data dictionaries, manuals, and all other materials.
├── src
│   └── kneemesher          <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

See the [contributor guide](docs/contributing.md) for development setup,
dependency management, testing, package builds, and documentation builds.
