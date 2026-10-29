# Contributing

Report bugs and discuss larger changes in the
[issue tracker](https://github.com/bsu-mbe-cbl/knee-mesher/issues).
Include the operating system, Python version, reproduction steps, and traceback.
Use synthetic geometry in reproductions; do not upload private datasets.

## Development environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone
the repository or your fork:

```shell
git clone https://github.com/bsu-mbe-cbl/knee-mesher.git
cd kneemesher
uv python install 3.14
uv sync --locked --python 3.14 --managed-python
uv run --locked pre-commit install
```

uv manages `.venv` and installs the project in editable mode. Select
`.venv/bin/python` in VS Code, or `.venv/Scripts/python.exe` on Windows.
Use `uv run --locked` to run project commands; activation is optional.
Native dependency builds require a C++ compiler and OpenMP support. On
Debian/Ubuntu, install `build-essential` and `libgomp1`.

The default `dev` dependency group includes pytest, coverage, Ruff, IPython,
and pre-commit. Add `--group docs` to sync documentation tools.
PyMeshFix is optional: add `--extra meshfix` to both sync and run commands
when testing that backend. A sync without the extra removes it.
Review the [dependency licensing requirements](docs/license.md) before enabling it.

## Project layout

- `src/kneemesher/reconstruction/` builds surfaces from segmentations.
- `src/kneemesher/meshing/` creates and repairs volume elements.
- `src/kneemesher/mesh_repair.py` selects surface-repair backends.
- `src/kneemesher/core_api.py` exposes tissue-mesh operations.
- `notebooks/meshing_pipeline/` contains the reconstruction and export scripts.
- `configs/` contains their sample settings.
- `tools/build_dataset_release.py` builds local GitHub Release data assets.
- `tests/` contains regression tests and geometry fixtures.

## Making changes

Create a branch with `git switch -c my-change`. Keep changes focused, add
regression tests for changed behavior, and document public APIs with Google-style
docstrings. Use Python 3.14 syntax, type annotations, and Ruff formatting.
Do not commit datasets, generated meshes, build output, or credentials.

Run checks on the files you change:

```shell
uv run --locked ruff check path/to/changed.py
uv run --locked ruff format --check path/to/changed.py
uv run --locked pre-commit run --files path/to/changed.py
uv run --locked pytest tests/test_mesh_repair.py
```

Ruff and formatting hooks can modify files. Review their changes before staging.
Avoid unrelated formatting changes.
Run the relevant tests first, then the suite when changes affect shared behavior:

```shell
uv run --locked pytest
```

pytest collects `tests/` and reports branch coverage by default. Use
`--no-cov` when debugging. VS Code's Python Test Explorer can run the same tests
with the project interpreter.

## Dependencies and builds

`pyproject.toml` defines package metadata, runtime requirements, the optional
`meshfix` extra, development/documentation groups, pytest, coverage, and Ruff.
Use uv to change dependencies and commit both the manifest and lockfile:

```shell
uv add package-name
uv add --dev development-tool
uv add --group docs documentation-tool
uv add --optional meshfix optional-repair-dependency
uv lock --check
```

Hatchling builds the package; hatch-vcs derives its version from Git tags using
the `no-guess-dev` scheme. Tags must contain PEP 440 versions. Untagged
checkouts receive a development version, and dirty checkouts can include a local
version suffix. Keep Git history and tags available when building a checkout.
A generated source distribution also builds without Git metadata.

```shell
uv build
```

The wheel contains only `kneemesher`, not datasets, MATLAB references, or
PyMeshFix. The source distribution also includes tests, pipeline scripts, sample
configs, the local data release builder, and documentation.
The downloadable HDF5 data is published separately as
GitHub Release assets; see [the dataset release guide](docs/datasets.md).
Put contributor tools in development or documentation dependency groups.

Keep local reference material out of releases. Preserve the `references/`
exclusions in both build targets and check package contents before release.

## Documentation

Documentation uses Sphinx and MyST Markdown, with reStructuredText for generated
API pages. Build it from the repository root:

```shell
uv run --locked --group docs sphinx-build -b html docs docs/_build/html
uv run --locked python -m http.server --directory docs/_build/html
```

Open `http://localhost:8000` for a preview. Sphinx generates API pages from
`src/kneemesher` during the build; do not commit generated pages.
Read the Docs uses the same locked environment with the `docs` group.

## Pull requests and licensing

Explain the change, link related issues, and list the checks you run. Note any
remaining failures or warnings. Submit a draft pull request when feedback is
needed before the change is ready.

Project-authored code uses the
[PolyForm Noncommercial License 1.0.0](docs/license.md).
Submit only code you have permission to contribute, preserve third-party notices,
and identify any new dependencies and their licenses. Read the
[licensing documentation](docs/license.md) for optional-dependency boundaries.
