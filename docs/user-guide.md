# User Guide

## Dataset releases

Follow the [dataset release guide](datasets.md#prepare-a-data-release) to build
and publish dataset assets. Pass `--repo bsu-mbe-cbl/knee-mesher` to
`gh release create` to select the public repository explicitly instead of relying
on the local Git remotes or GitHub CLI default repository.

## Batch export progress

The export script displays one batch bar with running `exported`, `skipped`, and
`failed` knee counts. Its total includes the configured knee-index slice;
subject inclusion and exclusion filters count as skipped knees.

Each selected knee has a fixed row for every requested tissue, including missing
inputs. Rows show the current operation, retry attempt, and elapsed stage time.
Finite loops report completed rays, slices, regions, or smoothing fronts.
Convergence loops report completed iterations without a percentage. Native
operations such as clustering and smoothing have no known internal work count
and show elapsed time rather than an estimated percentage.

Meshing retries stop when no setting changes remain.

`Mesh ready` means meshing succeeds, not that export finishes. A tissue reaches
`Exported` only after its INP file reaches the final export directory. A knee
counts as exported only when all requested outputs reach that directory.
Preparation, worker, writing, or move failures count as failed knees. Interrupted
knees show cancelled unfinished tissues and do not count as completed.
Temporary export data stays under the current working directory and is removed
on exit.

## Mesh repair

`kneemesher.mesh_repair.repair_mesh` repairs a triangular surface without modifying
the input:

```python
from kneemesher.mesh_repair import repair_mesh

closed = repair_mesh(surface, backend="trimesh_open3d", require_closed=True)
```

The function accepts PyVista datasets and Trimesh objects and returns a new
`pyvista.PolyData`. Volume datasets are converted to their outer surface.
`kneemesher.utilities.repair_mesh` remains available with the same arguments.

### Backend selection

- `auto` (default): use MeshFix when installed, otherwise Trimesh + Open3D.
- `trimesh_open3d`: always use the base implementation.
- `meshfix`: require MeshFix and report installation instructions if it is absent.

Install the base environment using the README uv instructions. To add MeshFix:

```shell
uv sync --locked --extra meshfix
```

Include `--extra meshfix` in subsequent `uv run` commands to keep it installed.
Installing the extra changes automatic selection and may change repaired
geometry. Use an explicit backend for reproducible processing.

Pass `repair_backend="trimesh_open3d"` or `repair_backend="meshfix"` to
`clustered_mesh`, `fix_protruding_bones`, `clamp_triangle_count_and_smooth`,
`prepare`, `prepare_with_bones`, `mesh_rigid_body`, reconstruction functions,
and `PolyData.slice_closed_surface` to select the backend for their repair stages.
The selected backend is logged at INFO level. A broken MeshFix installation or
failed MeshFix repair raises its error; `auto` does not silently switch algorithms.

### Geometry and limitations

The alternative removes duplicate/degenerate geometry, removes faces on
non-manifold edges, fixes winding and normals, and attempts to triangulate all
boundary holes. It keeps the edge-connected component with the greatest surface
area; equal areas keep the first encountered component. This follows the intent
of MeshFix's largest-component default, not an identical selection algorithm.
Components are not joined.

Hole filling uses Trimesh for simple boundaries and Open3D's CPU tensor API for
larger boundaries. Its radius threshold comes from the input bounds rather than
a fixed size in unspecified units. Neither implementation promises unchanged
point/face indices. Field data is copied; point/cell arrays are dropped with a
warning because repaired topology has no reliable correspondence to those arrays.

Self-intersections are detected and reported, but the alternative does not repair
them. A closed mesh is not necessarily a valid intersection-free solid. Face
removal on non-manifold edges can also leave holes, and irregular boundaries may
remain open. `require_closed=True` raises `MeshPreparationError` for remaining
boundary edges, non-manifold vertices, or inconsistent winding. Best-effort calls
warn about these defects instead. Empty or invalid geometry always raises an
error. Application paths that need closed geometry request strict validation.

Repair does not establish clinical validity or equivalent output from the two
backends. Inspect repaired geometry before using it in a simulation.

### Licensing boundary

Project-authored code uses the PolyForm Noncommercial License 1.0.0.
Third-party dependencies retain their own licenses. MeshFix is absent from base
environments and is imported on demand; `auto` selects it when installed.

PyMeshFix uses GNU GPLv3, and its MeshFix implementation has additional upstream
licensing statements and a separate commercial licensing route. GPLv3 itself
permits commercial use. Conditional imports and separate installation do not
establish compatibility between GPL-covered combined works and PolyForm's
noncommercial restriction.

A separate commercial license for this project does not grant third-party
rights. Users and redistributors must assess their dependency licenses and obtain
any necessary upstream permissions. See the
[license documentation](license.md) for the
upstream sources and full disclaimer. Select `repair_backend="trimesh_open3d"`
and omit the extra to avoid selecting MeshFix; the base dependencies still have
their own licenses.
