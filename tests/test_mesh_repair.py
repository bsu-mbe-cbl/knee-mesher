"""Test repair geometry, backend selection, and dependency boundaries."""

import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest
import trimesh

import kneemesher.mesh_repair as repair
from kneemesher.errors._errors import MeshPreparationError, MesherWarning

BACKENDS = [
    "trimesh_open3d",
    pytest.param(
        "meshfix",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("pymeshfix") is None,
            reason="optional MeshFix is not installed",
        ),
    ),
]


@pytest.fixture
def sphere() -> pv.PolyData:
    """Create a small closed surface without attached arrays.

    """
    mesh = pv.Sphere(theta_resolution=16, phi_resolution=16)
    mesh.clear_data()
    return mesh


@pytest.mark.parametrize("backend", BACKENDS)
def test_closed_mesh_contract(sphere, backend):
    """Preserve valid geometry, field data, and the caller's input.

    """
    sphere.field_data["subject"] = ["synthetic"]
    sphere.field_data["label"] = "subject"
    points, faces = sphere.points.copy(), sphere.faces.copy()
    result = repair.repair_mesh(sphere, backend=backend, require_closed=True)
    assert isinstance(result, pv.PolyData)
    assert result is not sphere
    assert result.is_all_triangles and result.is_manifold
    assert np.isfinite(result.points).all()
    assert result.regular_faces.min() >= 0
    assert result.regular_faces.max() < result.n_points
    assert result.volume > 0
    np.testing.assert_allclose(result.bounds, sphere.bounds, rtol=1e-6)
    assert result.area == pytest.approx(sphere.area, rel=1e-6)
    np.testing.assert_array_equal(sphere.points, points)
    np.testing.assert_array_equal(sphere.faces, faces)
    np.testing.assert_array_equal(result.field_data["subject"], ["synthetic"])
    assert result.field_data["label"] == "subject"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("large_hole", [False, True])
def test_fill_boundary(sphere, backend, large_hole):
    """Close missing triangle patches and a larger hemispherical boundary.

    """
    faces = sphere.regular_faces
    if large_hole:
        faces = faces[np.all(sphere.points[faces, 2] < 0.1, axis=1)]
    else:
        faces = faces[1:]
    damaged = pv.PolyData.from_regular_faces(sphere.points, faces)
    result = repair.repair_mesh(damaged, backend=backend, require_closed=True)
    assert result.is_manifold
    assert result.volume > 0


def test_duplicate_degenerate_and_inverted_geometry(sphere):
    """Remove redundant geometry and correct inverted normals.

    """
    vertices = np.vstack([sphere.points, sphere.points[0], [[10, 10, 10]]])
    faces = sphere.regular_faces[:, ::-1].copy()
    faces[faces == 0] = sphere.n_points
    faces = np.vstack([faces, faces[0], [0, 0, 0]])
    result = repair.repair_mesh(
        trimesh.Trimesh(vertices, faces, process=False),
        backend="trimesh_open3d",
        require_closed=True,
    )
    assert result.volume > 0
    assert result.n_points == sphere.n_points
    assert result.n_cells == sphere.n_cells


def test_largest_area_component(sphere):
    """Keep the largest component rather than preserving disconnected solids.

    """
    other = sphere.copy()
    other.points = other.points * 0.1 + [5, 0, 0]
    result = repair.repair_mesh(
        sphere.merge(other), backend="trimesh_open3d", require_closed=True
    )
    np.testing.assert_allclose(result.bounds, sphere.bounds)


def test_drop_arrays_with_warning(sphere):
    """Warn rather than attach arrays to unrelated output indices.

    """
    sphere.point_data["ids"] = np.arange(sphere.n_points)
    sphere.cell_data["labels"] = np.zeros(sphere.n_cells)
    with pytest.warns(MesherWarning, match="drops point/cell arrays"):
        result = repair.repair_mesh(sphere, backend="trimesh_open3d")
    assert not result.point_data and not result.cell_data
    assert "ids" in sphere.point_data


@pytest.mark.parametrize("backend", BACKENDS)
def test_volume_input(sphere, backend):
    """Extract a triangular boundary from volume-dataset inputs.

    """
    result = repair.repair_mesh(
        sphere.cast_to_unstructured_grid(), backend=backend, require_closed=True
    )
    assert result.is_manifold


@pytest.mark.parametrize("bad_input", ["empty", "nan", "index", "pv_index", "degenerate"])
def test_invalid_geometry(sphere, bad_input):
    """Reject invalid inputs before passing them to native repair algorithms.

    """
    if bad_input == "empty":
        mesh = pv.PolyData()
    elif bad_input == "nan":
        mesh = sphere
        mesh.points[0] = np.nan
    elif bad_input == "index":
        mesh = trimesh.Trimesh(sphere.points, [[0, 1, sphere.n_points]], process=False)
    elif bad_input == "pv_index":
        mesh = pv.PolyData.from_regular_faces(
            sphere.points, np.array([[0, 1, sphere.n_points]])
        )
    else:
        mesh = trimesh.Trimesh(sphere.points, [[0, 0, 0]], process=False)
    with pytest.raises(MeshPreparationError):
        repair.repair_mesh(mesh, backend="trimesh_open3d")


def test_invalid_backend_and_type(sphere):
    """Reject unsupported backend names and input types explicitly.

    """
    with pytest.raises(ValueError, match="Unknown mesh repair backend"):
        repair.repair_mesh(sphere, backend="invalid")
    with pytest.raises(TypeError, match="PyVista dataset or Trimesh"):
        repair.repair_mesh("mesh.stl")


def test_missing_meshfix_selection(sphere, monkeypatch):
    """Fall back only in auto mode when the top-level optional package is absent.

    """
    original_import = repair.importlib.import_module

    def missing_meshfix(name, *args, **kwargs):
        if name == "pymeshfix":
            raise ModuleNotFoundError("missing pymeshfix", name="pymeshfix")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(repair.importlib, "import_module", missing_meshfix)
    assert repair.repair_mesh(sphere, backend="auto").is_manifold
    with pytest.raises(ModuleNotFoundError, match="uv sync --locked --extra meshfix"):
        repair.repair_mesh(sphere, backend="meshfix")
    assert repair.repair_mesh(sphere, backend="trimesh_open3d").is_manifold


@pytest.mark.parametrize(
    "error",
    [ModuleNotFoundError("missing native module", name="native"), OSError("ABI")],
)
def test_broken_meshfix_does_not_fall_back(sphere, monkeypatch, error):
    """Propagate a broken installed backend instead of changing algorithms.

    """

    def broken_import(name):
        raise error

    monkeypatch.setattr(repair.importlib, "import_module", broken_import)
    with pytest.raises(type(error), match=str(error)):
        repair.repair_mesh(sphere, backend="auto")


def test_meshfix_mutation_and_runtime_error(sphere, monkeypatch):
    """Read the mutated repair object and propagate runtime repair failures.

    """

    class FakeMeshFix:
        def __init__(self, points, faces, *, verbose):
            assert verbose
            self.mesh = sphere.copy()

        def repair(self, *, joincomp, remove_smallest_components):
            assert not joincomp and remove_smallest_components
            return None

    module = type("FakeModule", (), {"MeshFix": FakeMeshFix})
    monkeypatch.setattr(repair.importlib, "import_module", lambda name: module)
    assert repair.repair_mesh(sphere, verbose=True).is_manifold

    def failing_repair(self, **kwargs):
        raise RuntimeError("repair failed")

    monkeypatch.setattr(FakeMeshFix, "repair", failing_repair)
    with pytest.raises(RuntimeError, match="repair failed"):
        repair.repair_mesh(sphere, verbose=True)


def test_incomplete_repair_strict_and_warning(monkeypatch):
    """Distinguish a best-effort open result from a required closed result.

    """
    plane = pv.Plane().triangulate()
    plane.clear_data()
    monkeypatch.setattr(repair, "_repair_trimesh_open3d", lambda mesh: mesh)
    with pytest.warns(MesherWarning, match="boundary/non-manifold edges"):
        repair.repair_mesh(plane, backend="trimesh_open3d")
    with pytest.raises(MeshPreparationError, match="boundary/non-manifold edges"):
        repair.repair_mesh(plane, backend="trimesh_open3d", require_closed=True)


def test_detect_self_intersections(sphere):
    """Report intersections without claiming they are repaired.

    """
    sphere.points[0] = [0.05, 0, -0.75]
    with pytest.warns(MesherWarning, match="self-intersecting triangle pairs"):
        result = repair.repair_mesh(
            sphere, backend="trimesh_open3d", require_closed=True
        )
    assert result.is_manifold


def test_imports_without_meshfix():
    """Import affected application modules in a process that blocks MeshFix.

    """
    script = """
import sys
from importlib.abc import MetaPathFinder
class BlockMeshFix(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pymeshfix" or fullname.startswith("pymeshfix."):
            raise ModuleNotFoundError("blocked optional dependency", name="pymeshfix")
sys.meta_path.insert(0, BlockMeshFix())
import kneemesher.utilities
import kneemesher.preprocessing
import kneemesher.reconstruction
import kneemesher.pv_helpers
import kneemesher.core_api
import pyvista as pv
from kneemesher.mesh_repair import repair_mesh
mesh = pv.Sphere()
mesh.clear_data()
assert repair_mesh(mesh, require_closed=True).is_manifold
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)


RECONSTRUCTIONS = (
    Path(__file__).parent / "resources/test_reconstruction/sag_3d_dess_right"
)
UNCLOSED_ALTERNATIVE_FIXTURES = {
    "9028904_M00_femur_cartilage.stl",
    "9028904_M00_patella_cartilage.stl",
    "9028904_M00_tibia_cartilage_lateral.stl",
    "9028904_M00_tibia_cartilage_medial.stl",
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("filename", sorted(RECONSTRUCTIONS.glob("*.stl")))
def test_anatomical_reconstruction(filename, backend):
    """Check real surfaces without requiring identical backend triangulations.

    """
    source = pv.read(filename)
    source.clear_data()
    requires_failure = (
        backend == "trimesh_open3d" and filename.name in UNCLOSED_ALTERNATIVE_FIXTURES
    )
    if requires_failure:
        with pytest.raises(MeshPreparationError, match="boundary/non-manifold edges"):
            repair.repair_mesh(source, backend=backend, require_closed=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MesherWarning)
        result = repair.repair_mesh(
            source, backend=backend, require_closed=not requires_failure
        )
    assert result.n_points and result.n_cells
    assert np.isfinite(result.points).all()
    if requires_failure:
        assert any(
            "boundary/non-manifold edges" in str(item.message) for item in caught
        )
    else:
        assert result.is_manifold and result.volume > 0
    if backend == "trimesh_open3d" and not requires_failure:
        np.testing.assert_allclose(result.bounds, source.bounds, rtol=1e-6)
        assert result.area / source.area == pytest.approx(1, rel=0.01)
    # These are observed measurements, not clinical acceptance thresholds.
    print(
        filename.name,
        backend,
        {
            "area_ratio": result.area / source.area,
            "volume_ratio": result.volume / source.volume,
            "bounds_delta": float(
                np.max(np.abs(np.asarray(result.bounds) - np.asarray(source.bounds)))
            ),
            "warnings": [str(item.message) for item in caught],
        },
    )


@pytest.mark.parametrize("vertex_only", [False, True])
def test_non_manifold_diagnostics(monkeypatch, vertex_only):
    """Reject remaining non-manifold vertices and edges in strict mode.

    """
    a = trimesh.creation.icosphere(subdivisions=0)
    if vertex_only:
        b = a.copy()
        b.vertices = -b.vertices + 2 * a.vertices[0]
        joined = trimesh.util.concatenate([a, b])
        joined.merge_vertices()
    else:
        joined = trimesh.Trimesh(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            process=False,
        )
    monkeypatch.setattr(repair, "_repair_trimesh_open3d", lambda mesh: mesh)
    with pytest.raises(MeshPreparationError, match="non-manifold"):
        repair.repair_mesh(joined, backend="trimesh_open3d", require_closed=True)


def test_component_area_tie(sphere):
    """Keep the first encountered component when component areas tie.

    """
    other = sphere.copy()
    other.points += [5, 0, 0]
    mesh = trimesh.util.concatenate(
        [repair.trimesh_wrap(sphere), repair.trimesh_wrap(other)]
    )
    result = repair.repair_mesh(mesh, backend="trimesh_open3d", require_closed=True)
    np.testing.assert_allclose(result.bounds, sphere.bounds)


@pytest.mark.parametrize("scale", [0.001, 1000])
def test_hole_filling_scales_with_input(sphere, scale):
    """Close the same boundary at different coordinate scales.

    """
    faces = sphere.regular_faces
    faces = faces[np.all(sphere.points[faces, 2] < 0.1, axis=1)]
    damaged = pv.PolyData.from_regular_faces(sphere.points * scale, faces)
    result = repair.repair_mesh(
        damaged, backend="trimesh_open3d", require_closed=True
    )
    assert result.is_manifold and result.volume > 0
