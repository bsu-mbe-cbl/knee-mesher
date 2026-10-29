"""Verify geometry and metadata contracts on the current PyVista API."""

import ast
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from kneemesher.errors._errors import MesherWarning
from kneemesher.mesh_repair import repair_mesh
from kneemesher.meshing.cells import HexElement
from kneemesher.pv_helpers import PolyData, add_mesh_clip_box
from kneemesher.utilities import _utilities as utilities


@pytest.fixture
def labeled_grid() -> pv.UnstructuredGrid:
    """Create a volume grid with independently verifiable source IDs."""
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    grid.point_data["source_pid"] = np.arange(grid.n_points)
    grid.cell_data["source_cid"] = np.arange(grid.n_cells)
    grid.field_data["subject"] = ["synthetic"]
    return grid


def test_extraction_selects_new_algorithm_explicitly() -> None:
    """Require every source extraction call to select its algorithm."""
    source = Path(__file__).parents[1] / "src"
    for path in source.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "extract_surface"
            ):
                continue
            algorithm = next(
                (kw.value for kw in node.keywords if kw.arg == "algorithm"), None
            )
            if algorithm is None:
                # ugrid_to_polydata forwards its explicit default in kwargs.
                assert path.name == "pv_helpers.py" and any(
                    kw.arg is None for kw in node.keywords
                )
                continue
            assert isinstance(algorithm, ast.Constant) and algorithm.value is None, (
                f"{path}:{node.lineno} does not select the new default algorithm"
            )


def test_edge_extraction_keeps_connectivity_and_source_arrays(labeled_grid) -> None:
    """Keep the same edge IDs and arrays when omitting the ignored legacy argument."""
    # Include an unused point to verify that edge IDs remain in input point order.
    grid = pv.UnstructuredGrid(
        labeled_grid.cells,
        labeled_grid.celltypes,
        np.vstack((labeled_grid.points, [9.0, 9.0, 9.0])),
    )
    grid.point_data["source_pid"] = np.arange(grid.n_points)
    grid.cell_data["source_cid"] = np.arange(grid.n_cells)
    grid.field_data["subject"] = ["synthetic"]
    with pytest.warns(pv.PyVistaDeprecationWarning, match="use_all_points"):
        previous = grid.extract_all_edges(use_all_points=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pv.PyVistaDeprecationWarning)
        current = grid.extract_all_edges()
        element = HexElement()
        connectivity = element.nodal_connectivity
    np.testing.assert_array_equal(current.points, previous.points)
    np.testing.assert_array_equal(current.lines, previous.lines)
    np.testing.assert_array_equal(current.points, grid.points)
    for association in ("point_data", "cell_data", "field_data"):
        actual = getattr(current, association)
        expected = getattr(previous, association)
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])
    assert connectivity.shape == (12, 2)
    expected_edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
            [4, 5],
            [5, 6],
            [6, 7],
            [4, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    assert set(map(tuple, np.sort(connectivity, axis=1))) == set(
        map(tuple, expected_edges)
    )


def test_new_surface_algorithm_preserves_source_mappings(
    labeled_grid: pv.UnstructuredGrid,
) -> None:
    """Map reordered surface vertices and faces back to their volume source."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", pv.PyVistaFutureWarning)
        surface = labeled_grid.extract_surface(algorithm=None)
        with pytest.warns(MesherWarning, match="drops point/cell arrays"):
            repaired = repair_mesh(
                labeled_grid, backend="trimesh_open3d", require_closed=True
            )
    np.testing.assert_array_equal(
        surface.points, labeled_grid.points[surface.point_data["vtkOriginalPointIds"]]
    )
    np.testing.assert_array_equal(
        surface.point_data["source_pid"], surface.point_data["vtkOriginalPointIds"]
    )
    np.testing.assert_array_equal(
        surface.cell_data["source_cid"], surface.cell_data["vtkOriginalCellIds"]
    )
    assert surface.n_cells == 24
    assert surface.n_open_edges == 0
    assert surface.volume == pytest.approx(8.0)
    assert repaired.is_manifold
    assert repaired.volume == pytest.approx(8.0)
    np.testing.assert_array_equal(repaired.field_data["subject"], ["synthetic"])


@pytest.mark.parametrize("merge_points", [False, True])
def test_merge_keeps_source_arrays_with_current_order(merge_points: bool) -> None:
    """Keep source IDs associated with coordinates rather than legacy merge offsets."""
    meshes = []
    for index in range(3):
        mesh = pv.Cube(center=(3 * index, 0, 0))
        mesh.point_data["source_pid"] = np.arange(mesh.n_points) + 100 * index
        mesh.cell_data["source_cid"] = np.arange(mesh.n_cells) + 100 * index
        meshes.append(mesh)
    merged = pv.merge(meshes, merge_points=merge_points)
    points = np.concatenate([mesh.points for mesh in meshes])
    pids = np.concatenate([mesh.point_data["source_pid"] for mesh in meshes])
    cids = np.concatenate([mesh.cell_data["source_cid"] for mesh in meshes])
    np.testing.assert_array_equal(merged.points, points)
    np.testing.assert_array_equal(merged.point_data["source_pid"], pids)
    np.testing.assert_array_equal(merged.cell_data["source_cid"], cids)
    for mesh in meshes:
        selected = np.isin(
            merged.point_data["source_pid"], mesh.point_data["source_pid"]
        )
        np.testing.assert_array_equal(merged.points[selected], mesh.points)


def test_merge_duplicate_point_arrays_prioritize_first_mesh() -> None:
    """Make VTK's first-input precedence explicit for coincident-point metadata."""
    first = pv.Cube()
    other = first.copy()
    first.point_data["source_pid"] = np.arange(first.n_points)
    other.point_data["source_pid"] = np.arange(other.n_points) + 100
    merged = pv.merge([first, other], merge_points=True)
    assert merged.n_points == first.n_points
    np.testing.assert_array_equal(merged["source_pid"], first["source_pid"])


@pytest.mark.parametrize("mode", ["any", "all"])
@pytest.mark.parametrize("inplace", [False, True])
def test_modern_point_removal_preserves_ids(mode: str, inplace: bool) -> None:
    """Preserve retained topology and metadata with the mesh-only removal API."""
    mesh = PolyData(pv.Sphere(theta_resolution=8, phi_resolution=8))
    mesh.point_data["source_pid"] = np.arange(mesh.n_points)
    mesh.cell_data["source_cid"] = np.arange(mesh.n_cells)
    points = mesh.points.copy()
    faces = mesh.regular_faces.copy()
    removed = np.zeros(mesh.n_points, dtype=bool)
    removed[faces[0]] = True
    keep_cells = ~(
        removed[faces].any(axis=1) if mode == "any" else removed[faces].all(axis=1)
    )
    expected_cells = np.flatnonzero(keep_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pv.PyVistaDeprecationWarning)
        result = mesh.remove_points(ind=removed, mode=mode, inplace=inplace)
    np.testing.assert_array_equal(result.cell_data["source_cid"], expected_cells)
    np.testing.assert_array_equal(
        result.points, points[result.point_data["source_pid"]]
    )
    np.testing.assert_array_equal(
        result.point_data["vtkOriginalPointIds"], result.point_data["source_pid"]
    )
    assert (result is mesh) == inplace
    if not inplace:
        np.testing.assert_array_equal(mesh.points, points)
        np.testing.assert_array_equal(mesh.regular_faces, faces)


def test_raytracing_reverses_inverted_cartilage_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use in-place face reversal on the raytracing bone-repair path."""
    cartilage = pv.Sphere(theta_resolution=8, phi_resolution=8)
    cartilage.flip_faces(inplace=True)
    cartilage = cartilage.cast_to_unstructured_grid()
    cartilage.point_data["mesh_pids"] = np.arange(cartilage.n_points)
    cartilage.field_data["bone_side_pids"] = np.arange(cartilage.n_points)
    calls = []
    original_flip = pv.PolyData.flip_faces

    def flip(mesh: pv.PolyData, *, inplace: bool = False) -> pv.PolyData:
        """Record the mutation choice while running the actual face reversal."""
        calls.append(inplace)
        return original_flip(mesh, inplace=inplace)

    monkeypatch.setattr(pv.PolyData, "flip_faces", flip)
    bone = pv.Sphere(radius=0.25, theta_resolution=8, phi_resolution=8)
    original = bone.points.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", pv.PyVistaDeprecationWarning)
        result = utilities.fix_protruding_bones_with_raytracing(
            bone, [cartilage], max_displacement=0.0
        )
    assert calls == [True]
    assert result is bone
    np.testing.assert_array_equal(bone.points, original)


@pytest.mark.parametrize("crinkle", [False, True])
@pytest.mark.parametrize("invert", [False, True])
def test_box_widget_callback_updates_geometry(
    labeled_grid: pv.UnstructuredGrid, crinkle: bool, invert: bool
) -> None:
    """Exercise the real offscreen widget and its metadata callback without a GUI."""
    plotter = pv.Plotter(off_screen=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pv.PyVistaDeprecationWarning)
            warnings.simplefilter("error", UserWarning)
            actor = add_mesh_clip_box(
                plotter, labeled_grid, crinkle=crinkle, invert=invert
            )
        assert actor is not None
        result = plotter.widgets.box_clipped_meshes[-1]
        if not invert:
            assert result.n_cells > 0
        assert plotter.clipping_planes.n_blocks == 6
        assert len(plotter.clipped_bounds) == 12
        assert "box_clip_bounds" in result.field_data
        assert np.isin(
            result.cell_data["source_cid"], labeled_grid.cell_data["source_cid"]
        ).all()
        widget = plotter.widgets.box_widgets[-1]
        widget.SetPlaceFactor(1.0)
        widget.PlaceWidget(0.25, 0.75, 0.25, 0.75, 0.25, 0.75)
        widget.InvokeEvent("EndInteractionEvent")
        assert result.n_cells > 0
        assert len(plotter.clipped_bounds) == 12
        np.testing.assert_array_equal(result.field_data["subject"], ["synthetic"])
        np.testing.assert_array_equal(
            result.field_data["box_clip_bounds"], plotter.clipped_bounds
        )
        if crinkle:
            assert result.n_cells == (8 if invert else 1)
            np.testing.assert_array_equal(
                result.cell_data["source_cid"],
                np.arange(8) if invert else [0],
            )
        else:
            # The VTK box clipper retains unused points outside the clipped cells.
            surface = result.extract_surface(algorithm=None)
            expected_bounds = [0.0, 2.0] * 3 if invert else [0.25, 0.75] * 3
            np.testing.assert_allclose(surface.bounds, expected_bounds)
            assert surface.volume == pytest.approx(8.0 - 0.125 if invert else 0.125)
    finally:
        plotter.close()
