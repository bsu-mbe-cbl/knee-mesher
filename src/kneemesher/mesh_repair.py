"""Repair triangular surfaces with optional MeshFix or Trimesh and Open3D."""

import importlib
import logging
import warnings
from copy import deepcopy
from typing import Literal

import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh
from vtk.util.numpy_support import vtk_to_numpy

from kneemesher.errors._errors import MeshPreparationError, MesherWarning

RepairBackend = Literal["auto", "meshfix", "trimesh_open3d"]
logger = logging.getLogger(__name__)


def trimesh_wrap(pv_polydata: pv.PolyData) -> trimesh.Trimesh:
    """Convert a PyVista surface to independent triangular geometry.

    Point and cell arrays are not transferred.

    """
    surface = pv_polydata.triangulate()
    return trimesh.Trimesh(
        vertices=surface.points.copy(),
        faces=surface.regular_faces.copy(),
        process=False,
    )


def _validate_geometry(mesh: trimesh.Trimesh) -> None:
    """Reject geometry that cannot be passed safely to native repair code.

    Check connectivity bounds before native libraries dereference indices.

    """
    if (
        mesh.vertices.ndim != 2
        or mesh.vertices.shape[1] != 3
        or mesh.faces.ndim != 2
        or mesh.faces.shape[1] != 3
        or not len(mesh.vertices)
        or not len(mesh.faces)
    ):
        raise MeshPreparationError("Mesh repair requires nonempty triangular geometry.")
    if not np.isfinite(mesh.vertices).all():
        raise MeshPreparationError("Mesh repair requires finite vertex coordinates.")
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise MeshPreparationError("Mesh repair found out-of-range vertex indices.")


def _open3d_mesh(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Copy triangular geometry into an Open3D mesh.

    Open3D stores triangle connectivity as 32-bit integers.

    """
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )


def _clean_geometry(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove duplicate and degenerate geometry and non-manifold edges.

    Removing faces on non-manifold edges can leave new boundary holes.

    """
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    _validate_geometry(mesh)
    surface = _open3d_mesh(mesh)
    surface.remove_duplicated_vertices()
    surface.remove_duplicated_triangles()
    surface.remove_degenerate_triangles()
    surface.remove_non_manifold_edges()
    surface.remove_unreferenced_vertices()
    result = trimesh.Trimesh(
        np.asarray(surface.vertices).copy(),
        np.asarray(surface.triangles).copy(),
        process=False,
    )
    _validate_geometry(result)
    return result


def _repair_trimesh_open3d(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean, orient, and attempt to close a surface without MeshFix.

    Use simple Trimesh hole filling before the CPU Open3D tensor operation.

    """
    mesh = _clean_geometry(mesh)
    components = mesh.split(only_watertight=False, repair=False)
    # max retains the first component when areas tie.
    mesh = max(components, key=lambda component: component.area)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)
    if not mesh.is_watertight:
        # A mesh-relative radius covers all boundaries regardless of input units.
        radius = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
        if not np.isfinite(radius) or radius <= 0:
            raise MeshPreparationError("Mesh repair requires a finite nonzero extent.")
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(
            _open3d_mesh(mesh), vertex_dtype=o3d.core.Dtype.Float64
        )
        filled = tensor_mesh.fill_holes(hole_size=radius).to_legacy()
        mesh = trimesh.Trimesh(
            np.asarray(filled.vertices).copy(),
            np.asarray(filled.triangles).copy(),
            process=False,
        )
    mesh = _clean_geometry(mesh)
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def _repair_meshfix(
    mesh: trimesh.Trimesh, verbose: bool, backend: RepairBackend
) -> trimesh.Trimesh:
    """Load MeshFix on demand, falling back only when the package is absent.

    Errors inside an installed MeshFix package must not change the algorithm.

    """
    try:
        pymeshfix = importlib.import_module("pymeshfix")
    except ModuleNotFoundError as error:
        if error.name != "pymeshfix":
            raise
        if backend == "meshfix":
            raise ModuleNotFoundError(
                "MeshFix is not installed. Run uv sync --locked --extra meshfix.",
                name="pymeshfix",
            ) from error
        logger.info("Repair backend: trimesh_open3d (MeshFix is not installed)")
        return _repair_trimesh_open3d(mesh)
    logger.info("Repair backend: meshfix")
    fixer = pymeshfix.MeshFix(mesh.vertices, mesh.faces, verbose=verbose)
    fixer.repair(joincomp=False, remove_smallest_components=True)
    return trimesh_wrap(fixer.mesh)


def repair_mesh(
    mesh: pv.DataSet | trimesh.Trimesh,
    verbose: bool = False,
    *,
    backend: RepairBackend = "auto",
    require_closed: bool = False,
) -> pv.PolyData:
    """Return repaired geometry without modifying the input.

    The alternative keeps the edge-connected component with the greatest area,
    attempts all hole closures, and does not repair self-intersections.
    Point/cell arrays are dropped with a warning; field data is copied.

    Args:
        mesh: A surface, volume dataset, or Trimesh to repair.
        verbose: Whether MeshFix prints repair progress.
        backend: Select MeshFix, Trimesh + Open3D, or automatic availability.
        require_closed: Raise if boundaries or non-manifold vertices remain.

    Returns:
        An independent triangular PyVista surface.

    Raises:
        MeshPreparationError: If geometry is invalid or required closure fails.
        ModuleNotFoundError: If explicitly selected MeshFix is unavailable.
        ValueError: If the backend name is invalid.

    """
    if backend not in ("auto", "meshfix", "trimesh_open3d"):
        raise ValueError(f"Unknown mesh repair backend: {backend!r}")
    surface = None
    if isinstance(mesh, trimesh.Trimesh):
        triangular = mesh.copy()
    elif isinstance(mesh, pv.DataSet):
        if not np.isfinite(mesh.points).all():
            raise MeshPreparationError(
                "Mesh repair requires finite vertex coordinates."
            )
        surface = (
            mesh
            if isinstance(mesh, pv.PolyData)
            else mesh.extract_surface(algorithm=None)
        )
        # Check indices before VTK triangulation can dereference invalid connectivity.
        if surface.n_faces:
            connectivity = vtk_to_numpy(surface.GetPolys().GetConnectivityArray())
            if connectivity.min() < 0 or connectivity.max() >= surface.n_points:
                raise MeshPreparationError(
                    "Mesh repair found out-of-range vertex indices."
                )
        triangular = trimesh_wrap(surface)
    else:
        raise TypeError("Mesh repair requires a PyVista dataset or Trimesh.")
    _validate_geometry(triangular)
    if backend == "trimesh_open3d":
        logger.info("Repair backend: trimesh_open3d")
        repaired = _repair_trimesh_open3d(triangular)
    else:
        repaired = _repair_meshfix(triangular, verbose, backend)
    _validate_geometry(repaired)
    checked = _open3d_mesh(repaired)
    boundary_edges = len(checked.get_non_manifold_edges(allow_boundary_edges=False))
    bad_vertices = len(checked.get_non_manifold_vertices())
    if boundary_edges or bad_vertices or not repaired.is_winding_consistent:
        message = (
            f"Mesh repair left {boundary_edges} boundary/non-manifold edges and "
            f"{bad_vertices} non-manifold vertices; "
            f"consistent winding={repaired.is_winding_consistent}. "
            "Consider the MeshFix backend or review the input geometry."
        )
        if require_closed:
            raise MeshPreparationError(message)
        warnings.warn(message, MesherWarning, stacklevel=2)
    intersections = len(checked.get_self_intersecting_triangles())
    if intersections:
        warnings.warn(
            f"Mesh repair left {intersections} self-intersecting triangle pairs; "
            "closure does not establish a valid solid.",
            MesherWarning,
            stacklevel=2,
        )
    result = pv.PolyData.from_regular_faces(
        repaired.vertices.copy(), repaired.faces.copy()
    )
    if surface is not None:
        for name, value in mesh.field_data.items():
            result.field_data[name] = deepcopy(value)
        if mesh.point_data or mesh.cell_data:
            warnings.warn(
                "Mesh repair drops point/cell arrays because topology may change.",
                MesherWarning,
                stacklevel=2,
            )
    return result
