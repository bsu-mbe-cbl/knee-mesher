"""Verify that mesh-processing entry points use the shared repair contract."""

from types import SimpleNamespace

import numpy as np
import pyvista as pv
import pytest

from kneemesher import preprocessing
from kneemesher.mesh_repair import repair_mesh
from kneemesher.pv_helpers import PolyData
from kneemesher.reconstruction import _reconstruction as reconstruction
from kneemesher.utilities import _utilities as utilities

BACKEND = "trimesh_open3d"


@pytest.fixture
def sphere():
    """Create a small closed input without data arrays.

    """
    mesh = pv.Sphere(theta_resolution=12, phi_resolution=12)
    mesh.clear_data()
    return mesh


def test_reconstruction_consumes_pre_smoothing_repair(sphere, monkeypatch):
    """Pass the actual repaired result to smoothing rather than discarding it.

    """
    repaired = sphere.copy()
    repaired.points *= 2
    calls = []

    def repair(mesh, **kwargs):
        calls.append(kwargs)
        return repaired

    def smooth(mesh, bone, **kwargs):
        assert mesh is repaired
        assert kwargs == {"n_iters": 3}
        return mesh

    monkeypatch.setattr(reconstruction, "repair_mesh", repair)
    monkeypatch.setattr(reconstruction, "smooth_mesh", smooth)
    result = reconstruction.postprocess_mesh(
        sphere,
        "femur_cartilage",
        False,
        False,
        repair_backend=BACKEND,
        n_iters=3,
    )
    assert result is repaired
    assert calls == [{"backend": BACKEND, "require_closed": True}]


def test_reconstruction_bone_stages(sphere, monkeypatch):
    """Use the same explicit backend before/after smoothing and positioning.

    """
    calls = []

    def repair(mesh, **kwargs):
        calls.append(kwargs)
        return repair_mesh(mesh, **kwargs)

    monkeypatch.setattr(reconstruction, "repair_mesh", repair)
    result = reconstruction.postprocess_mesh(
        sphere,
        "patella_bone",
        True,
        False,
        repair_backend=BACKEND,
        n_bone_iters=1,
    )
    result = reconstruction.transform_mesh(
        result, "patella_bone", True, repair_backend=BACKEND
    )
    assert result.is_manifold
    assert calls == [{"backend": BACKEND, "require_closed": True}] * 3


def test_clustering_repairs_all_three_stages(sphere, monkeypatch):
    """Keep input, subdivided, and clustered surface repair stages.

    """
    calls = []

    def repair(mesh, verbose=False, **kwargs):
        calls.append((verbose, kwargs))
        return repair_mesh(mesh, verbose, **kwargs)

    monkeypatch.setattr(utilities, "repair_mesh", repair)
    result = utilities.clustered_mesh(sphere, target_size=0.1, repair_backend=BACKEND)
    assert result.is_manifold
    assert calls == [(False, {"backend": BACKEND, "require_closed": True})] * 3


def test_closed_slice_repairs_open_input(sphere):
    """Close the input before slicing without modifying the original surface.

    """
    damaged = PolyData(
        pv.PolyData.from_regular_faces(sphere.points, sphere.regular_faces[1:])
    )
    result = damaged.slice_closed_surface(
        normal=np.array([0, 0, 1]), origin=(0, 0, 0), repair_backend=BACKEND
    )
    assert result.n_cells > 0
    assert not damaged.is_manifold
    assert np.allclose(result.points[:, 2], 0)


def test_preprocessing_forwards_backend(sphere, tmp_path, monkeypatch):
    """Use the requested backend for repair during file preprocessing.

    """
    filename = tmp_path / "damaged.stl"
    pv.PolyData.from_regular_faces(sphere.points, sphere.regular_faces[1:]).save(
        filename
    )
    calls = []

    def repair(mesh, **kwargs):
        calls.append(kwargs)
        return repair_mesh(mesh, **kwargs)

    monkeypatch.setattr(preprocessing.utils, "repair_mesh", repair)
    result = preprocessing.clamp_triangle_count_and_smooth(
        filename, repair_backend=BACKEND
    )
    assert result.is_watertight
    assert calls
    assert all(call == {"backend": BACKEND, "require_closed": True} for call in calls)


def test_boolean_bone_repair(sphere, monkeypatch):
    """Repair the merged boolean result and forward the optional clustering backend.

    """
    cartilage = sphere.cast_to_unstructured_grid()
    cartilage.point_data["mesh_pids"] = np.arange(cartilage.n_points)
    cartilage.field_data["joint_side_pids"] = np.arange(cartilage.n_points // 2)
    cartilage.field_data["bone_side_pids"] = np.arange(
        cartilage.n_points // 2, cartilage.n_points
    )
    calls = []

    def repair(mesh, **kwargs):
        calls.append(kwargs)
        return sphere.copy()

    def cluster(mesh, **kwargs):
        assert kwargs["repair_backend"] == BACKEND
        return mesh

    monkeypatch.setattr(pv.PolyData, "boolean_difference", lambda self, other: sphere)
    monkeypatch.setattr(utilities, "repair_mesh", repair)
    monkeypatch.setattr(utilities, "clustered_mesh", cluster)
    result = utilities.fix_protruding_bones(
        sphere, [cartilage], return_meshed=True, repair_backend=BACKEND
    )
    assert result.is_manifold
    assert calls == [{"backend": BACKEND, "require_closed": True}]


def test_reconstruct_forwards_backend_without_smoothing_leak(sphere, monkeypatch):
    """Keep repair selection separate from smoothing options throughout reconstruction.

    """
    calls = []
    monkeypatch.setattr(reconstruction, "preprocess_grid", lambda *args: object())
    monkeypatch.setattr(
        reconstruction, "apply_morphological_operations",
        lambda *args: SimpleNamespace(contour=lambda **kwargs: sphere),
    )

    def postprocess(mesh, tissue, bone, debug, **kwargs):
        assert kwargs == {"repair_backend": BACKEND, "n_iters": 3}
        calls.append("postprocess")
        return mesh

    def position(mesh, tissue, bone, **kwargs):
        assert kwargs == {"repair_backend": BACKEND}
        calls.append("position")
        return mesh

    monkeypatch.setattr(reconstruction, "postprocess_mesh", postprocess)
    monkeypatch.setattr(reconstruction, "transform_mesh", position)
    assert reconstruction.reconstruct(
        np.zeros((3, 3, 3)), "femur_cartilage",
        repair_backend=BACKEND, smooth_kwargs={"n_iters": 3},
    ) is sphere
    assert calls == ["postprocess", "position"]


def test_rigid_body_forwards_backend(sphere, monkeypatch):
    """Use one backend for rigid-body preprocessing and clustering.

    """
    from kneemesher.meshing import meshing

    calls = []

    def prepare(filename, **kwargs):
        assert kwargs["repair_backend"] == BACKEND
        calls.append("prepare")
        return sphere

    def cluster(mesh, **kwargs):
        assert kwargs["repair_backend"] == BACKEND
        calls.append("cluster")
        return mesh

    monkeypatch.setattr(meshing, "clamp_triangle_count_and_smooth", prepare)
    monkeypatch.setattr(meshing, "clustered_mesh", cluster)
    assert meshing.mesh_rigid_body("unused.stl", repair_backend=BACKEND) is sphere
    assert calls == ["prepare", "cluster"]
