"""Preprocess the STL files assuming right-handed-coordinate system.

Superior = +Z
Anterior = +Y
Left = +x
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pyvista as pv
import trimesh

from kneemesher import utilities as utils
from kneemesher.errors._errors import MeshPreparationError
from kneemesher.pv_helpers import PolyData
from kneemesher.mesh_repair import RepairBackend
from kneemesher.utilities import trimesh_wrap


# @with_debug_kwarg
def prepare_with_bones(
    femur_bone: Union[str, Path] = None,
    femur_cartilage: Union[str, Path] = None,
    tibia_bone: Union[str, Path] = None,
    tibia_cartilage_lateral: Union[str, Path] = None,
    tibia_cartilage_medial: Union[str, Path] = None,
    meniscus_lateral: Union[str, Path] = None,
    meniscus_medial: Union[str, Path] = None,
    patella_bone: Union[str, Path] = None,
    patella_cartilage: Union[str, Path] = None,
    *,
    repair_backend: RepairBackend = "auto",
    **kwargs,
) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Dict[str, float]]:
    """Load and validate tissue surfaces with the selected repair backend.

    """
    # Load the STL models and obtain triangular meshes
    tissue_names = [
        "femur_bone",
        "femur_cartilage",
        "tibia_bone",
        "tibia_cartilage_lateral",
        "tibia_cartilage_medial",
        "meniscus_lateral",
        "meniscus_medial",
        "patella_bone",
        "patella_cartilage",
    ]
    stl_filenames = [
        femur_bone,
        femur_cartilage,
        tibia_bone,
        tibia_cartilage_lateral,
        tibia_cartilage_medial,
        meniscus_lateral,
        meniscus_medial,
        patella_bone,
        patella_cartilage,
    ]

    mesh = {
        t: clamp_triangle_count_and_smooth(
            f, n_triangles=8000, smooth_iters=0, repair_backend=repair_backend
        )
        for t, f in zip(tissue_names, stl_filenames)
        if f is not None
    }
    if not all(m.is_watertight for m in mesh.values()):
        raise MeshPreparationError(
            f"Meshes are not watertight: {[k for k, m in mesh.items() if not m.is_watertight]}"
        )
    mesh = {t: PolyData(pv.wrap(m)) for t, m in mesh.items()}

    # TODO: Find the centers of tibial cartilages and menisci
    # - [ ] Find the bounding box of the femur bone and cartilage (guarenteeing that the
    #   bounding box is nearly axis-aligned?)
    # - [ ] Find the bounding box of the femoral cartilage (axis is located 25% from the
    #   superior face of this bbox)
    # - [ ] Use the lowest point of the femoral condyles to center the tibial cartilage
    #   COMs

    # TODO: Find the axis points for the femur and patella cylinders


# @with_debug_kwarg
def prepare(
    femur_cartilage: Union[str, Path] = None,
    tibia_cartilage_lateral: Union[str, Path] = None,
    tibia_cartilage_medial: Union[str, Path] = None,
    meniscus_lateral: Union[str, Path] = None,
    meniscus_medial: Union[str, Path] = None,
    patella_cartilage: Union[str, Path] = None,
    *,
    repair_backend: RepairBackend = "auto",
    **kwargs,
) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Dict[str, float]]:
    """Find centroids of each tissue and locate cylindrical raytracing axes.

    Args:
        femur_cartilage (Union[str, Path]): filename of the femoral cartilage STL.
        tibia_cartilage_lateral (Union[str, Path]): filename of the lateral tibial
            cartilage STL.
        tibia_cartilage_medial (Union[str, Path]): filename of the medial tibial
            cartilage STL.
        meniscus_lateral (Union[str, Path]): filename of the lateral meniscus STL.
        meniscus_medial (Union[str, Path]): filename of the medial meniscus STL.
        patella_cartilage (Union[str, Path]): filename of the patellar cartilage STL.
        repair_backend: The backend used to prepare the input surfaces.

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float], Dict[str, float]]: The points
            creating the cylindrical axes used for raytracing during femoral and
            patellar cartilage meshing, as well as a dictionary containing the centers
            of mass for each of the tissues.

    """
    # Load the STL models and obtain triangular meshes
    tissue_names = [
        "femur_cartilage",
        "tibia_cartilage_lateral",
        "tibia_cartilage_medial",
        "meniscus_lateral",
        "meniscus_medial",
        "patella_cartilage",
    ]
    stl_filenames = [
        femur_cartilage,
        tibia_cartilage_lateral,
        tibia_cartilage_medial,
        meniscus_lateral,
        meniscus_medial,
        patella_cartilage,
    ]
    mesh = {
        t: clamp_triangle_count_and_smooth(
            f, n_triangles=8000, smooth_iters=0, repair_backend=repair_backend
        )
        for t, f in zip(tissue_names, stl_filenames)
        if f is not None
    }
    if not all(m.is_watertight for m in mesh.values()):
        raise MeshPreparationError(
            f"Meshes are not watertight: {[k for k, m in mesh.items() if not m.is_watertight]}"
        )

    # Find the plane (bottom of a box) containing the 25% most superior portion of
    # bounding box
    femur_cart_bbox = TissueBoundingBox(mesh["femur_cartilage"])
    right_ant_sup = femur_cart_bbox.corner("right", "anterior", "superior")
    right_ant_inf = femur_cart_bbox.corner("right", "anterior", "inferior")
    fem_cart_sup_dir = right_ant_sup - right_ant_inf
    fem_cart_sup_unit_vec = fem_cart_sup_dir / np.linalg.norm(fem_cart_sup_dir)

    cut_origin = right_ant_inf + 0.75 * fem_cart_sup_dir
    cut_normal = fem_cart_sup_unit_vec

    upper_quarter_bounding_box = femur_cart_bbox.bbox.slice_plane(
        plane_origin=cut_origin, plane_normal=cut_normal, cap=True
    )

    # Find the intersection of the bottom of this sub-box with rays originating in the
    # center of the tibial cartilages bounding boxes
    ray_origins = np.asarray(
        [
            mesh[x].bounding_box_oriented.center_mass
            for x in ["tibia_cartilage_medial", "tibia_cartilage_lateral"]
        ]
    )

    # Create a pyvista plane that spans a larger area than the tibial cartilage
    plane = pv.Plane(
        center=cut_origin,
        direction=cut_normal,
        i_size=upper_quarter_bounding_box.extents.max() * 4,
        j_size=upper_quarter_bounding_box.extents.max() * 4,
    )

    ray_directions = np.asarray([cut_normal, cut_normal])
    plane = trimesh_wrap(plane)
    try:
        tri_idx, ray_idx, locations = plane.ray.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            return_locations=True,
            multiple_hits=False,
        )
    except IndexError:
        raise MeshPreparationError(
            "Raytracing failed to find the intersection of the bottom of the sub-box with rays originating in the center of the tibial cartilages bounding boxes."
        )

    sort = ray_idx.argsort()
    tri_idx, ray_idx, locations = tri_idx[sort], ray_idx[sort], locations[sort]

    cylinder = trimesh.creation.cylinder(15, segment=locations)

    # Find the centers of mass for the bounding boxes of the tibial cartilages and
    # menisci
    mass_centers = {t: mesh[t].bounding_box_oriented.center_mass for t in mesh.keys()}

    # Patella
    # ----------------------------------------------------------------------------------
    # Find the orientation of the patella
    patella_cart_bbox = TissueBoundingBox(mesh["patella_cartilage"])
    femur_anterior_dir = femur_cart_bbox.face_normal("anterior")
    face_normals = mesh["patella_cartilage"].bounding_box_oriented.face_normals
    closest_normal_idx = np.asarray(
        [
            trimesh.geometry.vector_angle(np.asarray([femur_anterior_dir, n]))
            for n in face_normals
        ]
    ).argmin()
    patella_cartilage_anterior_dir = face_normals[closest_normal_idx]

    femur_superior_dir = femur_cart_bbox.face_normal("superior")
    closest_normal_idx = np.asarray(
        [
            trimesh.geometry.vector_angle(np.asarray([femur_superior_dir, n]))
            for n in face_normals
        ]
    ).argmin()
    patella_cartilage_superior_dir = face_normals[closest_normal_idx]

    # Find the axis points for the patella cylinder
    max_extent = patella_cart_bbox.bbox.extents.max()
    centroid = patella_cart_bbox.bbox.centroid
    patella_axis_center_superior = (
        centroid
        + 0.5 * max_extent * patella_cartilage_superior_dir
        + 0.5 * max_extent * patella_cartilage_anterior_dir
    )
    patella_axis_center_inferior = (
        centroid
        - 0.5 * max_extent * patella_cartilage_superior_dir
        + 0.5 * max_extent * patella_cartilage_anterior_dir
    )
    patella_axis_points = np.array(
        [patella_axis_center_inferior, patella_axis_center_superior]
    )

    if kwargs.get("debug", False):
        trimesh.Scene(
            [
                # mesh["femur_cartilage"],
                mesh["patella_cartilage"],
                mesh["tibia_cartilage_lateral"],
                mesh["tibia_cartilage_medial"],
                # patella_cart_bbox.bbox,
                # femur_cart_bbox.bbox,
                trimesh.PointCloud(patella_axis_points),
                trimesh.creation.cylinder(
                    30,
                    segment=np.array(patella_axis_points),
                ),
            ]
        ).show()
    # ----------------------------------------------------------------------------------

    if kwargs.get("debug", False):
        ray_visualize = trimesh.load_path(
            np.hstack((ray_origins, ray_origins + ray_directions * 50.0)).reshape(
                -1, 2, 3
            )
        )

        # create and show a visualization
        upper_quarter_bounding_box.unmerge_vertices()
        upper_quarter_bounding_box.visual.face_colors[tri_idx] = [255, 0, 0, 255]
        scene = trimesh.Scene(
            [
                cylinder,
                mesh["femur_cartilage"],
                mesh["patella_cartilage"],
                mesh["tibia_cartilage_lateral"],
                mesh["tibia_cartilage_medial"],
                upper_quarter_bounding_box,
                ray_visualize,
            ]
        )
        scene.show()

    femur_cylinder_axis_points = locations
    patella_cylinder_axis_points = np.asarray(
        (patella_axis_center_inferior, patella_axis_center_superior)
    )
    # NOTE: Rodriguez-Villa centers meniscus on tibia for return values
    for side_str in ["lateral", "medial"]:
        if not mesh.get(f"meniscus_{side_str}"):
            continue
        tib_mc = mass_centers[f"tibia_cartilage_{side_str}"]
        men_mc = (
            mass_centers[f"meniscus_{side_str}"]
            if mesh.get(f"meniscus_{side_str}")
            else tib_mc
        )
        mass_centers[f"meniscus_{side_str}"][:2] = np.vstack((tib_mc, men_mc)).mean(
            axis=0
        )[:2]

    mass_centers["meniscus_lateral"] = mass_centers["tibia_cartilage_lateral"]
    mass_centers["meniscus_medial"] = mass_centers["tibia_cartilage_medial"]
    return femur_cylinder_axis_points, patella_cylinder_axis_points, mass_centers


def clamp_triangle_count_and_smooth(
    filename: Union[str, Path],
    n_triangles: int = None,
    smooth_iters: int = None,
    use_trimesh: bool = True,
    repair_mesh: bool = True,
    *,
    repair_backend: RepairBackend = "auto",
    **kwargs,
) -> trimesh.Trimesh:
    """Decimate to target number of faces and smooth a mesh.

    Args:
        filename (Union[str, Path]): Filename pointing to a triangular mesh compatible
            with the trimesh package. Normally an STL file.
        n_triangles (int, optional): Target number of triangles following decimation.
            Defaults to None. Rodriquez-Vila et al. used 8000.
        smooth_iters (int, optional): The number of Laplacian smoothing iterations.
            Defaults to None.
        repair_backend: The backend used when surface repair is needed.

    Returns:
        Trimesh: The decimated and smoothed triangular mesh. Use Trimesh.vertices an
            Trimesh.faces to access vertices and triangles.

    """
    if use_trimesh:
        mesh = trimesh.load_mesh(filename)
        if repair_mesh:
            trimesh.repair.fix_normals(mesh)
            if not mesh.is_watertight:
                mesh = trimesh_wrap(
                    utils.repair_mesh(
                        pv.wrap(mesh), backend=repair_backend, require_closed=True
                    )
                )
        if n_triangles is not None:
            orig_n_triangles = mesh.faces.shape[0]
            if orig_n_triangles > n_triangles:
                mesh = mesh.simplify_quadric_decimation(face_count=n_triangles)
        if smooth_iters is not None:
            trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters)
    else:
        mesh = pv.read(filename)
        if n_triangles is not None:
            orig_n_triangles = mesh.n_cells
            if orig_n_triangles > n_triangles:
                mesh = mesh.decimate_pro(1 - n_triangles / orig_n_triangles)
            if smooth_iters is not None:
                try:
                    mesh = mesh.smooth_taubin(
                        n_iter=smooth_iters, normalize_coordinates=True
                    )
                except AttributeError:
                    mesh = mesh.smooth(n_iter=smooth_iters)

            mesh = trimesh.Trimesh(
                vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:]
            )
    if repair_mesh is False and np.isnan(mesh.centroid).any():
        mesh = clamp_triangle_count_and_smooth(
            filename,
            n_triangles=n_triangles,
            smooth_iters=smooth_iters,
            use_trimesh=use_trimesh,
            repair_mesh=True,
            repair_backend=repair_backend,
        )
    if repair_mesh:
        trimesh.repair.fix_normals(mesh)
        if not mesh.is_watertight:
            mesh = trimesh_wrap(
                utils.repair_mesh(
                    pv.wrap(mesh), backend=repair_backend, require_closed=True
                )
            )
    if kwargs.get("debug", False):
        bbox = mesh.bounding_box_oriented
        scene = trimesh.Scene([mesh, bbox])
        scene.show()
    return mesh


class TissueBoundingBox:
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        left_right: str = None,
    ) -> None:
        self.bbox = mesh.bounding_box_oriented
        self.verts = self.bbox.vertices
        self.transform = self.bbox.primitive.transform

        # Implement Trimesh #573
        extents = mesh.bounding_box_oriented.extents
        half = mesh.bounding_box_oriented.primitive.extents / 2
        oriented_corners = trimesh.transform_points(
            trimesh.bounds.corners([-half, half]), mesh.bounding_box_oriented.transform
        )
        corners = trimesh.bounds.corners([-half, half])
        rotation = np.eye(4)
        rotation[:3, :3] = self.transform[:3, :3]

        box = trimesh.primitives.Box(
            extents=mesh.bounding_box_oriented.extents,
            # bounds=self.bbox.bounds,
            transform=self.bbox.transform,
            mutable=True,
        )


        is_right, is_anterior, is_superior = (self.verts > self.bbox.center_mass).T
        is_left, is_posterior, is_inferior = ~is_right, ~is_anterior, ~is_superior

        self.position_map = dict(
            right=is_right,
            left=is_left,
            anterior=is_anterior,
            posterior=is_posterior,
            superior=is_superior,
            inferior=is_inferior,
        )

        if left_right is not None and "left" in left_right.lower():
            self.position_map["medial"] = self.position_map["right"]
            self.position_map["lateral"] = self.position_map["left"]
        elif left_right is not None and "right" in left_right.lower():
            self.position_map["medial"] = self.position_map["left"]
            self.position_map["lateral"] = self.position_map["right"]
        else:
            # TODO: Add an error message
            pass

    def corner(
        self,
        *joint_axes,
    ):
        if len(joint_axes) == 1:
            joint_axes = joint_axes * 3
        if len(joint_axes) != 3:
            # TODO: Add an exception
            pass
        queries = np.empty(3, dtype="object")
        for idx, ax in enumerate(joint_axes):
            if ax not in self.position_map.keys():
                # TODO: Add an exception
                pass
            queries[idx] = self.position_map[ax]
        return self.verts[queries[0] * queries[1] * queries[2]].squeeze()

    def face_normal(
        self,
        joint_axis,
    ):
        corners = self.corner(joint_axis)
        centroid = trimesh.PointCloud(corners).centroid
        direction = centroid - self.bbox.centroid
        return trimesh.unitize(direction)
