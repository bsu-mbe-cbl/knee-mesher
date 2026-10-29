import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from numpy.typing import ArrayLike  # noqa
from trimesh.util import euclidean, row_norm
from vtk import VTK_HEXAHEDRON

import kneemesher.meshing.hexa_creation as hc
import kneemesher.meshing.raytracing as rays
from kneemesher.core_api import (  # noqa
    GridCartilageMesh,
    MeniscusMesh,
    SectorCartilageMesh,
)
from kneemesher.errors import (
    BlendingError,
    BlendingFailureWarning,
    ElementQualityError,
    LimitReachedError,
)
from kneemesher.mesh_repair import RepairBackend
from kneemesher.meshing.progress import report
from kneemesher.preprocessing import clamp_triangle_count_and_smooth
from kneemesher.utilities import (
    cartesian_to_polar,
    clustered_mesh,
    import_matlab_data,  # noqa
    with_debug_kwarg,
)

resource_dir = Path.cwd().joinpath("tests", "resources")
matfile_dir = resource_dir.joinpath("mat-files")


def mesh_rigid_body(
    stl_filename: Union[str, Path],
    clamp_n_triangles: int = 8000,
    target_size: float = 3.0,
    *,
    repair_backend: RepairBackend = "auto",
    **clusterd_mesh_kwargs,
) -> pv.PolyData:
    """Surface mesh rigid body with well-conditioned triangle elements.

    The STL mesh undergoes an initial decimation to clamp the number of elements, before
    undergoing adaptive subdivision. This ensures that STL files with very flat surfaces
    (where the triangles are very large and ill-conditioned) will have enough points to
    allow for well-conditioned rigid-body elements.

    Args:
        stl_filename (Union[str, Path]): filename of the STL file.
        target_size (float, optional): desired average triangle area. Defaults to 3.0.
        clamp_n_triangles (int, optional): The number of triangles allowed in the StL
            geometry before adaptive subdivision. Defaults to 8000.
        show_cluster_plot (bool, optional): if true, plot clusters used in triangle
            distribution. Defaults to False.
        verbose_mesh_repair (bool, optional): if true, show verbose output of
            the selected repair backend. Defaults to False.
        repair_backend: The backend used for preprocessing and clustering.

    Returns:
        pv.PolyData: The triangular surface mesh.
    """
    report("Preprocessing rigid surface")
    stl_mesh = pv.wrap(
        clamp_triangle_count_and_smooth(
            stl_filename,
            n_triangles=clamp_n_triangles,
            repair_backend=repair_backend,
        )
    )
    report("Clustering rigid surface")
    return clustered_mesh(
        stl_mesh,
        target_size=target_size,
        repair_backend=repair_backend,
        **clusterd_mesh_kwargs,
    )


def mesh_femur_cartilage(
    reconstruction_path: Union[str, Path],
    points: Union[Tuple[float], np.ndarray],
    radial_longitudinal_resolutions: Union[Tuple[float], np.ndarray],
    bone_reconstruction_path: Optional[Union[str, Path]] = None,
    n_triangles: int = 8000,
    reconstruction_smoothing_iters: int = 1,
    blending: bool = True,
    blending_grade: Union[float, int] = 1,
    bone_to_joint_displacement_weights: Tuple[float, float] = (0.7, 0.3),
    early_subdivision: bool = False,
    fuse_cartilage_to_bone: bool = False,
    allow_blended_edge_bone_fusion: bool = False,
    minimum_grade_scaling: float = 0.3,
    n_layers_or_slice_locs: int = 4,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh the femur cartilage based on the given parameters.

    Args:
        reconstruction_path (Union[str, Path]): Path to the reconstruction file.
        points (Union[Tuple[float], np.ndarray]): Points defining the mesh.
        resolution (Union[Tuple[float], np.ndarray]): Radial and longitudinal resolution
            of the mesh.
        bone_reconstruction_path (Optional[Union[str, Path]], optional): Path to the
            bone reconstruction file. Defaults to None.
        n_triangles (int, optional): The reconstruction will be decimated until only
            this number of triangles remains as a preprocessing step. Defaults to 8000.
        reconstruction_smoothing_iters (int, optional): Number of smoothing iterations
            for the decimated reconstruction. Defaults to 1.
        blending (bool, optional): Flag indicating whether blending should be performed.
            Defaults to True.
        blending_grade (Union[float, int], optional): Grade of blending. Defaults to 1.
        bone_to_joint_displacement_weights (Tuple[float, float], optional): Displacement
            weights for blending (bone-side stretch, joint-side compress). Defaults to
            (0.7, 0.3).
        early_subdivision (bool, optional): Flag indicating whether subdivision
            should be performed before raytracing, instead of after meshing and
            repairing degenerates. Defaults to False.
        fuse_cartilage_to_bone (bool, optional): Flag indicating whether the entire
            cartilage surface should be fused to bone. Defaults to False.
        allow_blended_edge_bone_fusion (bool, optional): Flag indicating whether blended
            edge bone fusion (as opposed to the entire cartilage surface) is performed.
            Defaults to False.
        minimum_grade_scaling (float, optional): Minimum grade scaling for blending
            before abandoning blending.  Defaults to 0.3.
        n_layers_or_slice_locs (int, optional): Number of layers or slice locations for
            depthwise subdivision. Defaults to 4.
        **kwargs: Additional keyword arguments for development purposes.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the mesh points,
            cell definitions, and point locations.
    """
    report("Loading reconstruction")
    radial_resolution, longitudinal_resolution = radial_longitudinal_resolutions

    cart_mesh = GridCartilageMesh(label="femur_cartilage", default_color="thistle")
    cart_mesh.attach_reconstruction(
        reconstruction_path,
        n_triangles=n_triangles,
        smooth_iters=reconstruction_smoothing_iters,
    )
    if bone_reconstruction_path is not None:
        cart_mesh.attach_bone_reconstruction(bone_reconstruction_path)
    cart_mesh.raytrace_pattern = "capsule"

    if early_subdivision:
        radial_resolution /= 2
        longitudinal_resolution /= 2

    cart_mesh = cart_mesh.initialize_mesh(
        points,
        radial_resolution=radial_resolution,
        longitudinal_resolution=longitudinal_resolution,
    )
    cart_mesh = cart_mesh.repair_degenerate_cells()
    if fuse_cartilage_to_bone:
        cart_mesh.fuse_to_bone(only_edge=False)

    cart_mesh.points = cart_mesh.smooth_hexahedral_cells(
        points=cart_mesh.point_locs,
        cell_def=cart_mesh.cell_def,
        smoothing_operation="surface",
    )
    cart_mesh = cart_mesh.expand_laminar_edges(inplace=False)
    cart_mesh.optimize_scaled_jacobians(target_jacobian=0, inplace=True)

    if not early_subdivision:
        cart_mesh = cart_mesh.subdivide_in_plane()

    cart_mesh.points = cart_mesh.smooth_hexahedral_cells(
        points=cart_mesh.point_locs,
        cell_def=cart_mesh.cell_def,
        smoothing_operation="surface",
    )
    cart_mesh = cart_mesh.expand_laminar_edges(inplace=False)
    for _ in range(3):
        cart_mesh.optimize_scaled_jacobians(target_jacobian=0, inplace=True)
    cart_mesh.optimize_scaled_jacobians(target_jacobian=0.5, inplace=True)

    cart_mesh.optimize_iteratively(inplace=True)
    if fuse_cartilage_to_bone:
        # cart_mesh.fuse_to_bone(only_edge=True, match_edge_translation=True)
        cart_mesh.fuse_to_bone(only_edge=False, match_edge_translation=True)
        cart_mesh.smooth_side(side="joint_side", n_iter=200)
        cart_mesh.optimize_iteratively(upper_jacobian_limit=0, inplace=True)
    pre_blend_locs = cart_mesh.point_locs
    if blending:
        if kwargs.get("report_blend_quality", False):
            describe_before = cart_mesh.blended_cell_quality(depth=5)

        proposed_displacements = cart_mesh.blend_with_retries(
            dry_run=True,
            grade=blending_grade,
            bone_to_joint_displacement_weights=bone_to_joint_displacement_weights,
            include_bone_fusion=allow_blended_edge_bone_fusion,
            minimum_grade_scaling=minimum_grade_scaling,
            n_layers_or_slice_locs=n_layers_or_slice_locs,
        )
        cart_mesh.points += proposed_displacements

        if kwargs.get("report_blend_quality", False):
            describe_after = cart_mesh.blended_cell_quality(depth=5)
            save_dir = reconstruction_path.parent / "kneemesher"
            save_dir.mkdir(exist_ok=True, parents=True)
            describe_before.to_csv(
                save_dir / f"{reconstruction_path.stem}-blend-quality-before.csv"
            )
            describe_after.to_csv(
                save_dir / f"{reconstruction_path.stem}-blend-quality-after.csv"
            )

    cart_mesh = cart_mesh.subdivide_depth(n_layers_or_slice_locs=n_layers_or_slice_locs)

    try:
        if cart_mesh.find_low_quality_cells().any():
            cart_mesh.points = pre_blend_locs.copy()
            raise ElementQualityError("Jacobians less than zero detected.")
        else:
            return cart_mesh
    except ElementQualityError as e:
        del cart_mesh
        raise e


def mesh_patella_cartilage(*args, **kwargs):
    """Mesh the patella cartilage.

    The is a wrapper function which uses `mesh_femur_cartilage` to mesh the patella.

    Args:
        *args: All positional arguments are passes to the `mesh_femur_cartilage` call.
        **kwargs: All keyword arguments are passed to the `mesh_femur_cartilage` call.

    Returns:
        cart_mesh: The mesh of the patella cartilage.
    """
    kwargs['color'] = kwargs.get('color', 'mistyrose')
    cart_mesh = mesh_femur_cartilage(*args, **kwargs)
    cart_mesh.label = "patella_cartilage"
    return cart_mesh


def mesh_tibia_cartilage(
    reconstruction_path: Union[str, Path],
    elem_counts: Union[Tuple[int], np.ndarray] = None,
    elem_sizes: Union[Tuple[float], np.ndarray] = None,
    bounding_box_scaling_factor: float = None,
    bone_reconstruction_path: Optional[Union[str, Path]] = None,
    n_triangles: int = 8000,
    reconstruction_smoothing_iters: int = 0,
    blending: bool = True,
    blending_grade: Union[float, int] = 1,
    bone_to_joint_displacement_weights: Tuple[float, float] = (0.7, 0.3),
    fuse_cartilage_to_bone: bool = False,
    allow_blended_edge_bone_fusion: bool = False,
    minimum_grade_scaling: float = 0.3,
    n_layers_or_slice_locs: int = 4,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    report("Loading reconstruction")
    cart_mesh = SectorCartilageMesh(label="tibia_cartilage", default_color="thistle")
    cart_mesh.reverse_bone_joint_side = True
    cart_mesh.attach_reconstruction(
        reconstruction_path,
        n_triangles=n_triangles,
        smooth_iters=reconstruction_smoothing_iters,
    )
    if bone_reconstruction_path is not None:
        cart_mesh.attach_bone_reconstruction(bone_reconstruction_path)

    cart_mesh = cart_mesh.initialize_mesh(
        elem_counts=elem_counts,
        elem_sizes=elem_sizes,
        bounding_box_scaling_factor=bounding_box_scaling_factor,
        # debug=True,
        # debug_breaks=True,
    )

    cart_mesh = cart_mesh.repair_degenerate_cells()
    if fuse_cartilage_to_bone:
        cart_mesh.fuse_to_bone(only_edge=False)

    cart_mesh.points = cart_mesh.smooth_hexahedral_cells(
        points=cart_mesh.point_locs,
        cell_def=cart_mesh.cell_def,
        smoothing_operation="surface",
    )
    # cart_mesh = cart_mesh.expand_laminar_edges(inplace=False)
    cart_mesh.optimize_scaled_jacobians(target_jacobian=0, inplace=True)
    cart_mesh.optimize_iteratively(inplace=True)

    if fuse_cartilage_to_bone:
        # cart_mesh.fuse_to_bone(only_edge=True, match_edge_translation=True)
        cart_mesh.fuse_to_bone(only_edge=False, match_edge_translation=True)
        cart_mesh.smooth_side(side="joint_side", n_iter=200)
        cart_mesh.optimize_iteratively(upper_jacobian_limit=0, inplace=True)

    pre_blend_locs = cart_mesh.point_locs
    if blending:
        if kwargs.get("report_blend_quality", False):
            describe_before = cart_mesh.blended_cell_quality(depth=5)

        proposed_displacements = cart_mesh.blend_with_retries(
            dry_run=True,
            grade=blending_grade,
            bone_to_joint_displacement_weights=bone_to_joint_displacement_weights,
            include_bone_fusion=allow_blended_edge_bone_fusion,
            minimum_grade_scaling=minimum_grade_scaling,
            n_layers_or_slice_locs=n_layers_or_slice_locs,
        )
        cart_mesh.points += proposed_displacements

        if kwargs.get("report_blend_quality", False):
            describe_after = cart_mesh.blended_cell_quality(depth=5)
            save_dir = reconstruction_path.parent / "kneemesher"
            save_dir.mkdir(exist_ok=True, parents=True)
            describe_before.to_csv(
                save_dir / f"{reconstruction_path.stem}-blend-quality-before.csv"
            )
            describe_after.to_csv(
                save_dir / f"{reconstruction_path.stem}-blend-quality-after.csv"
            )

    cart_mesh = cart_mesh.subdivide_depth(n_layers_or_slice_locs=n_layers_or_slice_locs)

    try:
        if cart_mesh.find_low_quality_cells().any():
            cart_mesh.points = pre_blend_locs.copy()
            raise ElementQualityError("Jacobians less than zero detected.")
    except ElementQualityError as e:
        del cart_mesh
        raise e
    else:
        return cart_mesh


def mesh_tibia_cartilage_grid(
    reconstruction_path: Union[str, Path],
    resolution: Union[Tuple[float], np.ndarray],
    bone_reconstruction_path: Optional[Union[str, Path]] = None,
    n_triangles: int = 8000,
    reconstruction_smoothing_iters: int = 1,
    blending: bool = True,
    blending_grade: Union[float, int] = 1,
    bone_to_joint_displacement_weights: Tuple[float, float] = (0.7, 0.3),
    early_subdivision: bool = False,
    fuse_cartilage_to_bone: bool = False,
    allow_blended_edge_bone_fusion: bool = False,
    minimum_grade_scaling: float = 0.3,
    n_layers_or_slice_locs: int = 4,
    **kwargs,
):
    report("Loading reconstruction")
    i_resolution, j_resolution = resolution

    cart_mesh = GridCartilageMesh(label="tibia_cartilage", default_color="thistle")
    cart_mesh.reverse_bone_joint_side = False
    cart_mesh.attach_reconstruction(
        reconstruction_path,
        n_triangles=n_triangles,
        smooth_iters=reconstruction_smoothing_iters,
    )
    if bone_reconstruction_path is not None:
        cart_mesh.attach_bone_reconstruction(bone_reconstruction_path)
    cart_mesh.raytrace_pattern = "plane"

    if early_subdivision:
        i_resolution /= 2
        j_resolution /= 2

    cart_mesh = cart_mesh.initialize_mesh(
        i_resolution=i_resolution,
        j_resolution=j_resolution,
        plane_center=None,
        plane_normal=None,
    )

    cart_mesh = cart_mesh.repair_degenerate_cells()
    if fuse_cartilage_to_bone:
        cart_mesh.fuse_to_bone(only_edge=False)

    cart_mesh.points = cart_mesh.smooth_hexahedral_cells(
        points=cart_mesh.point_locs,
        cell_def=cart_mesh.cell_def,
        smoothing_operation="surface",
    )
    cart_mesh = cart_mesh.expand_laminar_edges(inplace=False)
    cart_mesh.optimize_scaled_jacobians(target_jacobian=0, inplace=True)

    if not early_subdivision:
        cart_mesh = cart_mesh.subdivide_in_plane()

    cart_mesh.points = cart_mesh.smooth_hexahedral_cells(
        points=cart_mesh.point_locs,
        cell_def=cart_mesh.cell_def,
        smoothing_operation="surface",
    )
    cart_mesh = cart_mesh.expand_laminar_edges(inplace=False)
    for _ in range(3):
        cart_mesh.optimize_scaled_jacobians(target_jacobian=0, inplace=True)
    cart_mesh.optimize_scaled_jacobians(target_jacobian=0.5, inplace=True)

    cart_mesh.optimize_iteratively(inplace=True)
    if fuse_cartilage_to_bone:
        # cart_mesh.fuse_to_bone(only_edge=True, match_edge_translation=True)
        cart_mesh.fuse_to_bone(only_edge=False, match_edge_translation=True)
        cart_mesh.smooth_side(side="joint_side", n_iter=200)
        cart_mesh.optimize_iteratively(upper_jacobian_limit=0, inplace=True)
    pre_blend_locs = cart_mesh.point_locs
    if blending:
        if kwargs.get("report_blend_quality", False):
            describe_before = cart_mesh.blended_cell_quality(depth=5)

        proposed_displacements = cart_mesh.blend_with_retries(
            dry_run=True,
            grade=blending_grade,
            bone_to_joint_displacement_weights=bone_to_joint_displacement_weights,
            include_bone_fusion=allow_blended_edge_bone_fusion,
            minimum_grade_scaling=minimum_grade_scaling,
            n_layers_or_slice_locs=n_layers_or_slice_locs,
        )
        cart_mesh.points += proposed_displacements

        if kwargs.get("report_blend_quality", False):
            describe_after = cart_mesh.blended_cell_quality(depth=5)
            save_dir = reconstruction_path.parent / "kneemesher"
            save_dir.mkdir(exist_ok=True, parents=True)
            describe_before.to_csv(
                save_dir / f"{reconstruction_path.stem}-blend-quality-before.csv"
            )
            describe_after.to_csv(
                save_dir / f"{reconstruction_path.stem}-blend-quality-after.csv"
            )

    cart_mesh = cart_mesh.subdivide_depth(n_layers_or_slice_locs=n_layers_or_slice_locs)

    try:
        if cart_mesh.find_low_quality_cells().any():
            cart_mesh.points = pre_blend_locs.copy()
            raise ElementQualityError("Jacobians less than zero detected.")
    except ElementQualityError as e:
        raise e
    return cart_mesh


# @with_debug_kwarg
def mesh_meniscus(
    is_medial: bool,
    origin: Union[Tuple[float], np.ndarray],
    reconstruction_path: Union[str, Path],
    resolution: Union[Tuple[float], np.ndarray],
    n_triangles: int = 4000,
    reconstruction_smoothing_iters: int = 1,
    **kwargs,
):
    report("Loading reconstruction")
    men_mesh = MeniscusMesh(label="meniscus", default_color="thistle")
    men_mesh.attach_reconstruction(
        reconstruction_path,
        n_triangles=n_triangles,
        smooth_iters=reconstruction_smoothing_iters,
    )

    triangle = dict(faces=np.arange(3))

    # Raytracing
    radial_res = resolution[0]

    if kwargs.get("debug", False):
        mat_dir = matfile_dir.joinpath("meniscus")
        mat_dict = import_matlab_data(
            mat_dir / "meniscus_hex_mesh_108_tri_loop_004.mat"
        )
        origin = mat_dict["orig"]
        radial_res = mat_dict["RadialRes"]
        men_mesh.reconstruction = trimesh.Trimesh(
            vertices=mat_dict["vMM"], faces=mat_dict["fMM"] - 1
        )

    if is_medial:
        theta = 180 + np.arange(0, 360, radial_res)
        theta = np.deg2rad(np.roll(theta, theta.size // 2))
    else:
        theta = 180 + np.arange(0, 360, radial_res)
        theta = np.deg2rad(theta)

    angle = np.vstack((np.cos(theta), np.sin(theta), np.zeros_like(theta))).T
    t1 = np.tile(origin, (theta.size, 1))
    t2 = t1 + 30 * angle
    t3 = t2.copy()
    t3[:, -1] += 60
    t2[:, -1] -= 60

    reconstruct = men_mesh.wrapped_reconstruction
    nodes = np.full((theta.size, 17, 3), np.nan)
    report("Raytracing meniscus", total=len(theta), unit="slices")
    for idx, (it1, it2, it3, ang) in enumerate(zip(t1, t2, t3, angle)):
        triangle["vertices"] = np.vstack((it1, it2, it3))
        tri = pv.Triangle(np.vstack((it1, it2, it3)))
        intersect = tri.intersection(
            reconstruct, split_first=False, split_second=False
        )[0]
        if intersect.n_points > 0:
            points2 = intersect.points.copy()
            points2[:, -1] *= 3

            # Find the closest point to the origin
            closest_pid = intersect.find_closest_point(origin)
            P1 = intersect.points[closest_pid]
            A1 = points2[closest_pid]

            # Find the furthest point from P1
            side_len_1 = row_norm(points2 - A1)
            furthest_pid = np.argmax(side_len_1)
            P2 = intersect.points[furthest_pid]
            A2 = points2[furthest_pid]

            # Find the point which maximizes the area of the triangle (Heron's formula)
            side_len_2 = row_norm(points2 - A2)
            side_len_3 = row_norm(np.atleast_2d(A2 - A1))
            semi_perim = 0.5 * (side_len_1 + side_len_2 + side_len_3)
            areas = np.sqrt(
                semi_perim
                * (semi_perim - side_len_1)
                * (semi_perim - side_len_2)
                * (semi_perim - side_len_3)
            )

            largest_area_pids = np.argsort(areas)[::-1]
            P3 = intersect.points[largest_area_pids[0]]
            A3 = points2[largest_area_pids[0]]

            z_distances = np.abs((points2[largest_area_pids] - A2))[:, -1]
            side23_z_distances = np.abs(points2[largest_area_pids] - A2)[:, -1]
            big_enough = side23_z_distances > np.min([1.5, 0.6 * z_distances.max()])
            best_pid = largest_area_pids[np.flatnonzero(big_enough)[0]]

            P3 = intersect.points[best_pid]
            A3 = points2[best_pid]

            if P3[-1] > P2[-1]:
                P3, P2 = P2, P3

            triangle = pv.Triangle(np.vstack((P1, P2, P3)))

            # Convert to cylindrical coordinates
            p = intersect.points - origin
            rho, phi, z = cartesian_to_polar(*p.T)
            edges = intersect.lines.reshape(-1, 3)[:, 1:].T

            rho1, phi1, z1 = cartesian_to_polar(*(P1 - origin).T)
            rho2, phi2, z2 = cartesian_to_polar(*(P2 - origin).T)
            rho3, phi3, z3 = cartesian_to_polar(*(P3 - origin).T)

            # Place 17 nodes around the triangle within the cylindrical coordinate
            # system
            segments = np.vstack(
                (rho[edges[0]], z[edges[0]], rho[edges[1]], z[edges[1]])
            )
            p_slice = place_slice_nodes([rho1, z1], [rho2, z2], [rho3, z3], segments)

            # Send them back to cartesian coordinates
            c1 = p_slice[:, 0] * ang[0] + origin[0]
            c2 = p_slice[:, 0] * ang[1] + origin[1]
            c3 = p_slice[:, 1] + origin[2]
            nodes[idx] = np.vstack((c1, c2, c3)).T
            if kwargs.get("debug", False):
                fname = f"meniscus_hex_mesh_108_tri_loop_{idx + 1:03d}.mat"
                if mat_dir.joinpath(fname).exists():
                    mat_dict = import_matlab_data(mat_dir / fname)
                    assert np.allclose(
                        nodes[~np.isnan(nodes)].reshape(-1, 3), mat_dict["nodes"]
                    )
        report("Raytracing meniscus", idx + 1, len(theta), unit="slices")
    nodes = nodes[~np.isnan(nodes)].reshape(-1, 17, 3)

    quad_pids = (
        np.array(
            [
                1,
                8,
                14,
                7,
                8,
                2,
                16,
                14,
                14,
                16,
                3,
                10,
                4,
                7,
                14,
                10,
                7,
                4,
                11,
                15,
                15,
                13,
                1,
                7,
                15,
                11,
                5,
                17,
                15,
                17,
                6,
                13,
                2,
                9,
                3,
                16,
                17,
                5,
                12,
                6,
            ]
        ).reshape(-1, 4)
        - 1
    )

    # Make a layer of brick elements
    hex_pids = np.hstack((quad_pids, quad_pids + 17))
    hex_pids = np.tile(hex_pids, (nodes.shape[0] - 1, 1, 1))
    for idx, q_pids in enumerate(hex_pids):
        hex_pids[idx] = q_pids + 17 * idx

    nodes = nodes.reshape(-1, 3)
    hex_pids = hex_pids.reshape(-1, 8)

    if kwargs.get("debug", False):
        mat_dict = import_matlab_data(mat_dir / "meniscus_hex_mesh_131_pre_smooth.mat")
        assert np.allclose(nodes, mat_dict["nodes"])
        assert np.allclose(hex_pids + 1, mat_dict["hex"])

    men_mesh = men_mesh.update_cell_def(cell_def=hex_pids, points=nodes)
    men_mesh.points = men_mesh.smooth_hexahedral_cells(
        points=men_mesh.points,
        cell_def=men_mesh.cell_def,
        smoothing_operation="meniscus",
    )
    if kwargs.get("debug", False):
        mat_dict = import_matlab_data(mat_dir / "meniscus_hex_mesh_133_post_smooth.mat")
        assert np.allclose(men_mesh.points, mat_dict["nodes"])

    if kwargs.get("debug", False):
        men_mesh = men_mesh.update_cell_def(cell_def=hex_pids, points=mat_dict["nodes"])
    men_mesh = men_mesh.expand_profile(
        max_node_shift=5, inplace=False, both_directions=True, expand_tips=False
    )

    if kwargs.get("debug", False):
        mat_dict = import_matlab_data(mat_dir / "meniscus_hex_mesh_138_post_expand.mat")
        p = pv.Plotter()
        p.add_mesh(men_mesh, color="w", show_edges=True, opacity=0.5)
        mm = men_mesh.copy().update_cell_def(
            cell_def=hex_pids, points=mat_dict["nodes"]
        )
        p.add_mesh(mm, color="r", show_edges=True, opacity=0.5)
        p.add_axes()
        p.show()
        # assert np.allclose(men_mesh.points, mat_dict["nodes"])

    men_mesh.points = men_mesh.smooth_hexahedral_cells(
        points=men_mesh.points,
        cell_def=men_mesh.cell_def,
        smoothing_operation="total",
    )
    men_mesh.optimize_scaled_jacobians(target_jacobian=0, inplace=True)
    men_mesh.optimize_scaled_jacobians(target_jacobian=0.5, inplace=True)
    men_mesh = men_mesh.subdivide()

    men_mesh.smooth_by_slice(n_iter=200, boundary_smoothing=True, inplace=True)

    men_mesh.points = men_mesh.smooth_hexahedral_cells(
        points=men_mesh.points,
        cell_def=men_mesh.cell_def,
        smoothing_operation="meniscus",
    )
    men_mesh.expand_profile(
        max_node_shift=5, inplace=True, both_directions=True, expand_tips=False
    )
    men_mesh.points = men_mesh.smooth_hexahedral_cells(
        points=men_mesh.points,
        cell_def=men_mesh.cell_def,
        smoothing_operation="total",
    )
    men_mesh.optimize_scaled_jacobians(target_jacobian=0, inplace=True)
    men_mesh.optimize_scaled_jacobians(target_jacobian=0.5, inplace=True)

    men_mesh.optimize_iteratively(inplace=True, max_iters=30)

    men_mesh = men_mesh.create_horn_node_set(inplace=False)

    return men_mesh


def place_slice_nodes(P1, P2, P3, segments):
    segment_origins = np.vstack((segments[:2], np.zeros(segments.shape[1]))).T
    segment_terminations = np.zeros_like(segment_origins)
    segment_terminations[:, :2] = segments[2:].T
    segment_interleave = np.zeros((segment_origins.shape[0] * 2, 3))
    segment_interleave[0::2] = segment_origins
    segment_interleave[1::2] = segment_terminations
    segment_mesh = pv.line_segments_from_points(segment_interleave)
    shell_mesh = segment_mesh.translate([0, 0, -1], inplace=False).extrude(
        [0, 0, 1], capping=False
    )

    # Find the center of mass of the 2D triangle
    P1, P2, P3 = [np.array(x) for x in (P1, P2, P3)]
    tri = np.zeros((3, 3))
    tri[:, :2] = np.vstack((P1, P2, P3))
    C = pv.Triangle(tri).center_of_mass()
    P1, P2, P3 = tri

    # Place first point
    ray = np.vstack([C, P1])
    intersect_locs = shell_mesh.ray_trace(*ray)[0]
    if intersect_locs.size > 0:
        scaling = row_norm(intersect_locs - C).min()
        if scaling < 0.9999:
            P1 = C + scaling * trimesh.unitize(
                P1 - C
            )  # FIXME: Did he mean to unitize the direction?
    p1 = P3

    # Place third point
    orig = C + 0.8 * (P1 - C)
    direction = trimesh.unitize(P1 - C)
    direction[0], direction[1] = direction[1], -direction[0]
    p3 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place second point
    direction = trimesh.unitize(P1 - C)
    direction[0], direction[1] = -direction[1], direction[0]
    p2 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place fourth point
    orig = C
    direction = trimesh.unitize(C - P3)
    p4 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place sixth point
    orig = C + 0.9 * (P2 - C)
    direction = trimesh.unitize(P2 - C)
    direction[0], direction[1] = direction[1], -direction[0]
    p6 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place the fifth point
    direction = trimesh.unitize(P2 - C)
    direction[0], direction[1] = -direction[1], direction[0]
    p5 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place the seventh (center) point
    p7 = np.vstack((p1, p4)).mean(axis=0)

    # Place the point between the first and second point
    orig = p7
    direction = trimesh.unitize(np.vstack((p1, p2)).mean(axis=0) - p7)
    p12 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place the point between the second and third point
    orig = p7
    direction = trimesh.unitize(np.vstack((p2, p3)).mean(axis=0) - p7)
    p23 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place the point between 3 and 4
    orig = p7
    direction = np.vstack((p3, p4)).mean(axis=0) - p7
    p34 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place the point between 4 and 5
    orig = p7
    direction = trimesh.unitize(np.vstack((p4, p5)).mean(axis=0) - p7)
    p45 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place the point between 5 and 6
    orig = p7
    direction = trimesh.unitize(np.vstack((p5, p6)).mean(axis=0) - p7)
    p56 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Place the point between 6 and 1
    orig = p7
    direction = trimesh.unitize(np.vstack((p6, p1)).mean(axis=0) - p7)
    p61 = raytrace_meniscus_slice(
        orig, orig + 100 * direction, slice_extrusion=shell_mesh
    )

    # Get the interior points
    aux1 = np.vstack((p7, p23)).mean(axis=0)
    aux2 = np.vstack((p7, p56)).mean(axis=0)

    aux11 = np.vstack((aux1, p23)).mean(axis=0)
    aux22 = np.vstack((aux2, p56)).mean(axis=0)

    points = np.vstack(
        (
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p12,
            p23,
            p34,
            p45,
            p56,
            p61,
            aux1,
            aux2,
            aux11,
            aux22,
        )
    )
    return points
    pass


def raytrace_meniscus_slice(ray_origin, ray_termination, slice_extrusion):
    direction = trimesh.unitize(ray_termination - ray_origin)
    locs = slice_extrusion.ray_trace(ray_origin, ray_termination)[0]
    if locs.size > 0:
        scaling = row_norm(locs - ray_origin).min()
    else:
        scaling = 0
    return ray_origin + scaling * direction
