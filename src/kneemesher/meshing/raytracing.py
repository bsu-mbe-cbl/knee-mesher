from __future__ import annotations

import warnings
from itertools import combinations
from typing import Dict, Tuple, Union

import deprecation
import matplotlib.pyplot as plt  # noqa
import numpy as np
import pyvista as pv
import trimesh
import trimesh.bounds as tri_bounds
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from trimesh import unitize
from trimesh.path import polygons

from kneemesher import __version__
from kneemesher.errors import RaySectorError
from kneemesher.meshing.progress import WorkProgress, report
from kneemesher.plotting import ray_glyphs, rays_to_paths, scene  # noqa
from kneemesher.utilities import clustered_mesh, shortest_align_vectors


def apply_transformation(
    points: np.ndarray,
    transformation_matrix: np.ndarray,
) -> np.ndarray:
    """Apply transformation matrix to an array of shape (n, 3).

    Args:
        points (np.ndarray): The points in need of rotation/translations.
        transformation_matrix (np.ndarray): A homogenous transformation matrix of shape
            (4, 4).

    Returns:
        np.ndarray: The transformed points.
    """
    points = np.array(points)
    points = points.T if points.shape[1] != 3 else points
    pcloud = trimesh.PointCloud(points)
    pcloud.apply_transform(transformation_matrix)
    return pcloud.vertices


def femur_cartilage(
    axis_origin: np.ndarray,
    axis_termination: np.ndarray,
    radial_res: Union[int, float] = 5,
    axial_res: Union[int, float] = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return directions and origins of rays used for initializing femoral cartilage.

    Creates a capsule-shaped geometry aligned with an estimated epicondyle axis. The
    topological ordering of these rays is important downstream in the algorithm.

    Args:
        axis_origin (np.ndarray): The origin of the estimatd epicondyle axis.
        axis_termination (np.ndarray): The termiating point of the estimated epicondyle
            axis.
        radial_res (int, optional): The element resolution in the radial direction.
            Defaults to 5.
        axial_res (int, optional): The element resolution in the axial direction.
            Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The ray directions,
        origins, the number of nodes in the radial direction, and the number of
        nodes in the axial direction.
    """
    # Creation of the angles of the rays
    theta = np.arange(0, 360, radial_res)
    theta = np.deg2rad(np.hstack((np.roll(theta, theta.size // 2), 180)))

    # Creation of the spherical endcaps
    phi = np.deg2rad(np.arange(0, 90, radial_res))
    rho = np.flip(phi)
    right_endcap = np.vstack(
        (
            (np.cos(theta) * np.sin(rho[..., np.newaxis])).reshape(1, -1),
            (np.sin(theta) * np.sin(rho[..., np.newaxis])).reshape(1, -1),
            (np.tile(np.cos(rho), (theta.size, 1)).T).reshape(1, -1),
        )
    ).T
    left_endcap = np.vstack(
        (
            (np.cos(theta) * np.sin(phi[..., np.newaxis])).reshape(1, -1),
            (np.sin(theta) * np.sin(phi[..., np.newaxis])).reshape(1, -1),
            (-np.tile(np.cos(phi), (theta.size, 1)).T).reshape(1, -1),
        )
    ).T

    # Capsule midsection
    circle_in_xy = np.array((np.cos(theta), np.sin(theta), np.zeros_like(theta))).T
    cylinder_axis_vector = axis_termination - axis_origin
    parameterized_input = cylinder_axis_vector / np.round(
        np.linalg.norm(cylinder_axis_vector) / axial_res
    )
    n_slices = np.round(
        np.linalg.norm(cylinder_axis_vector) / np.linalg.norm(parameterized_input)
    )
    slice_origins = np.linspace(axis_origin, axis_termination, int(n_slices + 1))
    midsection_origins = np.moveaxis(
        np.tile(slice_origins[..., np.newaxis], (1, 1, theta.size)), 0, -1
    )
    midsection_directions = np.tile(
        circle_in_xy.T[..., np.newaxis], (1, 1, midsection_origins.shape[-1])
    )

    # Concatenating the capsule
    _, n_radial_nodes, n_axial_nodes = midsection_origins.shape
    n_axial_nodes = n_axial_nodes + 2 * phi.size
    origins = np.vstack(
        (
            np.tile(axis_origin, (left_endcap.shape[0], 1)),
            midsection_origins.reshape(3, -1, order="F").T,
            np.tile(axis_termination, (right_endcap.shape[0], 1)),
        )
    )
    directions = np.vstack(
        (left_endcap, midsection_directions.reshape(3, -1, order="F").T, right_endcap)
    )

    # Align the capsule geometry with the cylinder axis
    z_normal = np.array((0, 0, 1))
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = shortest_align_vectors(z_normal, cylinder_axis_vector)
    directions = apply_transformation(directions, transform_mat)

    # #region DEBUG: plot the resulting rays.
    # import kneemesher.plotting as kmplt

    return directions, origins, n_radial_nodes, n_axial_nodes


def patella_cartilage(
    axis_origin: np.ndarray,
    axis_termination: np.ndarray,
    radial_res: int = 5,
    long_res: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return directions and origins of rays used for initializing patellar cartilage.

    Creates a capsule-shaped geometry aligned with an approximate superior-inferior axis
    within or anterior to the bounding box of the patellar bone. The topological
    ordering of these rays is important downstream in the algorithm.

    Args:
        axis_origin (np.ndarray): The origin of the estimatd SI patellar bone axis.
        axis_termination (np.ndarray): The termiating point of the estimated SI patellar
            bone axis.
        radial_res (int, optional): The element resolution in the radial direction.
            Defaults to 5.
        axial_res (int, optional): The element resolution in the axial direction.
            Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The ray directions,
        origins, the number of nodes in the radial direction, and the number of
        nodes in the axial direction.
    """
    directions, origins, n_radial_nodes, n_axial_nodes = femur_cartilage(
        axis_origin, axis_termination, radial_res, long_res
    )
    return directions, origins, n_radial_nodes, n_axial_nodes


def tibia_cartilage(
    ray_origin_plane_point: Tuple[float],
    reconstruction: Union[trimesh.Trimesh, pv.PolyData],
    elem_counts: Union[Tuple[int], int] = None,
    elem_sizes: Union[Tuple[float], float] = None,
    bounding_box_scaling_factor: float = None,
    **kwargs,
):
    """Find -Z oriented arrays through tibial cartilage.

    The rays will be distributed evenly within a central rectangle the same shape as the
    oriented bounding square of the reconstruction after xy-plane projection. With a
    radial distribution from the boundary nodes of this rectangle. Looking along the
    direction of the rays, the pattern will look like a spiderweb of radial points
    surrounding the rectangular meshgrid center.

    Args:
        ray_origin_plane_point (Tuple[float]): (3,) point lying in the plane of origin
            points of the rays.
        reconstruction (Trimesh, PolyData): the triangular mesh reconstruction of the
            tibial cartilage geometry.
        elem_counts (int, Tuple): the number of elements in long, short, and radial
            axes. If a single integer, then the uniform number of points is assigned to
            all three axes. Defaults to None.
        elem_size (float, Tuple): the target element sizes in the long, short, and
            radial axes. If the passed value is a single float, assign uniform size to
            all three axes. Defaults to None.
        bbox_scaling_factor (float): The scaling factor applied to the bounding
            rectangle of the geometry projected onto the z-plane, before generating the
            central brick of points. Defaults to 0.35 (35%).

    Returns:
        Tuple[np.ndarray, Tuple[int]]: (n, 3) arrays of direction vectors, origin
        points, and the resulting element counts in the long and short axes of the
        internal bricks, and the element counts in the radial direction.
    """
    reconstruction = reconstruction.copy()
    if elem_sizes is not None:
        points, elem_counts = distribution(
            reconstruction,
            elem_size=elem_sizes,
            bbox_scaling_factor=bounding_box_scaling_factor,
            **kwargs,
        )
    elif elem_counts is not None:
        points, elem_counts = distribution(
            reconstruction,
            elem_counts=elem_counts,
            bbox_scaling_factor=bounding_box_scaling_factor,
            **kwargs,
        )
    origins = np.tile(ray_origin_plane_point, (points.shape[0], 1))
    origins[:, :2] = points
    directions = np.zeros_like(origins)
    directions[:, -1] = -1

    # plot the ray paths going through the cartilage
    if kwargs.get("debug", False):
        scene(
            reconstructions=[reconstruction],
            rays=[(origins, directions)],
            opacities=[0.5, 0.5, 0.5],
        ).show()
    return directions, origins, elem_counts


def distribution(
    reconstruction: Union[trimesh.Trimesh, pv.PolyData],
    bbox_scaling_factor: float = None,
    elem_size: Union[float, Tuple[float, float, float]] = None,
    elem_counts: Union[int, Tuple[int, int, int]] = None,
    **kwargs,
) -> np.ndarray:
    """Distribute hex element node locations within medial or lateral tibial cartilages.

    Nodes are distributed within a central brick of well-conditioned hexahedral
    elements. Raytracing is then performed in a radial pattern from the edges of the
    central brick (but not the top and bottom), to surround the central brick with
    uniformly skewed hexahedral elements, like a spiderweb of nodes with a rectangular
    center.

    Args:
        reconstruction (Trimesh, PolyData): the triangular mesh reconstruction of the
            tibial cartilage geometry.
        bbox_scaling_factor (float): The scaling factor applied to the bounding
            rectangle of the geometry projected onto the z-plane, before generating the
            central brick of points. Defaults to 0.35 (35%).
        elem_size (float, Tuple): the target element sizes in the long, short, and
            radial axes. If the passed value is a single float, assign uniform size to
            all three axes. Defaults to None.
        elem_counts (int, Tuple): the number of elements in long, short, and radial
            axes. If a single integer, then the uniform number of points is assigned to
            all three axes. Defaults to None.

    Returns:
        Tuple[np.ndarray, Tuple[int]]: (n, 3) ordered array of nodal locations and the
        resulting element counts in the long and short axes of the internal bricks, and
        the element counts in the radial direction.
    """
    bbox_scaling_factor = 0.35 if bbox_scaling_factor is None else bbox_scaling_factor
    target_tri_size = 0.4
    normal_dot_limit = 0.34

    projected, outline_bounds = _z_plane_projection(
        reconstruction, target_tri_size, normal_dot_limit, **kwargs
    )

    bbox_points, indices, counts, sizes = _scaled_bsquare_meshgrid(
        projected=projected,
        elem_size=elem_size,
        elem_counts=elem_counts,
        bbox_scaling_factor=bbox_scaling_factor,
        **kwargs,
    )

    rotation_matrix = _get_projected_tibia_rotation_2D(projected)

    (
        rotated_center_brick_node_field,
        rotated_dist_nodes,
        rotated_ray_directions,
    ) = _get_node_field(
        indices, counts, bbox_points, outline_bounds, rotation_matrix, **kwargs
    )

    origins, arrow_heads, magnitudes, hits = _raytrace_bounds(  # noqa
        rotated_center_brick_node_field,
        rotated_ray_directions,
        outline_bounds,
        projected,
        **kwargs,
    )

    if sizes is not None:
        radial_size = sizes["radial"]
        n_radial_elems = np.ceil((magnitudes / radial_size).max()).astype(int) - 1
    elif counts is not None:
        n_radial_elems = counts["radial"]
    radial_nodes = np.linspace(origins, arrow_heads, counts["radial"] + 1, axis=1)
    radial_stack = radial_nodes.reshape(-1, 2)

    if kwargs.get("debug", False):
        ax = plt.gca()
        trimesh.load_path(outline_bounds).show()
        ax.scatter(*rotated_dist_nodes.T)
        ax.scatter(*radial_nodes.T)
        for idx, line in enumerate(np.array(hits)):
            ax.plot(*line, color="r")
            if idx > 20:
                break
        if kwargs.get("debug_breaks", False):
            breakpoint()
    distribution_nodes = np.vstack((rotated_dist_nodes, radial_stack))

    short_idx = indices["short"]
    n_short_side_elems, n_long_side_elems = counts["short"], counts["long"]
    top_bottom_n_elems = n_short_side_elems if short_idx == 0 else n_long_side_elems
    left_right_n_elems = (
        n_long_side_elems
        if top_bottom_n_elems == n_short_side_elems
        else n_short_side_elems
    )
    # #endregion
    n_vert_elems, n_horz_elems = left_right_n_elems, top_bottom_n_elems
    del projected
    return distribution_nodes, (n_vert_elems, n_horz_elems, n_radial_elems)


def _z_plane_projection(
    reconstruction: Union[trimesh.Trimesh, pv.PolyData],
    target_tri_size: Union[int, float],
    normal_dot_limit: Union[int, float],
    **kwargs,
):
    """Project tibial cartilage reconstruction into the xy-plane and outline +Z faces.

    Args:
        reconstruction (Union[trimesh.Trimesh, pv.PolyData]): tibial cartilage
            triangular reconstruction.
        target_tri_size (Union[int, float]): the target element size for the
            intermediate mesh before deleting faces with a -Z normal.
        normal_dot_limit (Union[int, float]): limit on unit-vector-dot-product for
            determining +Z triangular faces. Needs to be on the interval [-1, 1], with
            values greater than 0.5 being nearer to +Z than -Z.

    Returns:
        Tuple[pv.PolyData, Polygon]: The xy-plane projected mesh with back-faces
        removed, and the outline of the projected mesh.
    """
    pvmesh = pv.wrap(reconstruction)
    cmesh = clustered_mesh(
        pvmesh,
        target_size=target_tri_size,
        repair_backend=kwargs.get("repair_backend", "auto"),
    )
    cmesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    cmesh_dot = np.dot(cmesh.cell_data["Normals"], (0, 0, 1))
    cmesh_front_cell_ids = np.flatnonzero(cmesh_dot > normal_dot_limit)
    extracted = cmesh.extract_cells(cmesh_front_cell_ids).extract_surface(
        algorithm=None
    )
    projected = extracted.project_points_to_plane()

    projected_tmesh = trimesh.Trimesh(
        faces=projected.faces.reshape(-1, 4)[:, 1:], vertices=projected.points
    )
    outline = polygons.projected(
        projected_tmesh, normal=(0, 0, 1), apad=target_tri_size / 10
    )
    if isinstance(outline, MultiPolygon):
        areas = np.array([x.area for x in outline.geoms])
        idx_max = np.argmax(areas)
        largest_area = areas[idx_max]
        try:
            if largest_area / projected_tmesh.area < 0.9:
                raise RaySectorError(
                    "Encountered MultiPolygon object with significant area loss."
                )
        finally:
            del reconstruction
        outline = outline.geoms[idx_max]

    outline_bounds = trimesh.load_path(list(outline.exterior.coords)).polygons_full[0]
    if kwargs.get("debug", False):
        polygons.plot(outline_bounds)
        trimesh.Scene([projected_tmesh.bounding_box_oriented, projected_tmesh]).show()
        if kwargs.get("debug_breaks", False):
            breakpoint()
    del cmesh, cmesh_dot, extracted, pvmesh
    return projected, outline_bounds


def _scaled_bsquare_meshgrid(
    projected: pv.PolyData,
    elem_size: Union[float, Tuple[float]],
    elem_counts: Union[int, Tuple[int]],
    bbox_scaling_factor: float = None,
    **kwargs,
) -> Tuple[np.ndarray, int, int]:
    """Create and scale a meshgrid of 2D points from the projected tibial cartilage.

    Always produces the grid using element counts, to ensure optimal coverage of the
    bounding square. If element sizes are specified, a linear step is first performed to
    calculate the number of points in the resulting meshgrid, but sizes are unlikely to
    be exact.

    Args:
        projected (pv.PolyData): projected +Z surface mesh of tibial cartilage.
        elem_size (Union[float, Tuple[float]]): desired size of elements in the long,
            short, and radial directions. If a single float is passed, that value is
            used for all directions.
        elem_counts (Union[int, Tuple[int]]): desired number of elements in the long,
            short, and radial directions. If a single integer, the same number of
            elements are used for each direction.
        bbox_scaling_factor (float, optional): the scaling factor applied to bounding
            square. Defaults to 1.

    Returns:
        Tuple[np.ndarray, int, int]:

          * the coordinates of the sample in the 3D cartesian system
          * a dictionary containing indices of the short, long, and flat axes
          * a dictionary containing number of elements produced along each axis
          * if elem_size was passed, a dictionary containing the optimized element
            sizes. None otherwise.
    """
    projected_tmesh = trimesh.Trimesh(
        faces=projected.faces.reshape(-1, 4)[:, 1:], vertices=projected.points
    )
    scaled_bbox = projected_tmesh.copy().apply_scale(bbox_scaling_factor).bounding_box
    bbox_extents = scaled_bbox.extents
    flat_idx, short_idx, long_idx = bbox_extents.argsort()
    if elem_size:
        elem_size = np.array(elem_size)
        elem_size = elem_size.repeat(3) if elem_size.size == 1 else elem_size
        long_size, short_size, radial_size = elem_size
        if short_idx == 0:
            counts = [
                np.arange(0, bbox_extents[i], s).size
                for i, s in zip(
                    [short_idx, long_idx, flat_idx], [short_size, long_size, 1]
                )
            ]
            n_short_side_elems, n_long_side_elems, _ = np.array(counts) - 1
        else:
            counts = [
                np.arange(0, bbox_extents[i], s).size
                for i, s in zip(
                    [long_idx, short_idx, flat_idx], [long_size, short_size, 1]
                )
            ]
            n_long_side_elems, n_short_side_elems, _ = np.array(counts) - 1
    elif elem_counts:
        n_long_side_elems, n_short_side_elems, n_radial_elems = elem_counts
        counts = np.ones(3, dtype=int)
        counts[short_idx] = n_short_side_elems + 1
        counts[long_idx] = n_long_side_elems + 1

    bbox_points = trimesh.PointCloud(
        trimesh.creation.box(extents=bbox_extents).bounding_box.sample_grid(
            count=counts
        )
    )
    bbox_points.apply_translation(projected_tmesh.bounding_box.centroid)
    # view the bounding box
    if kwargs.get("debug", False):
        trimesh.Scene(
            [
                projected_tmesh,
                bbox_points,
            ]
        ).show()
        if kwargs.get("debug_breaks", False):
            breakpoint()
    indices = dict(short=short_idx, long=long_idx, flat=flat_idx)
    if elem_counts:
        elem_counts = dict(
            short=n_short_side_elems, long=n_long_side_elems, radial=n_radial_elems
        )
        elem_sizes = None
    elif elem_size:
        elem_counts = dict(short=n_short_side_elems, long=n_long_side_elems)
        elem_sizes = dict(short=short_size, long=long_size, radial=radial_size)
    del projected
    return bbox_points, indices, elem_counts, elem_sizes


def _raytrace_bounds(
    rotated_center_brick_node_field: np.ndarray,
    rotated_ray_directions: np.ndarray,
    outline_bounds: Polygon,
    projected: pv.PolyData,
    **kwargs,
):
    """Create rays from the internal meshgrid, to the projection outline intersections.

    Args:
        rotated_center_brick_node_field (np.ndarray): meshgrid of the central rectangle.
        rotated_ray_directions (np.ndarray): angles of each array in radians.
        outline_bounds (Polygon): the outline of the xy-projected tibial cartilage mesh.
        projected (pv.PolyData): the xy-projected tibial cartilage mesh.

    Returns:
        Tuple[np.ndarray]: arrays of vector origins, terminations, magnitudes, and
            shapely Line objects of the intersection vectors.
    """
    projected_tmesh = trimesh.Trimesh(
        faces=projected.faces.reshape(-1, 4)[:, 1:], vertices=projected.points
    )
    center_xy = projected_tmesh.bounding_box.centroid[:2]
    ray_origins = np.concatenate(
        [
            rotated_center_brick_node_field[:, :, 0].T,  # lowest vert
            rotated_center_brick_node_field[:, -1, :].T,  # highest horz
            np.fliplr(rotated_center_brick_node_field[:, :, -1]).T,  # highest vert
            np.fliplr(rotated_center_brick_node_field[:, 0, :]).T,  # lowest horz
        ]
    )
    starts = ray_origins
    stops = starts + 100 * rotated_ray_directions
    hits = [None] * rotated_ray_directions.shape[0]
    origins = hits.copy()
    arrow_heads = hits.copy()
    for ray_idx, (start, stop) in enumerate(zip(starts, stops)):
        ray = LineString((start, stop))
        intersects = outline_bounds.intersection(ray)
        if isinstance(intersects, MultiLineString):
            temporary_hits = np.array([i.xy for i in intersects.geoms])
            temp_origins = temporary_hits[:, :, 0]
            temp_ends = temporary_hits[:, :, 1]
            near_idx = np.linalg.norm(temp_origins - center_xy, axis=1).argmin()
            far_idx = np.linalg.norm(temp_ends - center_xy, axis=1).argmax()
            final_origin = temp_origins[near_idx]
            final_end = temp_ends[far_idx]
            hits[ray_idx] = np.vstack((final_origin, final_end)).T
        else:
            hits[ray_idx] = np.array(intersects.xy)

        if hits[ray_idx].size > 0:
            origins[ray_idx] = hits[ray_idx][:, 0]
            arrow_heads[ray_idx] = hits[ray_idx][:, 1]
        else:
            origins[ray_idx] = np.full(2, np.nan)
            arrow_heads[ray_idx] = np.full(2, np.nan)
    arrow_heads = np.array(arrow_heads)
    origins = np.array(origins)
    magnitudes = np.linalg.norm(arrow_heads - origins, axis=1)
    arrow_heads = (
        origins + (0.99 * magnitudes[..., np.newaxis]) * rotated_ray_directions
    )
    del projected
    return origins, arrow_heads, magnitudes, hits


def _get_projected_tibia_rotation_2D(
    projected: pv.PolyData,
) -> np.ndarray:
    """Find 2D rotation matrix rotating from axis-aligned to oriented bounding boxes.

    Args:
        projected (pv.PolyData): The xy-projected tibial cartilage surface mesh.

    Returns:
        np.ndarray: 2D rotation matrix.
    """
    projected_tmesh = trimesh.creation.extrude_triangulation(
        projected.points[:, :2], projected.faces.reshape(-1, 4)[:, 1:], 1
    )

    # Remove faces to make square
    bbox_align = projected_tmesh.bounding_box
    bbox_orient = projected_tmesh.bounding_box_oriented
    corners_align = tri_bounds.corners(bbox_align.bounds)[:4, :2]
    half = bbox_orient.primitive.extents / 2
    corners_orient = trimesh.transform_points(
        tri_bounds.corners([-half, half]), bbox_orient.primitive.transform
    )
    corners_orient = np.squeeze(np.sort(corners_orient[np.newaxis, :, :2], axis=0))[::2]

    # Find two longest vectors in the squares (hypotenuses), sum the two that aren't
    # opposite
    align_combos = np.array(list(combinations(corners_align, 2)))
    longest = np.linalg.norm(
        np.squeeze(np.diff(align_combos, axis=1)), axis=1
    ).argsort()[-2:]
    aligned_bias_vecs = unitize(np.squeeze(np.diff(align_combos[longest], axis=1)))
    if np.dot(*aligned_bias_vecs) > 0.5:
        aligned_bias_vecs[1] *= -1
    aligned_vec = unitize(np.sum(aligned_bias_vecs, axis=0))

    orient_combos = np.array(list(combinations(corners_orient, 2)))
    longest = np.linalg.norm(
        np.squeeze(np.diff(orient_combos, axis=1)), axis=1
    ).argsort()[-2:]
    orient_bias_vecs = unitize(np.squeeze(np.diff(orient_combos[longest], axis=1)))
    if np.dot(*orient_bias_vecs) > 0.5:
        orient_bias_vecs[1] *= -1
    orient_vec = unitize(np.sum(orient_bias_vecs, axis=0))
    del projected
    return shortest_align_vectors(aligned_vec, orient_vec)


def _get_node_field(
    indices: Dict,
    counts: Dict,
    bbox_points: trimesh.PointCloud,
    outline_bounds: Polygon,
    rotation_matrix: ArrayLike = None,
    **kwargs,
):
    """Create vertices of inner brick elements and associate angles with boundary nodes.

    Args:
        indices (Dict): indices representing the long, and short axes of the projected
            bounding box.
        counts (Dict): the element counts along the long, and short axes of the tibial
            mesh. Only the long and short axes are used.
        bbox_points (trimesh.PointCloud): projected bounding box meshgrid.
        outline_bounds (Polygon): the outline of the projected tibial cartilage.
        rotation_matrix (ArrayLike, optional): A 2D rotation matrix used to rotate the
            resulting node field. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: the node field, the (n, 3) nodal
        locations, and the unit vector cartesian directions of the associated angles.
    """
    rotation_matrix = rotation_matrix if rotation_matrix is not None else np.eye(2)
    short_idx = indices["short"]
    n_short_side_elems, n_long_side_elems = counts["short"], counts["long"]
    nodes_x = np.unique(bbox_points.vertices[:, 0])[::-1]
    nodes_y = np.unique(bbox_points.vertices[:, 1])[::-1]
    center_brick_node_field = np.stack(np.meshgrid(nodes_x, nodes_y, indexing="ij"))
    distribution_nodes = center_brick_node_field.reshape(2, -1, order="F").copy().T
    if short_idx == 0:
        top_bottom_n_elems = n_short_side_elems
        left_right_n_elems = n_long_side_elems
    else:
        top_bottom_n_elems = n_long_side_elems
        left_right_n_elems = n_short_side_elems
    angles = np.deg2rad(
        np.concatenate(
            [
                np.linspace(45, 135, top_bottom_n_elems + 1),  # north sector
                np.linspace(135, 225, left_right_n_elems + 1),  # west sector
                np.linspace(225, 315, top_bottom_n_elems + 1),  # south sector
                np.linspace(315, 405, left_right_n_elems + 1),  # east sector
            ]
        )
    )
    projected_centroid = np.array(outline_bounds.centroid.xy).flatten()
    bbox_centroid = bbox_points.centroid[:2]
    ray_directions = np.stack([np.cos(angles), np.sin(angles)]).T
    rotate_mat_2d = rotation_matrix
    rotated_dist_nodes = (
        rotate_mat_2d @ (distribution_nodes - bbox_centroid).T
    ).T + projected_centroid
    rotated_ray_directions = (rotate_mat_2d @ ray_directions.T).T
    rotated_center_brick_node_field = rotated_dist_nodes.T.reshape(
        *center_brick_node_field.shape, order="F"
    )
    if kwargs.get("debug", False):
        import matplotlib.pyplot as plt

        # Plot the outline
        polygons.plot(outline_bounds)
        ax = plt.gca()
        ax.scatter(*rotated_dist_nodes.T)
        # Plot the top nodes
        ax.scatter(*rotated_dist_nodes[: top_bottom_n_elems + 1].T)
        # Plot the bottom nodes
        ax.scatter(*rotated_dist_nodes[-top_bottom_n_elems - 1 :].T)
        plt.show()
        if kwargs.get("debug_breaks", False):
            breakpoint()
    return rotated_center_brick_node_field, rotated_dist_nodes, rotated_ray_directions


def find_surface_vertices(
    ray_directions: np.ndarray,
    ray_origins: np.ndarray,
    mesh: trimesh.Trimesh,
    max_degree_redirection: Union[int, float] = 30,
    neighbor_radius: Union[int, float] = 5,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform raytracing through a cartilage mesh to find surface intersections.

    Args:
        ray_directions (np.ndarray): The ray direction vectors in an (n, 3) array.
        ray_origins (np.ndarray): The ray origin points in an (n, 3) array.
        mesh (trimesh.Trimesh): trimesh object of the cartilage reconstruction.
        near_radius (int, optional): the radius used to define neighboring rays.
            Locations within this distance from each other will be used in calculating
            average ray directions Defaults to 5.
        max_degree_deviation (int): the maximum angle allowed between original and
            redirected ray directions. Defaults to 30.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The surface intersection points of the interior
        (first) and exterior (second) cartesian locations where the rays intersected the
        mesh.
    """
    # Perform raytracing to find where the rays intersect the inner surface
    report("Raytracing inner surface", total=len(ray_origins), unit="rays")
    try:
        inner_tri_idx, inner_ray_idx, inner_locations = mesh.ray.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            return_locations=True,
            multiple_hits=False,
        )
    except Exception as e:
        msg = e.__repr__()
        raise RaySectorError(
            f"Try reducing bounding box scaling factor. \nOriginal message: {msg}"
        )

    report(
        "Raytracing inner surface", len(ray_origins), len(ray_origins), unit="rays"
    )
    if kwargs.get("debug", False):
        il = trimesh.PointCloud(inner_locations)
        il.visual.vertex_colors = [255, 128, 128]
        rays_paths = rays_to_paths(
            ray_origins[inner_ray_idx][::5],
            ray_directions[inner_ray_idx][::5],
            ray_length=3 / 5 * mesh.extents.mean(),
        )
        trimesh.Scene([rays_paths, mesh, il]).show()

    inner_locations, outer_ray_directions = _redirect_internal_rays(
        mesh,
        inner_tri_idx,
        inner_locations,
        ray_directions[inner_ray_idx],
        max_degree_deviation=max_degree_redirection,
        near_radius=neighbor_radius,
    )

    if kwargs.get("debug", False):
        rays_paths = rays_to_paths(
            inner_locations[::4],
            outer_ray_directions[::4],
            ray_length=1 * mesh.extents.mean(),
        )
        il = trimesh.PointCloud(inner_locations)
        il.visual.vertex_colors = [254, 128, 128]
        trimesh.Scene([rays_paths, mesh, il]).show()

    report("Raytracing outer surface", total=len(inner_locations), unit="rays")
    _, outer_ray_idx, potential_outer_locations = mesh.ray.intersects_id(
        ray_origins=inner_locations,
        ray_directions=outer_ray_directions,
        return_locations=True,
        multiple_hits=True,
    )

    report(
        "Raytracing outer surface", len(inner_locations), len(inner_locations),
        unit="rays",
    )
    # Find the furthest rays
    outer_locations = np.full_like(inner_locations, np.nan)
    for ray_idx in WorkProgress(
        np.unique(outer_ray_idx), desc="Selecting surface intersections", unit="rays"
    ):
        has_hit = outer_ray_idx == ray_idx
        hit_locations = potential_outer_locations[has_hit]
        ray_length = np.linalg.norm((hit_locations - inner_locations[ray_idx]), axis=1)
        if ~np.isclose(ray_length, 0).all():
            outer_locations[ray_idx] = hit_locations[ray_length.argmax()]

    # Handle edge hits
    is_missing = np.isnan(outer_locations)
    outer_locations[is_missing] = (
        inner_locations[is_missing] + 0.1 * outer_ray_directions[is_missing]
    )

    interior_points = np.full_like(ray_origins, np.nan)
    exterior_points = np.full_like(ray_origins, np.nan)
    interior_points[inner_ray_idx] = inner_locations
    exterior_points[inner_ray_idx] = outer_locations

    if kwargs.get("debug", False):
        il = trimesh.PointCloud(inner_locations)
        il.visual.vertex_colors = [255, 128, 128]

        ol = trimesh.PointCloud(outer_locations)
        ol.visual.vertex_colors = [128, 128, 255]

        trimesh.Scene([mesh, il, ol]).show()

    return interior_points, exterior_points


def _redirect_internal_rays(
    mesh: trimesh.Trimesh,
    inner_tri_idx: np.ndarray,
    inner_locations: np.ndarray,
    internal_ray_directions: np.ndarray,
    near_radius: int = 5,
    max_degree_deviation: int = 30,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Redirect rays using the average direction of neighboring intersected faces.

    Neighboring faces are defined by inner surface ray intersection locations falling
    within a specified radius. The neighboring triangle normals are then superimposed
    and converted to unit vectors, becoming the redirected ray directions.

    Args:
        mesh (trimesh.Trimesh): the mesh which rays are being directed through.
        inner_tri_idx (np.ndarray): the (n, ) triangle IDs of the first ray-mesh
            intersections.
        inner_locations (np.ndarray): the (n, 3) cartesian locations where rays
            intersected the mesh.
        internal_ray_directions (np.ndarray): the (n, 3) direction vectors that will be
            redirected.
        near_radius (int, optional): the radius used to define neighboring rays.
            Locations within this distance from each other will be used in calculating
            average ray directions Defaults to 5.
        max_degree_deviation (int): the maximum angle allowed between original and
            redirected ray directions. Defaults to 30.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The outer ray origins, which are just the
        originally passed inner intersection locations, and the redirected directions.
    """
    # Create new rays based on the surface normals of the hit triangles
    # Average directions between the normal of the triangle and the previous direction
    hit_reversed_normals = -mesh.face_normals[inner_tri_idx]

    outer_ray_directions = np.full_like(hit_reversed_normals, np.nan)

    # Find the nearest ray_origins
    any_near = False
    while not any_near:
        if np.allclose([near_radius], [0]):
            break
        distances = squareform(pdist(inner_locations, "euclidean"))
        are_near = distances < near_radius
        any_near = np.any(are_near)
        if not any_near:
            near_radius *= 1.1
            warnings.warn(f"No nearby rays; near_radius increased to {near_radius}")

    # Find the new ray directions
    for idx, (is_near, distance) in enumerate(WorkProgress(
        zip(are_near, distances), total=len(are_near),
        desc="Redirecting rays", unit="rays",
    )):
        new_rays = hit_reversed_normals[is_near].sum(axis=0)
        outer_ray_directions[idx] = new_rays / np.linalg.norm(new_rays)

    angles = trimesh.geometry.vector_angle(
        np.stack((internal_ray_directions, outer_ray_directions), axis=1)
    )
    is_too_large = angles > np.deg2rad(max_degree_deviation)
    vector_interpolation = np.linspace(
        internal_ray_directions[is_too_large], outer_ray_directions[is_too_large]
    )
    check_against = np.tile(
        internal_ray_directions[is_too_large], (vector_interpolation.shape[0], 1, 1)
    )
    interp_angles = trimesh.geometry.vector_angle(
        np.stack(
            (vector_interpolation.reshape(-1, 3), check_against.reshape(-1, 3)), axis=1
        )
    ).reshape(50, np.count_nonzero(is_too_large))
    deltas = interp_angles - np.deg2rad(max_degree_deviation)
    deltas[deltas > 0] = np.nan
    correct_idxs = np.nanargmax(deltas, axis=0)
    np.rad2deg(interp_angles[correct_idxs, np.arange(correct_idxs.size)])
    new_vectors = vector_interpolation[correct_idxs, np.arange(correct_idxs.size)]
    outer_ray_directions[is_too_large] = new_vectors
    outer_ray_origins = inner_locations
    return outer_ray_origins, outer_ray_directions


@deprecation.deprecated(
    deprecated_in="0.001",
    removed_in="1.0",
    current_version=__version__,
    details="Use trimesh and shapely tools as seen in distribution()",
)
def quad_patch(p0, p1, p2, p3, ratio_h, ratio_v):  # deprecated
    v1 = np.linspace(p0, p1, ratio_h + 1).T
    v2 = np.linspace(p2, p3, ratio_h + 1).T
    V = np.full((2, ratio_v + 1, ratio_h + 1), np.nan)

    for i in range(ratio_h + 1):
        V[:, :, i] = np.linspace(v1[:, i], v2[:, i], ratio_v + 1).T
    return V


@deprecation.deprecated(
    deprecated_in="0.001",
    removed_in="1.0",
    current_version=__version__,
    details="Use `numpy.linspace` with an axis argument instead.",
)
def divide_size(p0, p1, ratio):  # deprecated
    p0 = np.array(p0)[np.newaxis]
    p1 = np.array(p1)[np.newaxis]

    step = (p1 - p0) / ratio
    segment_len = np.arange(0, ratio + 1)[np.newaxis]

    x = segment_len * step[0][0]
    y = segment_len * step[0][1]
    t = np.concatenate((x, y), axis=0)

    p0 = p0[np.newaxis]
    p0 = p0.reshape(2, 1)

    t_siz = t.shape[1]

    pp = np.tile(p0, (1, t_siz)) + t
    return pp


@deprecation.deprecated(
    deprecated_in="0.001",
    removed_in="1.0",
    current_version=__version__,
    details="Use trimesh and shapely tools as seen in distribution()",
)
def line_segment_intersect(xy1, xy2):  # deprecated
    # # add parameter checking with a try-catch exception setup?
    # validateattributes(xy1,{'numeric'},{'2d','finite'})
    # validateattributes(xy2,{'numeric'},{'2d','finite'})

    xy1 = np.atleast_2d(xy1)
    xy2 = np.atleast_2d(xy2)

    n_rows_1 = xy1.shape[0]
    n_cols_1 = xy1.shape[1]

    n_rows_2 = xy2.shape[0]
    n_cols_2 = xy2.shape[1]

    if n_cols_1 != 4 or n_cols_2 != 4:
        raise Exception(
            "Params for line_segment_intersect, xy1 and xy2, must be 2D matrices "
        )

    # Prepare matrices for vectorized computation of line intersection points.
    x1 = np.tile(xy1[:, 0], (1, n_rows_2)).T
    x2 = np.tile(xy1[:, 2], (1, n_rows_2)).T
    y1 = np.tile(xy1[:, 1], (1, n_rows_2)).T
    y2 = np.tile(xy1[:, 3], (1, n_rows_2)).T

    xy2 = xy2.T

    x3 = np.tile(xy2[0, :], (1, n_rows_1)).T
    x4 = np.tile(xy2[2, :], (1, n_rows_1)).T
    y3 = np.tile(xy2[1, :], (1, n_rows_1)).T
    y4 = np.tile(xy2[3, :], (1, n_rows_1)).T

    x4_x3 = x4 - x3
    y1_y3 = y1 - y3
    y4_y3 = y4 - y3
    x1_x3 = x1 - x3
    x2_x1 = x2 - x1
    y2_y1 = y2 - y1

    numerator_a = x4_x3 * y1_y3 - y4_y3 * x1_x3
    numerator_b = x2_x1 * y1_y3 - y2_y1 * x1_x3
    denominator = y4_y3 * x2_x1 - x4_x3 * y2_y1

    # FIXME: Check for divide by zero!!!! Debug 9007827/M12
    u_a = numerator_a / denominator
    u_b = numerator_b / denominator

    # Find the adjacency matrix A of intersecting lines.
    int_x = x1 + x2_x1 * u_a
    int_y = y1 + y2_y1 * u_a

    n = int_x.shape[0]
    int_b = []
    for i in range(n):
        all_true = (
            (u_a[i, 0] >= 0)
            and (u_a[i, 0] <= 1)
            and (u_b[i, 0] >= 0)
            and (u_b[i, 0] <= 1)
        ).astype(int)
        int_b.append(all_true)
    int_b = np.array(int_b)[np.newaxis].T

    par_b = (denominator == 0).astype(
        int
    )  # converts boolean True and False to 1 and 0.

    n = numerator_a.shape[0]
    conic_b = []
    for i in range(n):
        true_row = (
            (numerator_a[i, 0] == 0) and (numerator_b[i, 0] == 0) and (par_b[i, 0] == 1)
        ).astype(int)
        conic_b.append(true_row)
    conic_b = np.array(conic_b)[np.newaxis].T

    # Arrange output.
    out = {}
    out["int_adjacency_matrix"] = int_b
    out["int_matrix_x"] = int_x * int_b
    out["int_matrix_y"] = int_y * int_b
    out["int_normalized_distance1_2"] = u_a
    out["int_normalized_distance2_1"] = u_b
    out["par_adjacency_matrix"] = par_b
    out["coinc_adjacency_matrix"] = conic_b
    return out
