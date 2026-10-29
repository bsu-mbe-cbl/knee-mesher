"""A module for code ported from the UCONN version which has since been replaced.

The replacements for these functions use significantly different approaches, but I did
not want to completely remove these functions until the 1.0 release.

"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import ConvexHull
from vtk import VTK_HEXAHEDRON, VTK_QUAD  # noqa

# import kneemesher.meshing.degenerates as degens
# import kneemesher.meshing.raytracing as rays
from kneemesher.meshing.cells import HexElement, HexMesh

# from kneemesher.preprocessing import clamp_triangle_count_and_smooth


def get_convex_hull_volume(
    nodes: np.ndarray,
    elems: np.ndarray,
) -> np.ndarray:
    """Calculate the smallest convex volume enclosing the mesh.

    Args:
        nodes (np.ndarray): an (n, 3) array of node locations.
        elems (np.ndarray): an (m, 8) array of element definitions composed of nodal
            IDs.

    Returns:
        np.ndarray: The smallest convex hull volume enclosing the mesh.
    """
    siz = elems.shape[0]
    volume = np.full(siz, np.nan)
    for idx in range(siz):
        X = np.full((8, 3), np.nan)
        for jj in range(8):
            X[jj, :] = nodes[elems[idx, jj], :]
        volume[idx] = ConvexHull(points=X, qhull_options="Qt").volume
    return volume


def subdivide_cartilage_depth(
    node_locations: np.ndarray,
    element_definitions: np.ndarray,
    reverse_direction: bool = False,
    n_layers: int = 4,
    use_uconn_method: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subdivide a single-element-deep layer of elements.

    Args:
        node_locations (np.ndarray): a (n, 3) array of nodal locations.
        element_definitions (np.ndarray): a (m, 8) array of nodal IDs.
        n_layers (int, optional): the final element depth after subdivision.  Defaults
            to 4.
        reverse_direction (bool, optional): flag to reverse the layer indexes. Defaults
            to False.
        use_uconn_method (bool, optional): Whether to use the UCONN method. Defaults to
            True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: the subdivided nodal locations,
        subdivided element definitions, and a 1-D array of nodal layer indexes.
    """
    # Define the number of subdivisions
    use_uconn_method = True
    n_elems = element_definitions.shape[0]
    layer_idx = np.hstack([x * np.ones(n_elems, dtype=int) for x in range(4)])
    layer_idx = np.flipud(layer_idx) if reverse_direction else layer_idx

    if use_uconn_method:
        # Create the point clouds
        n_nodes_per_swept_elem = 4 * (n_layers + 1)  # 4 nodes per swept face
        points = np.full((n_nodes_per_swept_elem, n_elems, 3), np.nan)
        points[:8] = np.moveaxis(node_locations[element_definitions[:, :8]], 1, 0)
        node_layer_idxs = np.arange(8, 20).reshape(-1, 3)
        # Create the intermediate nodes, concatenating them to the bottom of the point
        # array
        bot_points = points[:4]
        top_points = points[4:8]
        points[node_layer_idxs.T] = np.linspace(bot_points, top_points, 5)[1:-1]

        origin_node_ids = np.arange(n_elems)
        offset = (n_nodes_per_swept_elem - 1) * n_elems
        node_ids = np.linspace(
            origin_node_ids,
            origin_node_ids + offset,
            n_nodes_per_swept_elem,
            dtype=int,
        )

        # Create the new hexahedra representing four equally sized elements
        indices = np.arange(20).reshape(-1, 4)
        indices = np.vstack((indices[0], indices[2:].reshape(-1, 3).T, indices[1]))
        new_elems = [None] * 4
        for count, layer_node_idxs in enumerate(indices[1:]):
            prev_node_idxs = indices[count]
            new_elems[count] = np.vstack(
                (node_ids[prev_node_idxs], node_ids[layer_node_idxs])
            ).T
        new_elems = np.vstack(new_elems)

        layer_idx = np.hstack([x * np.ones(n_elems, dtype=int) for x in range(4)])
        layer_idx = np.flipud(layer_idx) if reverse_direction else layer_idx

        # Calculate N
        new_node_locs = np.vstack(points)

        new_node_locs, layer_node_idxs = np.unique(
            new_node_locs, axis=0, return_inverse=True
        )
        new_elems = layer_node_idxs[new_elems]
        return new_node_locs, new_elems, layer_idx


class BorderElement:
    def __init__(
        self,
        elem_idx: int,
        elems: np.ndarray,
        top_node_idxs: np.ndarray,
        bot_node_idxs: np.ndarray,
        normals: np.ndarray,
    ) -> None:
        self.elem_idx = elem_idx
        self.elem = elems[elem_idx]
        self.top_node_idxs = top_node_idxs
        self.bot_node_idxs = bot_node_idxs
        self.normals = normals


def expand_border_elements(
    mesh: trimesh.Trimesh,
    nodes: np.ndarray,
    elems: np.ndarray,
    max_node_shift: Union[int, float] = None,
    only_edges: bool = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # FIXME: Rodriguez-Vila made it so max_node_shift is always overwritten if mask
    # is None. Does not look intentional
    if only_edges is None or max_node_shift is None:
        max_node_shift = 5  # uconn
        only_edges = True

    FACE_NODE_MAPS = get_face_node_mask(only_edges)
    # Expand the hexahedral mesh until the nodes are in contact with the STL triangular
    # model formed by vertices and face
    femur = HexMesh(elems, nodes)

    def process_border_elems(nodes, elems):
        mesh = HexMesh(elems, nodes)
        # Should return node numbers in sets of four (n, 2, 2)
        is_external_node = np.full_like(elems, False, dtype=bool)
        is_internal_node = np.full_like(elems, False, dtype=bool)
        border_face_normals = np.full((elems.shape[0], 3), np.nan)
        np.full((elems.shape[0], 2, 2), np.nan)
        border_elems = []

        perim_edges, perim_ribbon = mesh.perimeter_edges
        perim_cells = mesh.perimeter_cells
        perim_cells["vtkOriginalCellIds"]
        perim_edges["vtkOriginalPointIds"]

        for elem_idx, elem in enumerate(elems):
            shares_nodes = np.isin(elems, elem)
            shares_quad_face = np.flatnonzero(shares_nodes.sum(axis=1) == 4)
            n_shared_quads = shares_quad_face.size
            if n_shared_quads < 4:  # then it's a border element
                face = np.full((n_shared_quads, 8), False)
                for face_idx, shared_elem_idx in enumerate(shares_quad_face):
                    has_face_nodes = np.isin(elem, elems[shared_elem_idx])
                    face[face_idx, :] = has_face_nodes

                # Find the faces that are not shared
                is_in_mask = (
                    (face == FACE_NODE_MAPS[:, np.newaxis]).all(axis=2).any(axis=1)
                )
                border_face_idxs = np.flatnonzero(~is_in_mask)

                # Compute the normal vector to that face in the In and Out nodes
                hex_elem = HexElement(points=nodes[elem])
                face_node_idxs = np.nonzero(FACE_NODE_MAPS[border_face_idxs])[
                    1
                ].reshape(-1, 4)
                normal = hex_elem.face_normals(face_node_idxs, use_uconn_method=True)
                if normal.size > 4:
                    normal = normal.mean(axis=0)

                if ~np.isnan(normal[0]):
                    top_node_idxs = np.arange(4)
                    bot_node_idxs = np.arange(4, 8)
                    has_top_node = np.isin(face_node_idxs, top_node_idxs)
                    has_bot_node = np.isin(face_node_idxs, bot_node_idxs)
                    top_node_ids = elem[face_node_idxs[has_top_node]].ravel()
                    bot_node_ids = elem[face_node_idxs[has_bot_node]].ravel()
                    border_face_normals[elem_idx, :] = normal
                    is_internal_node[elem_idx, face_node_idxs[has_top_node]] = True
                    is_external_node[elem_idx, face_node_idxs[has_bot_node]] = True
                    border_elems.append(
                        BorderElement(
                            elem_idx, elems, top_node_ids, bot_node_ids, normal
                        )
                    )
        return border_elems

    def add_face_rays(border_elems, nodes):
        border_elem_defs = np.vstack([x.elem for x in border_elems])
        normals = np.vstack([x.normals for x in border_elems])
        for b_idx, b_elem in enumerate(border_elems):
            in_nodes = b_elem.top_node_idxs
            in_origins = nodes[in_nodes]
            in_directions = np.zeros((in_nodes.size, 3))
            for n_idx, n in enumerate(in_nodes):
                has_nodes = np.isin(border_elem_defs, n).any(axis=1)
                in_directions[n_idx] = normals[has_nodes].mean(axis=0)
            border_elems[b_idx].in_origins = in_origins
            border_elems[b_idx].in_directions = trimesh.unitize(in_directions)

            out_nodes = b_elem.bot_node_idxs
            out_origins = nodes[out_nodes]
            out_directions = np.zeros((out_nodes.size, 3))
            for n_idx, n in enumerate(out_nodes):
                has_nodes = np.isin(border_elem_defs, n).any(axis=1)
                out_directions[n_idx] = normals[has_nodes].mean(axis=0)
            border_elems[b_idx].out_origins = out_origins
            border_elems[b_idx].out_directions = trimesh.unitize(out_directions)

        return border_elems

    def add_face_displacements(border_elems, mesh, max_distance_to_shift):
        out_nodes = np.hstack([be.bot_node_idxs.ravel() for be in border_elems])
        out_nodes, out_node_idx = np.unique(out_nodes, return_index=True)
        out_origins = np.vstack([be.out_origins for be in border_elems])[out_node_idx]
        out_dirs = np.vstack([be.out_directions for be in border_elems])[out_node_idx]

        out_tri_idx, out_ray_idx, out_locations = mesh.ray.intersects_id(
            ray_origins=out_origins,
            ray_directions=out_dirs,
            return_locations=True,
            multiple_hits=False,
        )

        unit_out_dirs = trimesh.unitize(out_dirs)[out_ray_idx]
        intersect_face_normals = mesh.face_normals[out_tri_idx]
        angle_between = trimesh.geometry.vector_angle(
            np.stack((unit_out_dirs, intersect_face_normals), axis=1)
        )
        is_backface = np.cos(angle_between) > 0.5
        out_ray_idx = out_ray_idx[is_backface]
        out_locations = out_locations[is_backface]

        out_displacement_vectors = np.zeros_like(out_origins)
        out_displacement_vectors[out_ray_idx] = out_locations - out_origins[out_ray_idx]
        out_distances = np.linalg.norm(out_displacement_vectors, axis=1)
        needs_update = out_distances < max_distance_to_shift
        out_distances[~needs_update] = 0
        out_displacement_vectors[~needs_update] = np.zeros(3)

        in_nodes = np.hstack([be.top_node_idxs for be in border_elems])[out_node_idx]
        in_origins = np.vstack([be.in_origins for be in border_elems])[out_node_idx]
        in_dirs = np.vstack([be.in_directions for be in border_elems])[out_node_idx]

        in_tri_idx, in_ray_idx, in_locations = mesh.ray.intersects_id(
            ray_origins=in_origins,
            ray_directions=in_dirs,
            return_locations=True,
            multiple_hits=False,
        )

        unit_in_dirs = trimesh.unitize(in_dirs)[in_ray_idx]
        intersect_face_normals = mesh.face_normals[in_tri_idx]
        angle_between = trimesh.geometry.vector_angle(
            np.stack((unit_in_dirs, intersect_face_normals), axis=1)
        )
        is_backface = np.cos(angle_between) > 0.5
        in_ray_idx = in_ray_idx[is_backface]
        in_locations = in_locations[is_backface]

        in_displacement_vectors = np.zeros_like(in_origins)
        in_displacement_vectors[in_ray_idx] = in_locations - in_origins[in_ray_idx]
        in_distances = np.linalg.norm(in_displacement_vectors, axis=1)
        needs_update = in_distances < max_distance_to_shift
        in_distances[~needs_update] = 0
        in_displacement_vectors[~needs_update] = np.zeros(3)

        max_displacements = np.zeros_like(in_displacement_vectors)
        in_is_further = in_distances > out_distances
        max_displacements[in_is_further] = in_displacement_vectors[in_is_further]
        max_displacements[~in_is_further] = out_displacement_vectors[~in_is_further]
        return in_nodes, out_nodes, max_displacements

    border_elems = process_border_elems(nodes, elems)
    border_elems = add_face_rays(border_elems, nodes)
    out_nodes_idxs, in_node_idxs, max_displacements = add_face_displacements(
        border_elems, mesh, max_node_shift
    )

    disp_vectors = np.zeros_like(nodes)
    disp_vectors[out_nodes_idxs] = max_displacements
    disp_vectors[in_node_idxs] = max_displacements
    disp_vectors = smooth_hex_mesh(elems, disp_vectors, "surface", 0)

    # TODO: wrap in debug clause
    debug = False
    if debug:
        orig_out = np.vstack([be.out_origins for be in border_elems])
        rays_out = np.vstack([be.out_directions for be in border_elems])
        u_orig_out, u_orig_inv = np.unique(orig_out, axis=0, return_inverse=True)
        for uoo in u_orig_out:
            has_uoo = (orig_out == uoo).any(axis=1)
            rays_out[has_uoo] = rays_out[has_uoo].mean(axis=0)
        femur.plotter.add_mesh(pv.wrap(mesh), color="red", opacity=0.7)
        matlab_out_ray_mesh = pv.PolyData(orig_out)
        femur.plotter.add_mesh(matlab_out_ray_mesh, color="white", point_size=10)
        matlab_out_ray_mesh = pv.PolyData(orig_out)
        matlab_out_ray_mesh.vectors = rays_out * 2
        femur.plotter.add_mesh(matlab_out_ray_mesh.arrows, color="white")

        orig_in = np.vstack([be.in_origins for be in border_elems])
        rays_in = np.vstack([be.in_directions for be in border_elems])
        u_orig_in, u_orig_inv = np.unique(orig_in, axis=0, return_inverse=True)
        for uoi in u_orig_in:
            has_uoi = (orig_in == uoo).any(axis=1)
            rays_in[has_uoi] = rays_in[has_uoi].mean(axis=0)
        matlab_in_ray_mesh = pv.PolyData(orig_in)
        femur.plotter.add_mesh(matlab_in_ray_mesh, color="yellow", point_size=10)
        matlab_in_ray_mesh = pv.PolyData(orig_in)
        matlab_in_ray_mesh.vectors = rays_in * 2
        femur.plotter.add_mesh(matlab_in_ray_mesh.arrows, color="yellow")

        new_femur = HexMesh(elems, nodes + disp_vectors)
        femur.plotter.add_mesh(new_femur.mesh, color="b", opacity=0.6)

        femur.show(node_normals=False)

    return nodes + disp_vectors


def get_face_node_mask(
    mask: bool,
) -> np.ndarray:
    base = np.full(8, False)
    base[np.array((0, 1, 4, 5))] = True
    if mask:
        face_node_maps = np.array(
            (
                base,  # Front: np.flatnonzero(base) => [0, 1, 4, 5]
                np.roll(base, 2),  # Back
                np.roll(base, -1),  # Left
                np.roll(base, 1),  # Right
            )
        )
    else:
        face_node_maps = np.array(
            (
                base,  # Front
                np.roll(base, 2),  # Back
                np.roll(base, -1),  # Left
                np.roll(base, 1),  # Right
                np.hstack((np.full(4, True), np.full(4, False))),  # Top
                np.hstack((np.full(4, False), np.full(4, True))),  # Bottom
            )
        )
    return face_node_maps
