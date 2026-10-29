"""Represent a single, first-order hexahedral element.

   .. code-block::

      (external)
          z
          |
          |
          4 ---------- 7
         /|           /|
        / |          / |
       5 -|-------- 6  |
       |  0 --------|- 3 ---------- y (long)
       | /          | /
       |/           |/
       1 ---------- 2
      /
     /
    x
(radial)
"""

from __future__ import annotations

import copy
import tracemalloc  # noqa
from collections import OrderedDict
from itertools import combinations
from typing import List, Tuple, Union

import matplotlib.pyplot as plt  # noqa
import numpy as np
import pyvista as pv
import trimesh
from numpy.typing import ArrayLike
from scipy.signal import savgol_filter  # noqa
from vtk import VTK_HEXAHEDRON, VTK_QUAD  # noqa

import kneemesher.meshing.degenerates as degens
from kneemesher import uconn
from kneemesher.errors import BlendingError, OptimizationError
from kneemesher.meshing.cells import HexElement, HexMesh, by_value
from kneemesher.plotting import ensure_hex_mesh, ensure_pyvista, scene  # noqa
from kneemesher.pv_helpers import UnstructuredGrid as UGrid



class HexPlaneContainer:
    def __init__(self, **hex_planes: EmptyHexahedralPlane):
        self.hex_planes = OrderedDict({k: v.copy() for k, v in hex_planes.items()})
        self.offsets = np.zeros(len(self.hex_planes))

        combined_points = [None] * len(self.hex_planes)
        offset = 0
        for plane_idx, (k, v) in enumerate(self.hex_planes.items()):
            self.offsets[plane_idx] = offset
            self.hex_planes[k].cell_def += offset
            offset += self.hex_planes[k].n_points
            combined_points[plane_idx] = self.hex_planes[k].points
        self.points = np.vstack(combined_points).copy()
        self.n_points = self.points.shape[0]

        self.is_marked_for_removal = np.full(self.points.shape[0], False)

    @property
    def cell_defs(self):
        return np.vstack([v.cell_def for v in self.hex_planes.values()])

    @cell_defs.setter
    def cell_defs(self, defs):
        start = 0
        for k in self.hex_planes.keys():
            n_cells = self.hex_planes[k].n_cells
            self.hex_planes[k].cell_def = defs[start : start + n_cells]
            start += n_cells

    def mark_weld(
        self,
        main_name: str,
        main_axis: str,
        welded_name: str,
        welded_axis: str,
    ):
        welded_plane = self.hex_planes[welded_name]
        main_plane = self.hex_planes[main_name]
        assert main_axis in ["-x", "+x", "-y", "+y"]
        assert welded_axis in ["-x", "+x", "-y", "+y"]

        reversed_combos = (
            ("-y", "+y"),
            ("+y", "-y"),
            ("-y", "+x"),
            ("+x", "-y"),
            ("+y", "+x"),
            ("+x", "+y"),
            ("-x", "-x"),
        )
        combo = main_axis, welded_axis
        reverse = True if combo in reversed_combos else False
        welded_pids = welded_plane.get_border_point_ids(
            welded_axis, axis_swapped=reverse
        )
        main_pids = main_plane.get_border_point_ids(main_axis)

        check_str = "".join(combo)
        has_both_axes = "x" in check_str and "y" in check_str
        has_both_signs = "-" in check_str and "+" in check_str
        is_duplicate = combo == ("-x", "-x")
        if is_duplicate or (has_both_axes and has_both_signs):
            main_pids = np.flipud(main_pids)
        welded_plane.set_border_point_ids(welded_axis, main_pids, axis_swapped=reverse)

        # Assign any of those matches to the updated point ID
        cell_defs = self.cell_defs
        pid_remains = np.isin(welded_pids, cell_defs)
        if pid_remains.any():
            for old, new in zip(welded_pids[pid_remains], main_pids[pid_remains]):
                cell_defs[cell_defs == old] = new
            self.cell_defs = cell_defs

    def plot(self):
        p = pv.Plotter()
        p.add_mesh(
            pv.MultiBlock({k: v.grid for k, v in self.hex_planes.items()}),
            multi_colors=True,
        )
        p.add_legend(labels=list(self.hex_planes.keys()))
        p.add_axes()
        p.show()

    @property
    def welded_mesh(self):
        combined_cell_def = self.cell_defs
        return UGrid.from_cell_def(points=self.points, cell_def=combined_cell_def)


class EmptyHexahedralPlane:
    def __init__(self, n_x_cells: int, n_y_cells: int, transposed=False) -> None:
        xrng = np.linspace(0, 3, n_x_cells + 1)
        yrng = np.linspace(0, 3, n_y_cells + 1)
        zrng = np.linspace(0, 1, 2)
        grid = pv.RectilinearGrid(xrng, yrng, zrng).cast_to_unstructured_grid()
        self.cell_def = grid.cells_dict[11]
        self.cell_def[:, [2, 3, 6, 7]] = self.cell_def[:, [3, 2, 7, 6]]
        self.points = np.full_like(grid.points, np.nan)
        self.n_x_cells, self.n_y_cells = n_x_cells, n_y_cells
        self.n_x_points, self.n_y_points = n_x_cells + 1, n_y_cells + 1
        self.n_cells = grid.n_cells
        self.n_points = grid.n_points
        self.transposed = False
        self.shape = self.n_x_cells, self.n_y_cells

        self.border_cell_slices = {
            "-y": np.s_[0:n_x_cells],
            "+y": np.s_[n_x_cells * (n_y_cells - 1) : n_x_cells * n_y_cells],
            "-x": np.s_[0::n_x_cells],
            "+x": np.s_[n_x_cells - 1 :: n_x_cells],
        }

        self.border_point_slices = {
            "-y": np.s_[:, np.array((0, 1, 4, 5))],
            "+y": np.s_[:, np.array((2, 3, 6, 7))],
            "-x": np.s_[:, np.array((0, 3, 4, 7))],
            "+x": np.s_[:, np.array((1, 2, 5, 6))],
        }

        if transposed:
            self.cell_def = self.cell_def.T
            self.transposed = True

    def __str__(self) -> str:
        return f"EmptyHexahedralPlane(shape={self.shape})"

    def __repr__(self) -> str:
        return f"EmptyHexahedralPlane(shape={self.shape})"

    def show(self):
        self.grid.plot(show_edges=True)

    def copy(self):
        return copy.deepcopy(self)

    @property
    def grid(self):
        return UGrid.from_cell_def(
            points=self.points.copy(), cell_def=self.cell_def.copy()
        )

    def get_border_cell_def(self, axis_str: str):
        axis_str = axis_str.lower()
        assert axis_str in ["-x", "+x", "-y", "+y"]
        cslice = self.border_cell_slices
        return self.cell_def[cslice[axis_str]]

    def set_border_cell_def(self, axis_str: str, values: ArrayLike):
        axis_str = axis_str.lower()
        assert axis_str in ["-x", "+x", "-y", "+y"]
        cslice = self.border_cell_slices
        self.cell_def[cslice[axis_str]] = np.array(values)

    def get_border_point_ids(self, axis_str: str, axis_swapped: bool = False):
        axis_str = axis_str.lower()
        assert axis_str in ["-x", "+x", "-y", "+y"]
        cslice = self.border_cell_slices[axis_str]
        pslice = self.border_point_slices[axis_str]
        pslice = np.s_[:, pslice[1][np.array((1, 0, 3, 2))]] if axis_swapped else pslice
        return self.cell_def[cslice][pslice]

    def set_border_point_ids(
        self, axis_str: str, values: ArrayLike, axis_swapped: bool = False
    ):
        axis_str = axis_str.lower()
        assert axis_str in ["-x", "+x", "-y", "+y"]
        cslice = self.border_cell_slices[axis_str]
        pslice = self.border_point_slices[axis_str]
        pslice = np.s_[:, pslice[1][np.array((1, 0, 3, 2))]] if axis_swapped else pslice
        self.cell_def[cslice][pslice] = np.array(values)


def initialize_hexahedral_plane(  # init_hex_elem_plane_defs?
    n_radial_nodes: int,
    n_nodes: int,
    **kwargs,
) -> np.ndarray:
    """Initialize a plane of hexahedral element definitions.

    Element definitions are a sequential list of eight node ID numbers, with the first
    four IDs representing the top nodes, and the last four IDs representing the bottom
    nodes. Both the top and bottom nodes are ordered to have outward facing normals.

    Args:
        n_radial_nodes (int): The number of nodes along the radial axis.
        n_nodes (int): The total number of nodes in the hexahedral plane.

    Returns:
        np.ndarray: An integer array of node IDs with each column containing a single
        set of eight node IDs defining the element.

    """
    # We conceptualize the nodes_ids as two parallel planes in space, one plane for the
    # top faces of the elements, and the other for the bottom. Node IDs within each
    # plane are located at the grid intersections.
    first_elem_top_node_ids = np.array((0, 1, n_radial_nodes + 1, n_radial_nodes))
    first_elem_bottom_node_ids = first_elem_top_node_ids + (n_nodes // 2)
    first_elem_node_ids = np.hstack(
        (first_elem_top_node_ids, first_elem_bottom_node_ids)
    )

    # Each element definition begins by incrementing the node numbers of the previous
    # element.
    n_axial_nodes = n_nodes // 2 // n_radial_nodes
    n_elems = np.prod((n_radial_nodes - 1, n_axial_nodes - 1))
    elems = np.linspace(
        first_elem_node_ids,
        first_elem_node_ids + n_elems - 1,
        num=n_elems,
        dtype=np.array(n_nodes).dtype,
    )

    # To prevent the final element of one axial row from being connected to the start of
    # the first element of the next row, we increment all node numbers in the following
    # rows of elements by one.
    for start in np.arange(n_radial_nodes - 1, n_elems, n_radial_nodes - 1):
        elems[start:] += 1

    return elems.T


def select_hexahedral_plane_border(
    cell_defs: np.ndarray,
    normal: List[int],
    n_x_cells: int,
    n_y_cells: int,
) -> np.ndarray:
    """Select hexahedral plane border elements on a single side.

    > NOTE: This only works on an unaltered hexahedral plane, before any NaN nodes
    >       have been deleted.

    Args:
        cell_defs (np.ndarray): an (n_cell, 8) cell definition where each entry is a
            point id.
        normal (List[int]): an (1, 2) or (1, 3) colinear with the +/- x or +/- y axis.
        n_x_cells (int): the number of cells in the x (radial) direction.
        n_y_cells (int): number of cells in the y (longitudinal) direction.

    Returns:
        np.ndarray: the (n_x_cell, 8) or (n_y_cell, 8) indexed cell definition.

    """
    pos_x = [1, 0, 0]
    neg_x = [-1, 0, 0]
    pos_y = [0, 1, 0]
    neg_y = [0, -1, 0]
    assert normal in (pos_x, neg_x, pos_y, neg_y)
    if normal == neg_y:
        return cell_defs[0:n_x_cells].view()
    elif normal == pos_y:
        return cell_defs[n_x_cells * (n_y_cells - 1) : n_x_cells * n_y_cells].view()
    elif normal == neg_x:
        return cell_defs[0::n_x_cells].view()
    elif normal == pos_x:
        return cell_defs[n_x_cells - 1 :: n_x_cells].view()


def subdivide_cartilage_depth(
    node_locations: np.ndarray,
    element_definitions: np.ndarray,
    reverse_direction: bool = False,
    n_layers: int = 4,
    use_uconn_method: bool = False,
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
    n_elems = element_definitions.shape[0]
    layer_idx = np.hstack([x * np.ones(n_elems, dtype=int) for x in range(4)])
    layer_idx = np.flipud(layer_idx) if reverse_direction else layer_idx

    if not use_uconn_method:
        # subdivide the single element depth into four elements
        points = np.moveaxis(node_locations[element_definitions], 1, 0)
        points = np.linspace(points[:4], points[4:], n_layers + 1)  # HexElem topology

        # assign IDs to all the points in the cloud, and reshape the cloud for indexing
        point_cloud_dims = points.shape[:-1]
        node_ids = np.arange(np.prod(point_cloud_dims)).reshape(point_cloud_dims)
        new_node_locs = points.reshape(-1, 3)

        # split the cloud of point IDs into element pairs
        new_elems = np.hstack([np.vstack(node_ids[a : a + 2]) for a in np.arange(4)]).T

        # remove coincident nodes, and adjust the element definitions
        new_node_locs, inverse_idx = np.unique(
            new_node_locs.round(2), axis=0, return_inverse=True
        )
        new_elems = inverse_idx[new_elems]
        return new_node_locs, new_elems, layer_idx

    elif use_uconn_method:
        return uconn.subdivide_cartilage_depth(
            node_locations,
            element_definitions,
            reverse_direction=reverse_direction,
            n_layers=n_layers,
        )


def squeeze_hex_mesh(
    nodes: np.ndarray,
    elems: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct degenerate cells following mesh initialization.

    After raytracing a CartilageMesh, some elements at the edge will only contain six of
    the necessary eight nodes. These degenerate elements need to have the missing nodes
    added, with the method of correction depending on the location of the missing nodes
    and the configuration of neighboring elements.

    Args:
        point_locations: an (n, 3) array of point locations.
        cell_def: an (m, 8) array of point IDs with each row defining a single element
            within the cartilage mesh.

    Returns:
        a tuple containing the cell definition and point locations after correcting
        degenerate elements and removing cells containing only undefined points.

    """
    corner_pair_cids = find_touching_corners(nodes, elems)

    nodes = fix_touching_corners(corner_pair_cids, nodes, elems)

    # # Find the number of nodes NaN
    # missing_node_mask = np.isnan(B[0])
    missing_node_mask = degens.get_missing_node_mask(elems, nodes)
    n_missing_nodes = degens.get_missing_node_mask(elems, nodes).sum(axis=1)
    # Squeeze the hexahedrons that have a single node NaN
    elem_7_node_idxs = np.flatnonzero(n_missing_nodes == 1)
    if np.any(elem_7_node_idxs):
        elems = degens.fix_7_node_hex(elems, missing_node_mask)

    # Squeeze the hexahedrons that have two nodes NaN
    try:
        elems, nodes = degens.fix_6_node_hex(elems, nodes)
    except ValueError as e:
        raise degens.DegenerateError("Failed to fix 6-node hexahedrons.") from e

    # Check again for NaN nodes
    H = elems.flatten()
    B = nodes[H, :].T.reshape((3, -1, 8), order="F")
    missing_node_mask = np.isnan(B[0])

    # Remove any remaining degenerate nodes
    is_still_degenerate = missing_node_mask.sum(axis=1) > 0
    elems = np.delete(elems, is_still_degenerate, axis=1)

    # Remove any orphaned nodes
    node_idx, inverse_idx = np.unique(elems, return_inverse=True)
    nodes = nodes[node_idx]
    elems = inverse_idx.reshape(8, -1)

    # Remove duplicate nodes
    nodes, inverse_idx = np.unique(nodes, axis=0, return_inverse=True)
    elems = inverse_idx[elems].T

    return by_value(nodes, elems)


def smooth_hex_mesh(
    elems: np.ndarray,
    nodes: np.ndarray,
    smoothing_operation: str,
    flag: bool = False,
) -> np.ndarray:
    nodes_smooth = nodes.copy()

    if smoothing_operation == "total":
        near_node_ids = np.vstack(
            [
                # counter clockwise first face
                elems[:, 0:2],
                elems[:, 1:3],
                elems[:, 2:4],
                elems[:, (3, 0)],
                # counter clockwise bottom face
                elems[:, 4:6],
                elems[:, 5:7],
                elems[:, 6:8],
                elems[:, (7, 4)],
                # front face
                elems[:, 0:2],
                elems[:, (1, 5)],
                elems[:, (5, 4)],
                elems[:, (4, 0)],
                # back face
                elems[:, 2:4],
                elems[:, (3, 7)],
                elems[:, (7, 6)],
                elems[:, (6, 2)],
                # left face
                elems[:, (0, 3)],
                elems[:, (3, 7)],
                elems[:, (7, 4)],
                elems[:, (4, 0)],
                # right face
                elems[:, (1, 2)],
                elems[:, (2, 6)],
                elems[:, (6, 5)],
                elems[:, (5, 1)],
            ]
        )
    # with py index ranges# shapes match at 7836x2 not being inclusive, some of the
    # upper limit index ranges may match matlab indexes.
    #   i.e. elems[:, 0:2] ---> zero up to 2, or zero through one.
    elif smoothing_operation == "surface":
        # Trace element face counter clockwise along the top and bottom surfaces. Do not
        # adjust the edges along articular cartilage depth.
        near_node_ids = np.vstack(
            [
                # bottom face
                elems[:, 0:2],
                elems[:, 1:3],
                elems[:, 2:4],
                elems[
                    :, (3, 0)
                ],  # last idx, first idx; for a 2d section of the model shown above.
                # top face
                elems[:, 4:6],  # i.e. the 2d square section of the 3d cube.
                elems[:, 5:7],
                elems[:, 6:8],
                elems[:, (7, 4)],  # last idx, first idx
            ]
        )
    elif smoothing_operation == "meniscus":
        near_node_ids = np.vstack(
            [
                elems[:, (0, 4)],  # traveling axially through the meniscus
                elems[:, (1, 5)],
                elems[:, (2, 6)],
                elems[:, (3, 7)],
            ]
        )

    near_node_ids = np.vstack((near_node_ids, np.fliplr(near_node_ids)))
    near_node_ids = np.unique(near_node_ids, axis=0)

    # Calculate edge vectors?
    vdist = nodes_smooth[near_node_ids[:, 1], :] - nodes_smooth[near_node_ids[:, 0], :]

    # Sum the distances?
    sdist = np.cumsum(vdist, axis=0)

    needs_masking = np.diff(near_node_ids[:, 0]) != 0
    msk = np.append(needs_masking, True)[:, np.newaxis]

    tmp_array = np.array([0, 0, 0]).reshape(1, 3)
    sdist_msk = sdist[msk[:, 0]]
    # combine into a 2031x3 array
    tmp_sdist_msk = np.concatenate([tmp_array, sdist_msk])

    svdist = np.diff(tmp_sdist_msk, n=1, axis=0)
    find_msk = np.flatnonzero(msk.ravel())
    d = np.diff(np.hstack((0, find_msk + 1)))

    if flag:
        d -= 1

    if nodes_smooth.shape[0] == svdist.shape[0]:
        repmater = np.tile(d, (3, 1)).T  # transpose
        nodes_smooth = nodes_smooth + svdist / repmater / 2

    return by_value(nodes_smooth)

# @deprecated
def subdivide_hex_mesh(
    nodes: np.ndarray,
    elems: np.ndarray,
    ind6_nodes: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # #########################################################################
    #  refineHexMesh.m python port
    #
    #   refineHexMesh divides a 8-nodes hexahedron into 4 8-node hexahedra
    #   nodes is a Mx3 matrix with the vertices' coordinates
    #   hex is a Nx8 matrix with the indices to the vertices of each hexahedron
    #   ind6Nodes is a Nx1 with 0 when 8-node and 1,2,3 or 4 if 6-node
    # #########################################################################
    siz = elems.shape[0]


    # Node 1
    v1 = nodes[elems[:, 0], :]
    ind1 = 0 * siz + np.arange(siz)
    # Node 2
    v2 = nodes[elems[:, 1], :]
    ind2 = 1 * siz + np.arange(siz)
    # Node 3
    v3 = nodes[elems[:, 2], :]
    ind3 = 2 * siz + np.arange(siz)
    # Node 4
    v4 = nodes[elems[:, 3], :]
    ind4 = 3 * siz + np.arange(siz)
    # Node 5
    v5 = nodes[elems[:, 4], :]
    ind5 = 4 * siz + np.arange(siz)
    # Node 6
    v6 = nodes[elems[:, 5], :]
    ind6 = 5 * siz + np.arange(siz)
    # Node 7
    v7 = nodes[elems[:, 6], :]
    ind7 = 6 * siz + np.arange(siz)
    # Node 8
    v8 = nodes[elems[:, 7], :]
    ind8 = 7 * siz + np.arange(siz)

    # New nodes
    # Node between 1 and 2
    v1_2 = (v1 + v2) / 2
    ind1_2 = 8 * siz + np.arange(siz)
    # Node between 2 and 3
    v2_3 = (v2 + v3) / 2
    ind2_3 = 9 * siz + np.arange(siz)
    # Node between 3 and 4
    v3_4 = (v3 + v4) / 2
    ind3_4 = 10 * siz + np.arange(siz)
    # Node between 4 and 1
    v4_1 = (v4 + v1) / 2
    ind4_1 = 11 * siz + np.arange(siz)
    # Center of the face 1-2-3-4
    c1_4 = (v1 + v2 + v3 + v4) / 4
    indc1_4 = 12 * siz + np.arange(siz)
    # Node between 5 and 6
    v5_6 = (v5 + v6) / 2
    ind5_6 = 13 * siz + np.arange(siz)
    # Node between 6 and 7
    v6_7 = (v6 + v7) / 2
    ind6_7 = 14 * siz + np.arange(siz)
    # Node between 7 and 8
    v7_8 = (v7 + v8) / 2
    ind7_8 = 15 * siz + np.arange(siz)
    # Node between 8 and 5
    v8_5 = (v8 + v5) / 2
    ind8_5 = 16 * siz + np.arange(siz)

    # Center of the face 5-6-7-8
    c5_8 = (v5 + v6 + v7 + v8) / 4
    indc5_8 = 17 * siz + np.arange(siz)

    if ind6_nodes is not None:
        # Find those hexahedra with 6 nodes and divide them into 1 8-node and 2 6-node
        c1_4[ind6_nodes == 2, :] = (v1[ind6_nodes == 2, :] + v3[ind6_nodes == 2, :]) / 2
        c5_8[ind6_nodes == 2, :] = (v5[ind6_nodes == 2, :] + v7[ind6_nodes == 2, :]) / 2
        c1_4[ind6_nodes == 4, :] = (v1[ind6_nodes == 4, :] + v3[ind6_nodes == 4, :]) / 2
        c5_8[ind6_nodes == 4, :] = (v5[ind6_nodes == 4, :] + v7[ind6_nodes == 4, :]) / 2
        c1_4[ind6_nodes == 1, :] = (v2[ind6_nodes == 1, :] + v4[ind6_nodes == 1, :]) / 2
        c5_8[ind6_nodes == 1, :] = (v6[ind6_nodes == 1, :] + v8[ind6_nodes == 1, :]) / 2
        c1_4[ind6_nodes == 3, :] = (v2[ind6_nodes == 3, :] + v4[ind6_nodes == 3, :]) / 2
        c5_8[ind6_nodes == 3, :] = (v6[ind6_nodes == 3, :] + v8[ind6_nodes == 3, :]) / 2

    # Create the new nodes and hexahedra
    hex1 = np.array([indc1_4, ind2_3, ind3, ind3_4, indc5_8, ind6_7, ind7, ind7_8])
    hex2 = np.array([ind4_1, indc1_4, ind3_4, ind4, ind8_5, indc5_8, ind7_8, ind8])
    hex3 = np.array([ind1, ind1_2, indc1_4, ind4_1, ind5, ind5_6, indc5_8, ind8_5])
    hex4 = np.array([ind1_2, ind2, ind2_3, indc1_4, ind5_6, ind6, ind6_7, indc5_8])

    H = np.hstack([hex1, hex2, hex3, hex4])[:]
    H = H.T
    N = np.vstack(
        [
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v1_2,
            v2_3,
            v3_4,
            v4_1,
            c1_4,
            v5_6,
            v6_7,
            v7_8,
            v8_5,
            c5_8,
        ],
    )
    # and eliminates the redundant nodes
    N, ia, ic = np.unique(N, axis=0, return_index=True, return_inverse=True)

    H = ic[H]

    nodes = N.copy()
    elems = H.copy()
    return by_value(nodes, elems)


# @deprecated
def compute_hexa_tibial(
    interior_nodes: np.ndarray,
    exterior_nodes: np.ndarray,
    n_vert_nodes: int,
    n_horz_nodes: int,
    n_radial_nodes: Union[int, float],
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide and mesh five tibial plateau quadrants with brick elements.

    The mesh is initialized as a centrally located rectangular plane of bricks.
    Raytracing is then performed around the perimeter of this central region, with rays
    directed in a radial pattern. The elements coming off of the sides (not top and
    bottom) of the central region are each meshed separately, before being concatenated
    together.

    Args:
        interior_nodes (np.ndarray): an (n, 3) array of interior (first) surface / ray
            intersection locations.
        exterior_nodes (np.ndarray): an (n, 3) array of exterior (second) surface / ray
            intersection locations.
        n_vertical (int): the number of elements in the vertical direction.
        n_horizontal (int): the number of elements in the horizontal direction.
        n_radial (Union[int, float]): the number of elements in the radial direction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Node locations, and element definitions composed
        of nodal IDs.

    """
    # Initialize a bunch of mini-meshes, which we'll later concatenate

    def get_submesh(start, stop, horz_vert_shape):
        hex_plane = EmptyHexahedralPlane(*horz_vert_shape)
        hex_plane.points = np.vstack(
            (interior_nodes[start:stop], exterior_nodes[start:stop])
        ).copy()
        return hex_plane

    center = get_submesh(
        0,
        n_horz_nodes * n_vert_nodes,  # h * v
        (n_horz_nodes - 1, n_vert_nodes - 1),
    )

    up = get_submesh(
        n_vert_nodes * n_horz_nodes,
        (n_vert_nodes + n_radial_nodes) * n_horz_nodes,  # h * (v + r)
        (n_radial_nodes - 1, n_horz_nodes - 1),
    )

    left = get_submesh(
        (n_vert_nodes + n_radial_nodes) * n_horz_nodes,
        (n_vert_nodes + n_radial_nodes) * n_horz_nodes
        + (n_radial_nodes * n_vert_nodes),
        (n_radial_nodes - 1, n_vert_nodes - 1),
    )

    down = get_submesh(
        (n_vert_nodes + n_radial_nodes) * n_horz_nodes
        + (n_radial_nodes * n_vert_nodes),
        (n_vert_nodes + 2 * n_radial_nodes) * n_horz_nodes
        + (n_radial_nodes * n_vert_nodes),
        (n_radial_nodes - 1, n_horz_nodes - 1),
    )

    right = get_submesh(
        (n_vert_nodes + 2 * n_radial_nodes) * n_horz_nodes
        + (n_radial_nodes * n_vert_nodes),
        (n_vert_nodes + 2 * n_radial_nodes) * n_horz_nodes
        + 2 * (n_radial_nodes * n_vert_nodes),
        (n_radial_nodes - 1, n_vert_nodes - 1),
    )

    if kwargs.get("debug", False):
        pv.MultiBlock(
            {
                "center": center.grid,
                "right": right.grid,
                "left": left.grid,
                "top": up.grid,
                "bottom": down.grid,
            }
        ).plot(show_edges=True, multi_colors=True)
        if kwargs.get("breakpoint", False):
            breakpoint()

    container = HexPlaneContainer(
        center=center, right=right, left=left, top=up, bottom=down
    )

    # Handle the sides
    container.mark_weld("right", "+y", "top", "-y")
    container.mark_weld("top", "+y", "left", "-y")
    container.mark_weld("left", "+y", "bottom", "-y")
    container.mark_weld("bottom", "+y", "right", "-y")

    # Handle the center
    container.mark_weld("left", "-x", "center", "+x")
    container.mark_weld("bottom", "-x", "center", "+y")
    container.mark_weld("right", "-x", "center", "-x")  # This is the only problem!
    container.mark_weld("top", "-x", "center", "-y")

    welded_mesh = container.welded_mesh
    elems = welded_mesh.cells_dict[VTK_HEXAHEDRON].T
    nodes = welded_mesh.points
    elems, nodes = by_value(elems, nodes)

    # FIXME: SHOULD CHOOSE ROUNDING DECIMAL BASED ON PERCENTAGE OF ELEMENT SIZE
    # NOTE: Needs to be smaller than the minimal distance in
    # raytracing.find_surface_vertices
    _, idx, inv_idx = np.unique(
        nodes.round(5), axis=0, return_index=True, return_inverse=True
    )
    nodes = nodes[idx]
    elems = inv_idx[elems].T
    elems = elems[:, np.array((0, 3, 2, 1, 4, 7, 6, 5))]  # swap 3 and 1; 5 and 7
    #                          1, 4, 3, 2, 5, 8, 7, 6
    del welded_mesh, center, down, left, right
    return by_value(nodes, elems)


# #TODO: Migrate into cells
def compute_scaled_jacobian(
    nodes: np.ndarray,
    elems: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.moveaxis(nodes[elems[:, :], :], 1, 0)

    terminations = np.array(
        (
            (1, 3, 4),
            (0, 5, 2),
            (1, 6, 3),
            (0, 2, 7),
            (0, 7, 5),
            (1, 4, 6),
            (2, 5, 7),
            (3, 6, 4),
        )
    )
    normal = np.moveaxis(
        vertices[terminations.T, ...] - vertices[np.arange(8), ...], 1, 0
    )
    norms = trimesh.util.unitize(normal.reshape(-1, 3)).reshape(*normal.shape)
    norms[np.isnan(norms)] = 0
    norms = np.moveaxis(norms, 2, 1)

    corner_node_jacobian_determinants = (
        np.linalg.det(norms.reshape(-1, 3, 3)).reshape(8, -1).T
    )
    scaled_jacobian = corner_node_jacobian_determinants.min(axis=1)
    return scaled_jacobian, corner_node_jacobian_determinants


def relax_skewed_elements(
    nodes: np.ndarray,
    elems: np.ndarray,
    acute_angle_limit: Union[float, int] = 45,
    obtuse_angle_limit: Union[float, int] = 135,
) -> np.ndarray:
    """Find nodal displacements to relax overly acute or obtuse element corners.

    Elements are ill-conditioned when their scaled Jacobian's are less than 0.5, which
    results when they appear "pinched" with pairwise corners appearing overly acute, or
    obtuse. Well conditioned elements will contain corners as near to right-angles as
    possible. Relaxing an ill-conditioned element will pull the acute corners towards
    each other, and push the obtuse angles further apart, resulting in corners nearer to
    right angles throughout the element.

    Args:
        nodes (np.ndarray): an (n, 3) array of nodes defining the mesh.
        elems (np.ndarray): an (m, 8) array of element definitions consisting of the
            nodal IDs matching the locations in nodes.
        acute_angle_limit (Union[float, int], optional): corners with angles less than
            the acute angle limit will be displaced towards the element center. Defaults
            to 45.
        obtuse_angle_limit (Union[float, int], optional): corners with angles greater
            than the obtuse angle limit will be displaced away from the element center.
            Defaults to 135.

    Returns:
        np.ndarray: The nodal displacement vectors which result in a more relaxed
        element.

    """
    disp = np.zeros_like(nodes)

    ref_hex_elem = HexElement()
    ref_hex_elem.nodal_connectivity
    for idx in range(8):
        legs = ref_hex_elem.corner_connectivity(idx, list_corner_first=True)
        combos = np.array(list(combinations(legs, 2)))
        for combo in combos:
            leg_nodes = combo[combo != idx]
            hyp = ref_hex_elem.get_hypotenuse_node(combo)
            disp = _adjust_displacement(
                nodes,
                elems,
                disp,
                obtuse_angle_limit,
                acute_angle_limit,
                _get_vectors(nodes, elems, *combo),
                np.hstack((idx, leg_nodes)),
                np.hstack((idx, leg_nodes[::-1])),
                np.hstack((hyp, leg_nodes)),
            )
    return by_value(disp)


def _get_vectors(nodes, elems, *start_end_indices):
    """Help relax_skewed_elements."""
    idxs = np.vstack(start_end_indices)
    vecs = nodes[elems[idxs[:, -1]], :] - nodes[elems[idxs[:, 0]], :]
    unit_vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    return unit_vecs


def _adjust_displacement(
    nodes,
    elems,
    disp,
    obtuse_angle_limit,
    acute_angle_limit,
    vectors,
    disp_idxs,
    head_idxs,
    tail_idxs,
):
    """Help relax_skewed_elements."""
    disp_idxs, head_idxs, tail_idxs = [
        np.asarray(x) for x in (disp_idxs, head_idxs, tail_idxs)
    ]
    angle = np.rad2deg(np.arccos(np.dot(*vectors)))
    scalers = np.asarray((0.1, 0.05, 0.05))
    elems[disp_idxs]
    if angle > obtuse_angle_limit:
        disp[elems[disp_idxs], :] = scalers[:, np.newaxis] * (
            nodes[elems[head_idxs], :] - nodes[elems[tail_idxs], :]
        )
    if angle < acute_angle_limit:
        disp[elems[disp_idxs], :] = -scalers[:, np.newaxis] * (
            nodes[elems[head_idxs], :] - nodes[elems[tail_idxs], :]
        )
    return disp


# @deprecated
def optimize_mesh(
    nodes: np.ndarray,
    elems: np.ndarray,
    target_jacobian: bool = None,
    allow_jacobian_break: bool = True,
) -> np.ndarray:
    if target_jacobian is None:
        target_jacobian = 0

    # Compute the neighbors of each vertex
    # Hexa is made of 2D quads, each traced counterclockwise to make positive faces
    neighbors = np.vstack(
        (
            elems[:, 0:4].reshape(-1, 2),  # top
            elems[:, np.array((3, 0))],
            elems[:, 4:8].reshape(-1, 2),  # bottom
            elems[:, np.array((7, 4))],
            elems[:, np.array((1, 5, 5, 6, 6, 2, 2, 1))].reshape(-1, 2),  # right
            elems[:, np.array((0, 4, 4, 7, 7, 3, 3, 0))].reshape(-1, 2),  # left
            elems[:, np.array((0, 4, 4, 5, 5, 1, 1, 0))].reshape(-1, 2),  # front
            elems[:, np.array((3, 7, 7, 6, 6, 2, 2, 3))].reshape(-1, 2),  # back
        )
    )
    neighbors = np.unique(np.vstack((neighbors, np.fliplr(neighbors))), axis=0)

    needs_optimization = True
    while needs_optimization:
        scaled, jacobian = compute_scaled_jacobian(nodes, elems)
        # compute scaled Jacobian
        is_suboptimal = jacobian < target_jacobian
        indexes = np.unique(elems[is_suboptimal])
        if indexes.size == 0:
            needs_optimization = False

        else:
            # For each index substitute the node by the average of it's neighbors
            disp = np.zeros_like(nodes)
            if target_jacobian == 0:
                for idx in indexes:
                    is_match = neighbors[:, 0] == idx
                    i_neighbor = neighbors[is_match, 1]
                    disp[idx, :] = nodes[i_neighbor].mean(axis=0) - nodes[idx, :]

            else:
                suboptimal_idx = np.flatnonzero(scaled < target_jacobian)
                for idx in suboptimal_idx:
                    suboptimal_elems = elems[idx]
                    disp = disp + relax_skewed_elements(nodes, suboptimal_elems)

                if np.linalg.norm(disp.sum(axis=0)) == 0:
                    needs_optimization = False

            shifted_nodes = nodes + disp
            scaled = (
                # HexMesh(elems, shifted_nodes)
                UGrid.from_cell_def(points=shifted_nodes, cell_def=elems)
                .mesh.cell_quality("scaled_jacobian")
                .cell_data["scaled_jacobian"]
            )
            if ~np.any(scaled < 0) or not allow_jacobian_break:
                nodes = shifted_nodes
            else:
                needs_optimization = False
    return by_value(nodes)


def expand_border_elements(
    reconstruction: trimesh.Trimesh,
    nodes: np.ndarray,
    elems: np.ndarray,
    max_node_shift: Union[int, float] = None,
    only_edges: bool = None,
    include_centers: bool = True,
) -> None:
    surf = UGrid.from_cell_def(points=nodes, cell_def=elems).extract_surface(
        algorithm=None
    )
    avg_surf_cell_area = surf.area / surf.n_cells
    avg_side_len = np.sqrt(avg_surf_cell_area)
    del surf
    max_node_shift = avg_side_len * 2 if max_node_shift is None else max_node_shift

    # Get border elements
    mesh = HexMesh(elems, nodes)
    top_pids, bot_pids = mesh.top_bottom_point_ids
    mesh.mesh.point_data["is_top"] = np.full(mesh.mesh.n_points, False, dtype=np.uint8)
    mesh.mesh.point_data["is_top"][top_pids] = True
    perim_edges, perim_ribbon = mesh.perimeter_edges  # noqa
    perim_edges["mesh_pids"] = perim_edges["vtkOriginalPointIds"]
    perim_edges["mesh_cids"] = perim_edges["vtkOriginalCellIds"]

    # Add face rays
    perim_tri_mesh = perim_ribbon.extract_surface(algorithm=None)
    perim_tri_mesh.compute_normals(inplace=True)
    point_normals = perim_tri_mesh.point_data["Normals"]
    face_normals = perim_tri_mesh.cell_data["Normals"]
    perim_ribbon.cell_data["Normals"] = np.full_like(face_normals, np.nan)
    perim_ribbon.point_data["Normals"] = np.full_like(point_normals, np.nan)
    perim_ribbon.cell_data["Normals"] = face_normals[
        perim_tri_mesh["vtkOriginalCellIds"]
    ]
    perim_ribbon.point_data["Normals"] = point_normals[
        perim_tri_mesh["vtkOriginalPointIds"]
    ]

    # Add center origins and directions
    perim_ribbon["mesh_pids"] = perim_ribbon["vtkOriginalPointIds"]
    perim_ribbon["mesh_cids"] = perim_ribbon["vtkOriginalCellIds"]
    depth_lines = perim_ribbon.extract_feature_edges(
        feature_edges=False,
        manifold_edges=True,
        non_manifold_edges=False,
        boundary_edges=False,
    )
    centers = depth_lines.cell_centers()
    centers["norms"] = depth_lines.point_data["Normals"][::2]
    perim_ribbon.set_active_vectors("Normals")

    depth_lines.field_data["is_first_edge"] = depth_lines["is_top"].astype(bool)
    depth_lines.field_data["is_second_edge"] = ~depth_lines["is_top"].astype(bool)

    line_pair_ids = depth_lines.lines.reshape(-1, 3)[:, 1:].copy()
    is_first_edge = np.isin(
        line_pair_ids, np.flatnonzero(depth_lines.field_data["is_first_edge"])
    )
    needs_flip = ~is_first_edge[:, 0]
    line_pair_ids[needs_flip] = np.fliplr(line_pair_ids[needs_flip])
    final_lines = depth_lines.lines.reshape(-1, 3)
    final_lines[:, 1:] = line_pair_ids
    depth_lines.lines = final_lines.flatten()

    # Arrange origin and direction points for raytracing
    origins = np.vstack(
        (
            depth_lines.points[line_pair_ids[:, 0]],
            depth_lines.points[line_pair_ids[:, 1]],
        )
    )
    directions = np.vstack(
        (
            depth_lines.point_data["Normals"][depth_lines.lines[1::3].copy()],
            depth_lines.point_data["Normals"][depth_lines.lines[2::3].copy()],
        )
    )
    if include_centers:
        origins = np.vstack((origins, centers.points))
        directions = np.vstack((directions, centers.point_data["norms"]))

    # perform raytracing
    tri_idx, ray_idx, locations = reconstruction.ray.intersects_id(
        ray_origins=origins,
        ray_directions=directions,
        return_locations=True,
        multiple_hits=False,
    )

    intersect_face_normals = reconstruction.face_normals[tri_idx]
    angle_between = trimesh.geometry.vector_angle(
        np.stack((directions[ray_idx], intersect_face_normals), axis=1)
    )
    is_backface = np.cos(angle_between) > 0.0
    ray_idx = ray_idx[is_backface]
    locations = locations[is_backface]

    # add face displacements
    displacement_vectors = np.zeros_like(origins)
    displacement_vectors[ray_idx] = locations - origins[ray_idx]
    distances = np.linalg.norm(displacement_vectors, axis=1)
    needs_update = distances < max_node_shift
    distances[~needs_update] = 0
    displacement_vectors[~needs_update] = np.zeros(3)
    n_lines = centers.points.shape[0]

    max_distances = distances.reshape(n_lines, -1)
    max_distances = np.tile(
        distances.reshape(n_lines, -1, order="F").max(axis=1), max_distances.shape[1]
    )
    displacement_vectors = max_distances[..., np.newaxis] * directions

    depth_lines.point_data["displacements"] = depth_lines.point_data["Normals"].copy()
    depth_lines.point_data["displacements"][line_pair_ids[:, 0]] = displacement_vectors[
        :n_lines
    ]
    depth_lines.point_data["displacements"][line_pair_ids[:, 1]] = displacement_vectors[
        n_lines : 2 * n_lines
    ]
    depth_lines.set_active_vectors("displacements")
    mesh.mesh.point_data["displacements"] = np.zeros(
        (mesh.mesh.n_points, 3), dtype=float
    )
    mesh.mesh.point_data["displacements"][depth_lines["mesh_pids"]] = (
        depth_lines.point_data["displacements"]
    )

    disp_vectors = smooth_hex_mesh(elems, mesh.mesh["displacements"], "surface", 0)


    del perim_ribbon, perim_edges, face_normals, mesh, depth_lines
    return by_value(nodes + disp_vectors)


def iterative_optimization(
    reconstruction: trimesh.Trimesh,
    hex_mesh: Union[HexMesh, pv.UnstructuredGrid],
    smoothing_operation: str = "surface",
    max_iters: int = 20,
    max_node_shift: Union[float, int] = 2,
    upper_jacobian_limit: Union[float, int] = 0.5,
    include_expansion: bool = True,
    include_smoothing: bool = True,
) -> Union[HexMesh, pv.UnstructuredGrid]:
    return_type = type(hex_mesh)
    hex_mesh = ensure_pyvista(hex_mesh)
    nodes = hex_mesh.points.copy()
    elems = hex_mesh.cells_dict[VTK_HEXAHEDRON].copy()
    nodes, elems = by_value(nodes, elems)
    s = hex_mesh.cell_quality("scaled_jacobian").cell_data["scaled_jacobian"]
    is_bad_jacobian = s < upper_jacobian_limit
    loop_counter = 0
    while loop_counter < max_iters and is_bad_jacobian.any():
        if include_smoothing:
            nodes = smooth_hex_mesh(
                elems, nodes, smoothing_operation=smoothing_operation
            )
        if include_expansion:
            nodes = expand_border_elements(
                reconstruction, nodes, elems, max_node_shift=max_node_shift
            )
        nodes = optimize_mesh(nodes, elems, target_jacobian=upper_jacobian_limit)
        hex_mesh.points = nodes.copy()
        s = hex_mesh.cell_quality("scaled_jacobian").cell_data["scaled_jacobian"]
        is_bad_jacobian = s < upper_jacobian_limit
        loop_counter += 1

        try:
            if loop_counter >= max_iters:
                raise OptimizationError(
                    f"Mesh optimization did not finish after {max_iters} iterations."
                )
        except OptimizationError as e:
            del hex_mesh, nodes, elems
            raise e

    # Process the return
    nodes, elems = by_value(nodes, elems)
    if return_type is HexMesh:
        hex_mesh = HexMesh(elems, nodes)
    elif return_type is pv.UnstructuredGrid:
        hex_mesh.points = nodes.copy()

    del s, is_bad_jacobian
    return hex_mesh


def cartilage_layer_depth(hm: HexMesh):
    grid = hm.mesh.copy()
    top_node_ids = grid.cells_dict[VTK_HEXAHEDRON][:, :4]
    bot_node_ids = grid.cells_dict[VTK_HEXAHEDRON][:, 4:]
    vectors = grid.points[top_node_ids] - grid.points[bot_node_ids]
    distances = np.linalg.norm(vectors, axis=2)
    del grid
    return distances


def blend_cartilage_mesh(
    hm: HexMesh,
    layer: np.array,
    column_depths: np.ndarray,
    grade: Union[float, int] = 1,
    reverse_smoothing_direction: bool = False,
    shift_to_centerline: bool = False,
    edge_smoothing_window: int = None,
    n_edge_smoothing_passes: int = None,
    is_tibial: bool = False,
):
    min_angle = 25
    max_angle = 30
    hm.mesh.point_data["displacements"] = np.zeros_like(hm.mesh.points)
    hm.mesh.point_data["edge_normals"] = np.zeros_like(hm.mesh.points)
    hm.mesh.cell_data["column_depth"] = np.zeros(hm.mesh.n_cells, dtype=float)
    hm.mesh.point_data["is_internal_point"] = np.full(hm.mesh.n_points, True)
    layer_idxs = np.unique(layer)
    n_layer = layer_idxs.size
    # Move layer by layer, moving the nodes to create the prescribed slope.
    for layer_idx in layer_idxs:
        grid, bound_edges = _compute_edge_normals(
            hm,
            layer,
            column_depths,
            layer_idx,
            reverse_smoothing_direction=reverse_smoothing_direction,
        )

        # Figure out which one is more distal
        top_node_ids, bot_node_ids = _get_top_bottom_pids(
            reverse_smoothing_direction, grid
        )
        grid.point_data["is_top_point"] = np.full(grid.n_points, False)
        grid["is_top_point"][top_node_ids] = True

        hm.mesh["edge_normals"][grid.point_data["mesh_pids"]] = grid.point_data[
            "edge_normals"
        ].copy()
        hm.mesh["is_internal_point"][grid.point_data["mesh_pids"]] = grid.point_data[
            "is_internal_point"
        ].copy()
        grid.point_data["edge_normals"][bot_node_ids] *= -1

        grid["grid_pids"] = np.arange(grid.n_points)

        # Calculate the scalars
        grid_pdata = grid.cell_data_to_point_data()
        scalars = grade * grid_pdata.point_data["depths"]
        quantile = np.quantile(scalars, 0.85)
        scalars[scalars > quantile] = quantile

        # Check if the region is too small (elems would overlap if nodes adjusted)
        conn = bound_edges.connectivity()
        for rid in np.unique(conn["RegionId"]):
            # #TODO: _closed_loop_travel_dist_ratio?
            line = conn.extract_cells(conn["RegionId"] == rid)
            llen = line.length
            median_depth = np.median(line["depths"])
            depth_scaled_len = llen * median_depth
            travel_dist = grade * median_depth * n_layer
            travel_dsl_ratio = travel_dist / depth_scaled_len
            # Calculate initial area
            init_area = line.length**2 / np.pi / 4

            # Calculate deformed area
            new_line = line.copy()
            new_line["scalars"] = grade * new_line.cell_data_to_point_data()["depths"]
            new_line.points += (
                new_line["scalars"][..., np.newaxis] * new_line["edge_normals"]
            )
            new_area = new_line.length**2 / np.pi / 4

            # Check if the deformed area is < 0.5 * calculated initial area
            if travel_dsl_ratio > 0.5:  # scale the scalars by (1 - travel_dist_ratio)
                scalars[line["grid_pids"]] *= 1 - travel_dsl_ratio
            elif new_area / init_area < 0.2:
                scalars[line["grid_pids"]] *= 0.2

        # for keeping the edge centerline
        if shift_to_centerline:
            if layer_idx == 0:
                grid.point_data["edge_normals"][bot_node_ids] = 0
            elif layer_idx == len(layer_idxs) - 1:
                grid.point_data["edge_normals"][top_node_ids] = 0
            else:
                scalars /= 2
        else:
            scalars *= (
                n_layer - layer_idx
                if not reverse_smoothing_direction
                else layer_idx + 1
            )
            grid.point_data["edge_normals"][bot_node_ids] = 0


        # #region TODO: Check that the displacements don't exceed the ideal_feature_angle
        displace_vectors = scalars[..., np.newaxis] * grid.point_data["edge_normals"]
        grid["displace_vectors"] = displace_vectors

        deformed_grid = grid.copy()
        deformed_grid.points += displace_vectors
        dg = deformed_grid.copy()

        # Check if any of the feature angles have gotten too small
        # Find the feature edges above the minimum angle
        all_edges = HexElement().mesh
        while all_edges.n_cells > 0:
            surf = deformed_grid.copy().extract_surface(algorithm=None)

            # Select the top and bottom edge of the surface
            edge_cells = surf.copy().extract_points(
                ~surf["is_internal_point"].astype(bool)
            )
            top_edge_cells = edge_cells.copy().extract_points(  # noqa
                ~edge_cells["is_internal_point"].astype(bool)
                & edge_cells["is_top_point"].astype(bool)
            )
            bot_edge_cells = edge_cells.copy().extract_points(
                ~edge_cells["is_internal_point"].astype(bool)
                & ~edge_cells["is_top_point"].astype(bool)
            )

            # Check if any of the feature angles have gotten too small
            # Find the feature edges above the minimum angle
            min_feature_edges = bot_edge_cells.extract_feature_edges(
                feature_angle=min_angle,
                feature_edges=True,
                boundary_edges=False,
                non_manifold_edges=False,
                manifold_edges=False,
            )
            lines = min_feature_edges.lines.reshape(-1, 3)[:, 1:]
            point_needs_removing = np.flatnonzero(
                min_feature_edges["is_internal_point"]
            )
            min_feature_edges.remove_cells(
                np.isin(lines, point_needs_removing).any(axis=1), inplace=True
            )
            lines = min_feature_edges.lines
            good_points, inv = np.unique(
                lines.reshape(-1, 3)[:, 1:], return_inverse=True
            )
            min_feature_edges.points = min_feature_edges.points[good_points].copy()
            min_feature_edges.lines = np.insert(inv, np.arange(0, inv.size, 2), 2)
            for k, v in min_feature_edges.point_data.items():
                min_feature_edges.point_data[k] = v[good_points]
            # Remove any _interior_ points lying on the _bottom_
            lines = min_feature_edges.lines.reshape(-1, 3)[:, 1:]
            point_needs_removing = np.flatnonzero(min_feature_edges["is_top_point"])
            min_feature_edges.remove_cells(
                np.isin(lines, point_needs_removing).any(axis=1), inplace=True
            )
            lines = min_feature_edges.lines
            good_points, inv = np.unique(
                lines.reshape(-1, 3)[:, 1:], return_inverse=True
            )
            min_feature_edges.points = min_feature_edges.points[good_points].copy()
            min_feature_edges.lines = np.insert(inv, np.arange(0, inv.size, 2), 2)
            for k, v in min_feature_edges.point_data.items():
                min_feature_edges.point_data[k] = v[good_points]

            # Invert the selection using extract_all_edges() and boolean indexing
            all_edges = bot_edge_cells.extract_all_edges()
            point_needs_removing = np.flatnonzero(
                np.isin(all_edges["grid_pids"], min_feature_edges["grid_pids"])
            )
            lines = all_edges.lines.reshape(-1, 3)[:, 1:]
            all_edges.remove_cells(
                np.isin(lines, point_needs_removing).all(axis=1), inplace=True
            )
            lines = all_edges.lines
            good_points, inv = np.unique(
                lines.reshape(-1, 3)[:, 1:], return_inverse=True
            )
            all_edges.points = all_edges.points[good_points].copy()
            all_edges.lines = np.insert(inv, np.arange(0, inv.size, 2), 2)
            for k, v in all_edges.point_data.items():
                all_edges.point_data[k] = v[good_points]
            if all_edges.n_cells < 1:
                break

            # Remove any _interior_ points lying on the _bottom_
            lines = all_edges.lines.reshape(-1, 3)[:, 1:]
            point_needs_removing = np.flatnonzero(all_edges["is_internal_point"])
            all_edges.remove_cells(
                np.isin(lines, point_needs_removing).any(axis=1), inplace=True
            )
            lines = all_edges.lines
            good_points, inv = np.unique(
                lines.reshape(-1, 3)[:, 1:], return_inverse=True
            )
            all_edges.points = all_edges.points[good_points].copy()
            all_edges.lines = np.insert(inv, np.arange(0, inv.size, 2), 2)
            for k, v in all_edges.point_data.items():
                all_edges.point_data[k] = v[good_points]
            if all_edges.n_cells < 1:
                break
            p = pv.Plotter()
            p.add_mesh(
                bot_edge_cells,
                color="tan",
                opacity=0.5,
                show_edges=True,
                edge_color="tan",
            )
            p.add_mesh(all_edges, color="r", line_width=2)
            if min_feature_edges.n_cells > 0:
                p.add_mesh(min_feature_edges, color="y", line_width=2)
            del p

            # Make sure to remove any depth-oriented edges
            # if edge contains a point matching a top point id, remove it
            lines = all_edges.lines.reshape(-1, 3)[:, 1:]
            point_needs_removing = np.flatnonzero(all_edges["is_top_point"])
            all_edges.remove_cells(
                np.isin(lines, point_needs_removing).all(axis=1), inplace=True
            )
            lines = all_edges.lines
            good_points, inv = np.unique(
                lines.reshape(-1, 3)[:, 1:], return_inverse=True
            )
            all_edges.points = all_edges.points[good_points].copy()
            all_edges.lines = np.insert(inv, np.arange(0, inv.size, 2), 2)
            for k, v in all_edges.point_data.items():
                all_edges.point_data[k] = v[good_points]
            if all_edges.n_cells < 1:
                break
            lines = all_edges.lines.reshape(-1, 3)[:, 1:]
            point_needs_removing = np.flatnonzero(all_edges["is_top_point"])
            all_edges.remove_cells(
                np.isin(lines, point_needs_removing).any(axis=1), inplace=True
            )
            lines = all_edges.lines
            good_points, inv = np.unique(
                lines.reshape(-1, 3)[:, 1:], return_inverse=True
            )
            all_edges.points = all_edges.points[good_points].copy()
            all_edges.lines = np.insert(inv, np.arange(0, inv.size, 2), 2)
            for k, v in all_edges.point_data.items():
                all_edges.point_data[k] = v[good_points]
            if all_edges.n_cells < 1:
                break
            p = pv.Plotter()
            p.add_mesh(
                bot_edge_cells,
                color="tan",
                opacity=0.5,
                show_edges=True,
                edge_color="tan",
            )
            p.add_mesh(all_edges, color="r", line_width=2)
            if min_feature_edges.n_cells > 0:
                p.add_mesh(min_feature_edges, color="y", line_width=2)
            del p

            # Select the top nodes which share the same cell IDs of the offending edges
            lines = all_edges.lines.reshape(-1, 3)[:, 1:]
            select_ids = np.flatnonzero(
                np.isin(
                    bot_edge_cells["grid_pids"],
                    all_edges["grid_pids"][np.unique(lines)],
                )
            )

            selected = bot_edge_cells.copy().extract_points(select_ids)
            selected = selected.copy().extract_points(
                selected["is_top_point"].astype(bool), include_cells=False
            )

            # iteratively move these points back along the displacement vector until they
            # are within specified angle (need to be able to run that selection again)
            moved_back = selected.copy()
            moved_back.points -= 0.05 * selected["displace_vectors"]
            p = pv.Plotter()
            p.add_axes()
            p.add_mesh(dg, color="tan", opacity=0.4)
            deformed_grid.points[moved_back["grid_pids"]] = moved_back.points
            p.add_mesh(deformed_grid, color="r", opacity=0.4, show_edges=True)
            p.add_mesh(all_edges, color="y", line_width=3)
            del p

        displace_vectors = scalars[..., np.newaxis] * grid.point_data["edge_normals"]
    hm.mesh.points = deformed_grid.points[deformed_grid.point_data["mesh_pids"]].copy()

    # Smooth edges
    delnames = (  # temporary fix for memory leak
        "all_edges",
        "bot_edge_cells",
        "bound_edges",
        "conn",
        "deformed_grid",
        "dg",
        "displace_vectors",
        "edge_cells",
        "grid_pdata",
        "line",
        "min_feature_edges",
        "moved_back",
        "new_line",
        "selected",
        "surf",
        "top_edge_cells",
        "v",
    )
    for k, v in locals().items():
        if k in delnames:
            del v
    hm = _smooth_closed_loop_edges(
        hm,
        layer,
        column_depths,
        layer_idxs,
        n_success_cutoff=3,
        edge_smoothing_window=edge_smoothing_window,
        n_edge_smoothing_passes=n_edge_smoothing_passes,
        reverse_direction=reverse_smoothing_direction,
    )
    hm = _smooth_blended_edges(
        hm,
        layer,
        column_depths,
        reverse_direction=reverse_smoothing_direction,
        is_tibial=is_tibial,
        max_feature_angle=max_angle,
    )
    p = pv.Plotter()
    p.add_axes()
    p.add_mesh(grid, opacity=1, color="teal", show_edges=True)
    p.add_mesh(hm.mesh, color="tan", opacity=0.5)
    del p, grid

    return hm


def _smooth_blended_edges(
    hm,
    cell_layers,
    column_depths,
    reverse_direction,
    is_tibial: bool = False,
    max_feature_angle: Union[float, int] = 30,
):
    hm.mesh.point_data["new_locs"] = by_value(hm.mesh.points)
    hm.mesh.point_data["displacements"] = np.zeros_like(hm.mesh.points)
    layer_idxs = np.unique(cell_layers)
    for layer_idx in layer_idxs:
        layer_grid, bound_edges = _compute_edge_normals(
            hm,
            cell_layers,
            column_depths,
            layer_idx,
            reverse_smoothing_direction=reverse_direction,
        )

        top_node_ids, bot_node_ids = _get_top_bottom_pids(reverse_direction, layer_grid)
        if is_tibial:
            top_node_ids, bot_node_ids = bot_node_ids, top_node_ids

        layer_grid.point_data["is_top_point"] = np.full(layer_grid.n_points, False)
        layer_grid["is_top_point"][top_node_ids] = True
        layer_surf = layer_grid.extract_surface(algorithm=None)
        layer_surf["grid_pids"] = layer_surf["vtkOriginalPointIds"]
        top_surf = layer_surf.extract_points(
            np.isin(layer_surf["grid_pids"], top_node_ids), adjacent_cells=False
        )
        is_internal_point = top_surf["is_internal_point"].astype(bool)
        edge_cells = top_surf.extract_points(~is_internal_point)
        ################################################################################
        # Select the two edgemost quads
        # NOTE: Possible recursion here
        selection_reduction = (top_surf.n_cells - edge_cells.n_cells) / top_surf.n_cells
        iters = selection_reduction * 550 // (layer_idx + 1)

        def smooth_front(iters, mesh, surf, edge_cells):
            n_bad_original = np.count_nonzero(
                find_bad_jacobians(mesh, min_jacobian=0, divide_first=True)
            )
            successful = False
            iters_choice = np.linspace(iters, 0, 10, dtype=int)
            for iters in iters_choice:
                edge_cells = surf.extract_points(
                    edge_cells["vtkOriginalPointIds"]
                ).copy()
                smoothed = pv.PolyData(edge_cells.points, edge_cells.cells).smooth(
                    # ).smooth_taubin(
                    n_iter=iters,
                    boundary_smoothing=False,  # , normalize_coordinates=True
                )
                p = pv.Plotter()
                p.add_mesh(surf, color="tan", show_edges=False, opacity=0.8)
                p.add_mesh(smoothed, color="r", show_edges=True, opacity=0.8)
                del p
                allclose = np.allclose(
                    surf.points[edge_cells["vtkOriginalPointIds"]],
                    smoothed.points,
                    atol=1e-2,
                )
                surf.points[edge_cells["vtkOriginalPointIds"]] = smoothed.points.copy()
                mesh.mesh.points[surf["mesh_pids"]] = surf.points
                n_bad_now = np.count_nonzero(
                    find_bad_jacobians(mesh, min_jacobian=0, divide_first=True)
                )
                if (n_bad_now > n_bad_original) or allclose:
                    smoothed.points = edge_cells.points.copy()
                    surf.points[edge_cells["vtkOriginalPointIds"]] = (
                        smoothed.points.copy()
                    )
                    mesh.mesh.points[surf["mesh_pids"]] = surf.points
                else:
                    successful = True
                    break
            mesh.mesh.points[surf["mesh_pids"][edge_cells["vtkOriginalPointIds"]]] = (
                edge_cells.points
            )
            mesh.mesh["new_locs"][surf["mesh_pids"]] = surf.points
            edge_cells.points = smoothed.points.copy()
            return iters, mesh, surf, edge_cells, successful

        for _ in range(4):
            iters, hm, t_s, edge_cells, successful = smooth_front(
                iters, hm, top_surf.copy(), edge_cells
            )
            if not successful:
                break
            else:
                top_surf = t_s
            iters //= 2
            edge_cells = top_surf.extract_points(edge_cells["vtkOriginalPointIds"])

        hm.mesh["new_locs"][top_surf["mesh_pids"]] = top_surf.points
        m = hm.mesh.copy()
        m.points = hm.mesh["new_locs"].copy()
        p = pv.Plotter()
        if layer_idxs.size > 1:
            p.add_mesh(m, color="b", opacity=0.4)
        p.add_mesh(layer_surf, color="tan", show_edges=True)
        p.add_mesh(top_surf, style="wireframe", color="r")
        del p
        if layer_idx == layer_idxs.max():
            bot_surf = layer_surf.copy().extract_points(
                np.isin(layer_surf["grid_pids"], bot_node_ids), adjacent_cells=True
            )
            bs_to_ls_pids = bot_surf["vtkOriginalPointIds"]
            volume_orig = layer_surf.volume
            ################################################################################
            # Select the two edgemost quads
            is_external_point = ~bot_surf["is_internal_point"].astype(bool)
            edge_cells = bot_surf.extract_points(is_external_point)
            feature_angle = max_feature_angle
            feature_edges = edge_cells.extract_feature_edges(
                feature_angle=feature_angle,
                feature_edges=True,
                boundary_edges=False,
                non_manifold_edges=False,
                manifold_edges=False,
            )
            lines = feature_edges.lines.reshape(-1, 3)[:, 1:]
            point_needs_removing = np.flatnonzero(feature_edges["is_top_point"])
            feature_edges.remove_cells(
                np.isin(lines, point_needs_removing).any(axis=1), inplace=True
            )
            loop_counter = 0
            while feature_edges.n_cells > 0:
                feature_edge_pids = edge_cells["vtkOriginalPointIds"][
                    feature_edges["vtkOriginalPointIds"]
                ]
                edge_cells = bot_surf.extract_points(feature_edge_pids)
                p = pv.Plotter()
                p.add_mesh(edge_cells, color="tan", show_edges=False)
                p.add_mesh(feature_edges, color="r")
                p.add_axes()
                del p
                iters = 20
                for _ in range(5):
                    iters, hm, bot_surf, edge_cells, successful = smooth_front(
                        iters, hm, bot_surf, edge_cells
                    )
                    layer_surf.points[bs_to_ls_pids] = bot_surf.points
                    if not successful or (layer_surf.volume / volume_orig < 0.95):
                        break
                    iters = int(iters * 0.75)
                    layer_surf.volume
                    # iters //= 2
                    edge_cells = bot_surf.copy().extract_points(
                        edge_cells["vtkOriginalPointIds"]
                    )
                if not successful or (layer_surf.volume / volume_orig < 0.97):
                    break

                feature_edges = edge_cells.extract_feature_edges(
                    feature_angle=feature_angle,
                    feature_edges=True,
                    boundary_edges=False,
                    non_manifold_edges=False,
                    manifold_edges=False,
                )
                lines = feature_edges.lines.reshape(-1, 3)[:, 1:]
                point_needs_removing = np.flatnonzero(feature_edges["is_top_point"])
                feature_edges.remove_cells(
                    np.isin(lines, point_needs_removing).any(axis=1), inplace=True
                )
                loop_counter += 1
            hm.mesh["new_locs"][bot_surf["mesh_pids"]] = bot_surf.points
            ################################################################################
    # hm.mesh.points = hm.mesh["new_locs"].copy()  # FIXME: Remove internal data arrays.
    hm.mesh.points = by_value(
        hm.mesh["new_locs"]
    )  # FIXME: Remove internal data arrays.
    del (
        bot_surf,
        bound_edges,
        edge_cells,
        feature_edges,
        layer_grid,
        layer_surf,
        m,
        t_s,
        top_surf,
    )
    return hm


def _get_top_bottom_pids(reverse_smoothing_direction, grid):
    if not reverse_smoothing_direction:
        top_node_ids = np.unique(grid.cells_dict[VTK_HEXAHEDRON][:, :4])
        bot_node_ids = np.unique(grid.cells_dict[VTK_HEXAHEDRON][:, 4:])
    else:
        bot_node_ids = np.unique(grid.cells_dict[VTK_HEXAHEDRON][:, :4])
        top_node_ids = np.unique(grid.cells_dict[VTK_HEXAHEDRON][:, 4:])
    return top_node_ids, bot_node_ids


def _smooth_closed_loop_edges(
    hm,
    layer,
    column_depths,
    layer_idxs,
    reverse_direction: bool = False,
    n_success_cutoff: int = 3,
    edge_smoothing_window: int = None,
    n_edge_smoothing_passes: int = None,
):
    if n_edge_smoothing_passes == 0:
        n_bad = np.count_nonzero(
            find_bad_jacobians(hm, min_jacobian=0, divide_first=True)
        )
        if n_bad < 0:
            return hm
        else:
            del hm
            raise BlendingError("Failure to smooth closed loops of blends.")

    if n_edge_smoothing_passes is None:
        n_edge_smoothing_passes = 1
    if edge_smoothing_window is None:
        edge_smoothing_window = 3

    # create a copy of the mesh
    new_hm = hm.copy()
    new_hm.mesh.point_data["new_locs"] = new_hm.mesh.points.copy()
    for layer_idx in layer_idxs:
        grid, bound_edges = _compute_edge_normals(
            new_hm.copy(),
            layer,
            column_depths,
            layer_idx,
            reverse_smoothing_direction=reverse_direction,
        )

        top_node_ids, bot_node_ids = _get_top_bottom_pids(reverse_direction, grid)

        grid.point_data["is_top_point"] = np.full(grid.n_points, False)
        grid["is_top_point"][top_node_ids] = True
        grid_orig_points = grid.copy().points.copy()

        # loop through closed loop segments
        conn = bound_edges.connectivity(largest=False)
        rids = np.unique(conn["RegionId"])
        loop_counter = 0
        n_passed = 0
        start_bad_cells = np.count_nonzero(
            find_bad_jacobians(new_hm, min_jacobian=0, divide_first=True)
        )
        made_worse_count = 0
        while n_passed < n_success_cutoff:
            new_hm = hm.copy()
            grid.points = by_value(grid_orig_points)
            for rid in rids:
                window = edge_smoothing_window + 2 * loop_counter
                polyorder = 2 if loop_counter == 0 else 3
                closed_loop = conn.extract_cells(
                    conn["RegionId"] == rid
                ).extract_surface(algorithm=None)
                if closed_loop.n_points < 7:  # loop is too small
                    continue
                elif closed_loop.n_points // 4 < window:  # window is too large
                    window = closed_loop.n_points // 4 * 2 - 1
                    polyorder = 2 if window == 3 else polyorder

                sorted_pids = closed_loop.strip(join=True).lines[1:-1]
                pts = closed_loop.points[sorted_pids]
                smoothed_points = pts.copy()

                for iter_ in range(n_edge_smoothing_passes):
                    smoothed_pts = savgol_filter(
                        smoothed_points,
                        window,
                        polyorder,
                        axis=0,
                        mode="wrap",
                    )

                mpids = closed_loop["grid_pids"][sorted_pids]
                if grid["is_top_point"][mpids].all():
                    grid.points[mpids] = by_value(smoothed_pts)
                    new_hm.mesh.points[grid["mesh_pids"]] = by_value(grid.points)
            has_bad_cell = find_bad_jacobians(new_hm, min_jacobian=0, divide_first=True)
            made_worse_count += (
                1 if np.count_nonzero(has_bad_cell) - start_bad_cells > 0 else 0
            )
            if made_worse_count > 3 or (edge_smoothing_window + 2 * loop_counter) > (
                closed_loop.n_points // 4 * 2 - 1 + 3
            ):
                try:
                    if n_passed > 0 and n_passed < n_success_cutoff:
                        n_success_cutoff = n_passed
                        # new_hm.mesh.points = hm.mesh.points.copy()
                        new_hm.mesh.points = by_value(hm.mesh.points)
                        loop_counter = 0  # noqa
                        n_passed = 0
                        made_worse_count = 0
                        continue
                    elif start_bad_cells == 0:
                        return hm
                    else:
                        raise BlendingError("Failure to smooth closed loops of blends.")
                except BlendingError:
                    del (
                        bound_edges,
                        closed_loop,
                        conn,
                        grid,
                        grid_orig_points,
                        new_hm,
                        pts,
                        smoothed_points,
                    )
                    raise
            n_passed += 1 if not has_bad_cell.any() else 0
            loop_counter += 1
            # snapshot = tracemalloc.take_snapshot()
            # snapshots.append(snapshot)

            # top_stats = snapshots[-1].compare_to(snapshots[-2], "lineno")

            # logging.info("[ Top 10 differences ]")
            # for stat in top_stats[:10]:
            #     logging.info(str(stat))

    # hm.mesh.points = new_hm.mesh.points.copy()
    hm.mesh.points = by_value(new_hm.mesh.points)
    return hm


def _compute_edge_normals(
    hm,
    cell_layer,
    column_depths,
    layer_idx,
    weight_toward_bone: float = 0.2,
    reverse_smoothing_direction: bool = False,
):
    is_layer_elem = cell_layer == layer_idx
    hm.mesh.cell_data["column_depth"][is_layer_elem] = column_depths
    grid = hm.mesh.extract_cells(is_layer_elem)
    # Figure out which one is more distal
    top_node_ids, bot_node_ids = _get_top_bottom_pids(reverse_smoothing_direction, grid)
    grid.point_data["mesh_pids"] = grid["vtkOriginalPointIds"]
    layer_hm = HexMesh(grid.cells_dict[VTK_HEXAHEDRON].copy(), grid.points.copy())
    layer_hm.mesh = grid
    grid.cell_data["depths"] = cartilage_layer_depth(layer_hm).mean(axis=1)
    bound_edges, bound_cells = layer_hm.perimeter_edges
    bound_edges["grid_pids"] = bound_edges["vtkOriginalPointIds"]
    bound_edges["depths"] = grid.cell_data["depths"][bound_edges["vtkOriginalCellIds"]]
    top_edge_pids = top_node_ids[
        np.isin(top_node_ids, bound_edges["vtkOriginalPointIds"])
    ]
    grid_surf = grid.extract_surface(algorithm=None)
    surf = grid_surf.extract_points(
        np.isin(grid_surf["vtkOriginalPointIds"], top_edge_pids)
    )
    surf_pd = pv.PolyData(surf.points, surf.cells)
    surf_pd.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    surf.point_data["edge_normals"] = surf_pd.point_normals
    grid_surf["edge_normals"][surf["vtkOriginalPointIds"]] = surf["edge_normals"]
    grid.point_data["edge_normals"][grid_surf["mesh_pids"]] = grid_surf["edge_normals"]
    grid.point_data["is_internal_point"][bound_cells["vtkOriginalPointIds"]] = False
    grid["edge_normals"][grid["is_internal_point"].astype(bool)] = np.zeros(3)
    grid.set_active_vectors("edge_normals", preference="point")

    surf = bound_cells.extract_surface(algorithm=None)
    surf.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    grid.point_data["edge_normals"][bound_cells["vtkOriginalPointIds"]] = np.average(
        np.stack(
            (
                surf.point_data["Normals"],
                grid.point_data["edge_normals"][bound_cells["vtkOriginalPointIds"]],
            )
        ),
        axis=0,
        weights=(1 - weight_toward_bone, weight_toward_bone),
    )
    grid.set_active_vectors("edge_normals", preference="point")
    bound_edges["edge_normals"] = grid.point_data["edge_normals"][
        bound_edges["vtkOriginalPointIds"]
    ]
    return grid, bound_edges


def find_bad_jacobians(
    mesh,
    min_jacobian: Union[float, int] = 0,
    divide_first: bool = False,
    show_plot: bool = False,
):
    if divide_first:
        nodes = mesh.mesh.points.copy()
        elems = mesh.mesh.cells_dict[VTK_HEXAHEDRON].copy()
        nodes, elems = by_value(nodes, elems)
        nodes, elems, _ = subdivide_cartilage_depth(nodes, elems)
        mesh_2 = HexMesh(elems, nodes)
    else:
        mesh_2 = mesh
    _, is_bad_cell = mesh_2.cell_quality(
        "scaled_jacobian",
        metric_min=min_jacobian,
        return_bad_cells=True,
        show_plot=show_plot,
    )
    del mesh_2
    return by_value(is_bad_cell)


def find_touching_corners(nodes: np.ndarray, elems: np.ndarray, **kwargs) -> np.ndarray:
    """Find pairs of cells touching only at a corner.

    Args:
        nodes (np.ndarray): an (n, 3) array of point locations.
        elems (np.ndarray): a (8, m) array of hexahedral element definitions.

    Returns:
        np.ndarray: An (n, 2) array of cell IDs representing pairs of
        corner-touching elements

    """
    fully_defined_node_mask = ~degens.get_missing_node_mask(elems, nodes)
    is_defined_cell = np.all(fully_defined_node_mask, axis=1)
    defined_cids = np.flatnonzero(is_defined_cell)

    # Filter down to cells having _at most_ two neighbors with fully connected faces
    corner_pair_cids = []
    for i_cid in defined_cids:
        i_cell_def = elems.T[i_cid]
        node_is_shared = np.isin(elems.T[defined_cids], i_cell_def)
        n_shared = node_is_shared.sum(axis=1)
        shares_corner_cids = defined_cids[np.flatnonzero(n_shared == 2)]
        shares_face_cids = defined_cids[np.flatnonzero(n_shared == 4)]
        for j_corner_cid in shares_corner_cids:
            j_corner_def = elems.T[j_corner_cid]
            face_defs = elems.T[shares_face_cids]
            is_shared_inception = np.isin(face_defs, j_corner_def)
            if ~is_shared_inception.any():
                corner_pair_cids.append(sorted([i_cid, j_corner_cid]))

    corner_pair_cids = np.unique(corner_pair_cids)
    if kwargs.get("debug", False) & corner_pair_cids.any():
        corner_pair_cell_defs = elems.T[corner_pair_cids.flatten()]
        all_mesh = UGrid.from_cell_def(points=nodes, cell_def=elems.T)
        hm = UGrid.from_cell_def(points=nodes, cell_def=corner_pair_cell_defs)
        p = pv.Plotter()
        p.add_mesh(all_mesh, opacity=0.8, culling="back", show_edges=True, label="mesh")
        p.add_mesh(hm, color="r", opacity=0.8, label="touching_corners")
        p.add_legend(size=[0.1, 0.1], origin=[0, 0.9])
        p.add_axes()
        del p
    return corner_pair_cids


def fix_touching_corners(
    corner_pair_cids: np.ndarray, nodes: np.ndarray, elems: np.ndarray
) -> np.ndarray:
    """Fill a cell adjacent to a pair of cells with only one corner touching.

    Args:
        corner_pair_cids (np.ndarray): a (n, 2) array of touching corner cell ID pairs.
        nodes (np.ndarray): a (m, 3) array of nodal locations.
        elems (np.ndarray): a (8, k) array of cell definitions.

    Returns:
        np.ndarray: the original nodal locations with additional node locatins filled in
        for the adjacent cells.

    """
    if corner_pair_cids.any():
        point_cloud_center = np.nanmean(nodes, axis=0)
        for cid_pair in np.atleast_2d(corner_pair_cids):
            corner_pair_cell_defs = elems.T[cid_pair.flatten()]

            # Fill in one of the cross-cells
            is_candidate = 6 == np.isin(elems.T, corner_pair_cell_defs).sum(axis=1)
            candidate_cids = np.flatnonzero(is_candidate)
            candidate_centers = np.nanmean(nodes[elems.T[candidate_cids]], axis=1)
            fill_idx = np.argmin(
                trimesh.util.row_norm(candidate_centers - point_cloud_center)
            )

            fill_cell_def = elems.T[candidate_cids[fill_idx]]
            new_locs = HexElement(nodes[fill_cell_def]).fill_missing_nodes().mesh.points
            nodes[fill_cell_def] = np.array(new_locs)
    return np.array(nodes.copy())
