from __future__ import annotations

from typing import Tuple

import numpy as np
import pyvista as pv

from kneemesher.errors import DegenerateError
from kneemesher.meshing.cells import HexElement, by_value  # noqa
from kneemesher.pv_helpers import UnstructuredGrid as UGrid


def fix_6_node_hex(
    elems: np.ndarray, nodes: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    #  Peaks: __|__
    #  Mirror: |__ __|
    #  3-stairs: __
    #              |__
    #                 |__
    #                    |__
    #                       |
    #  stairs: __
    #            |__
    #               |__
    #                  |
    #  corners: __
    #             |__
    #                |

    all_mesh = UGrid.from_cell_def(points=nodes, cell_def=elems.T)
    _, degen_elem_ids = find_6_node_elements(elems, nodes)
    degen_mesh = UGrid.from_cell_def(points=nodes, cell_def=elems.T[degen_elem_ids])
    p = pv.Plotter()
    p.add_mesh(all_mesh, opacity=0.8, culling="back", show_edges=True, label="mesh")
    p.add_mesh(degen_mesh, color="r", opacity=0.8, label="degenerates")
    p.add_legend(loc="upper right")
    p.add_axes()
    del p, degen_mesh
    # Find and fix peaks
    peak_elem_ids = find_peak_elem_ids(elems, nodes)
    peak_elems = elems.T[np.unique(peak_elem_ids)]
    if peak_elem_ids.size > 0:
        hm = UGrid.from_cell_def(points=nodes, cell_def=peak_elems)
        p = pv.Plotter()
        p.add_mesh(all_mesh, opacity=0.8, culling="back", show_edges=True, label="mesh")
        p.add_mesh(hm, color="r", opacity=0.8, label="peak")
        p.add_legend(loc="upper right")
        p.add_axes()
        del p, hm
        elems, nodes = fix_peaks(elems, nodes, peak_elem_ids=peak_elem_ids)

    # Find and fix the mirror stairs
    mirror_elem_ids = find_mirror_elem_ids(elems, nodes)
    mirror_elems = elems.T[np.unique(mirror_elem_ids)]
    if mirror_elem_ids.size > 0:
        hm = UGrid.from_cell_def(points=nodes, cell_def=mirror_elems)
        p = pv.Plotter()
        p.add_mesh(all_mesh, opacity=0.8, culling="back", show_edges=True, label="mesh")
        p.add_mesh(hm, color="r", opacity=0.8, label="mirror")
        p.add_legend(loc="upper right")
        p.add_axes()
        del p, hm
        elems, nodes = fix_mirror(elems, nodes, mirror_elem_ids)

    # Find and fix the 3-stairs
    step_elem_ids = find_step_elements(elems, nodes)
    step_elems = elems.T[np.unique(step_elem_ids)]
    step_mesh = UGrid.from_cell_def(points=nodes, cell_def=step_elems)
    if step_elem_ids.size > 0:
        hm = UGrid.from_cell_def(points=nodes, cell_def=step_elems)
        p = pv.Plotter()
        p.add_mesh(all_mesh, opacity=0.8, culling="back", show_edges=True, label="mesh")
        p.add_mesh(hm, color="r", opacity=0.8, label="steps")
        p.add_legend(loc="upper right")
        p.add_axes()
        del p, hm
        tri_stairs = find_tri_stairs(step_elem_ids=step_elem_ids)
        while tri_stairs.size > 0:
            tri_stair_elems = elems.T[np.unique(tri_stairs.flatten())]
            hm = UGrid.from_cell_def(points=nodes, cell_def=tri_stair_elems)
            p = pv.Plotter()
            p.add_mesh(
                all_mesh, opacity=0.8, culling="back", show_edges=True, label="mesh"
            )
            p.add_mesh(hm, color="r", opacity=0.8, label="steps")
            p.add_legend(loc="upper right")
            p.add_axes()
            del p, hm
            for pairwise_step_ids in tri_stairs:
                right_step_ids = tri_stairs[:, 1]
                left_step_ids = tri_stairs[:, 0]
                mid_step_id = pairwise_step_ids[0]
                is_shared_id = mid_step_id == right_step_ids
                if is_shared_id.any():
                    shared_row = np.flatnonzero(is_shared_id)
                    outer_step_ids = np.vstack(
                        (pairwise_step_ids[1], left_step_ids[shared_row])
                    )
                    break
            if kwargs.get("debug", False):
                couple_mesh = UGrid.from_cell_def(
                    points=nodes, cell_def=elems.T[outer_step_ids.flatten()]
                )
                ref_mesh = UGrid.from_cell_def(
                    points=nodes, cell_def=elems.T[mid_step_id]
                )
                p = pv.Plotter()
                p.add_mesh(all_mesh, opacity=0.8, show_edges=True, label="mesh")
                p.add_mesh(couple_mesh, opacity=0.8, color="b", label="outer_step")
                p.add_mesh(ref_mesh, color="r", opacity=0.8, label="mid_step")
                p.add_legend(loc="upper right")
                p.add_axes()
                p.show()
                del p, couple_mesh, ref_mesh
            elems, nodes = fix_tri_stairs(elems, nodes, mid_step_id, outer_step_ids)
            # Check for new mirrors that may have appeared
            mirror_elem_ids = find_mirror_elem_ids(elems, nodes)
            mirror_elems = elems.T[np.unique(mirror_elem_ids)]
            if mirror_elem_ids.size > 0:
                elems, nodes = fix_mirror(elems, nodes, mirror_elem_ids)
            # Check for tri_stairs again
            step_elem_ids = find_step_elements(elems, nodes)
            tri_stairs = find_tri_stairs(step_elem_ids=step_elem_ids)

    # Find more stairs
    step_elem_ids = find_step_elements(elems, nodes)
    step_elems = elems.T[np.unique(step_elem_ids)]
    step_mesh = UGrid.from_cell_def(points=nodes, cell_def=step_elems)
    p = pv.Plotter()
    p.add_mesh(all_mesh, opacity=0.8, show_edges=True, label="mesh")
    p.add_mesh(step_mesh, opacity=0.8, color="r", label="steps")
    del p, step_mesh
    if step_elem_ids.size > 0:
        for ids in step_elem_ids:
            hm = UGrid.from_cell_def(points=nodes, cell_def=elems.T[ids])
            p = pv.Plotter()
            p.add_mesh(
                all_mesh, opacity=0.8, culling="back", show_edges=True, label="mesh"
            )
            p.add_mesh(hm, color="r", opacity=0.8, label="steps")
            p.add_legend(loc="upper right")
            p.add_axes()
            del p, hm
        elems, nodes = fix_stairs(elems, nodes, step_elem_ids)
        all_mesh = UGrid.from_cell_def(points=nodes, cell_def=elems.T)

    # Find and fix corners
    _, degen_elem_ids = find_6_node_elements(elems, nodes)
    degen_mesh = UGrid.from_cell_def(points=nodes, cell_def=elems.T[degen_elem_ids])
    p = pv.Plotter()
    p.add_mesh(all_mesh, opacity=0.8, culling="back", show_edges=True, label="mesh")
    p.add_mesh(degen_mesh, color="r", opacity=0.8, label="degenerates")
    p.add_legend(loc="upper right")
    p.add_axes()
    del p, degen_mesh
    if degen_elem_ids.size > 0:
        elems, nodes = collapse_internal_corners(
            elems, nodes, degenerate_elem_ids=degen_elem_ids
        )
        all_mesh = UGrid.from_cell_def(points=nodes, cell_def=elems.T)
    del all_mesh
    return by_value(elems, nodes)


def find_mirror_elem_ids(
    elems: np.ndarray, nodes: np.ndarray, new_algorithm=False
) -> np.ndarray:
    n_shared_nodes_grid, cell_6_point_cids = find_6_node_elements(elems, nodes)
    if not new_algorithm:
        y, x = np.nonzero(n_shared_nodes_grid == 4)
        mirror_elem_ids = cell_6_point_cids[
            np.unique(np.sort((x, y), axis=0), axis=1).T
        ]
        return mirror_elem_ids
    else:
        pairs = np.array(np.nonzero(np.tril(n_shared_nodes_grid == 4)))
        if (n_shared_nodes_grid == 4).any():
            pairs = np.array(np.nonzero(np.tril(n_shared_nodes_grid == 4)))
            is_mirror_pair = np.full_like(pairs[0], False, dtype=bool)
            for pair_idx, (first_idx, second_idx) in enumerate(zip(*pairs)):
                first_cid = cell_6_point_cids[first_idx]
                second_cid = cell_6_point_cids[second_idx]
                first_cell_def = elems[:, first_cid]
                paired_cell_def = elems[:, second_cid]
                shared_pids = first_cell_def[np.isin(first_cell_def, paired_cell_def)]
                shared_face_has_nan = np.isnan(nodes[shared_pids]).any()
                if shared_face_has_nan:
                    is_mirror_pair[pair_idx] = True
            return np.unique(cell_6_point_cids[pairs[:, is_mirror_pair]]).reshape(-1, 2)


def fix_7_node_hex(elems: np.ndarray, missing_node_mask: np.ndarray) -> np.ndarray:
    """Fill undefined node definitions with their neighbor.

    For an element definition with a single undefined node, copy the neighbor node id
    into the missing location. Do this for the internal and external hexahedral planes,
    not the sides.

    Args:
        elems (np.ndarray): The element definitions with each column representing one
        hexahedral element.
        missing_node_mask (np.ndarray): A boolean mask the same dimension as elems, with
        true values representing missing nodal definitions.

    Returns:
        np.ndarray: The corrected element definitions.
    """
    elem_7_node_idxs = np.flatnonzero(missing_node_mask.sum(axis=1) == 1)
    elem_defs_with_7_nodes = elems[:, elem_7_node_idxs].T
    node_is_nan = missing_node_mask[elem_7_node_idxs, :]
    assert (node_is_nan.sum(axis=1) == 1).all(), "Detected fewer than 7 nodes."
    for elem_idx, (_, nan_loc) in enumerate(zip(elem_defs_with_7_nodes, node_is_nan)):
        nan_node_idx = nan_loc.argmax()
        # Copy the neighboring node in the internal / external "planes"
        if nan_node_idx == 3:
            elem_defs_with_7_nodes[elem_idx, nan_node_idx] = elem_defs_with_7_nodes[
                elem_idx, 0
            ]
        elif nan_node_idx == 7:
            elem_defs_with_7_nodes[elem_idx, nan_node_idx] = elem_defs_with_7_nodes[
                elem_idx, 4
            ]
        else:
            elem_defs_with_7_nodes[elem_idx, nan_node_idx] = elem_defs_with_7_nodes[
                elem_idx, nan_node_idx + 1
            ]
    elems = elems.T
    elems[elem_7_node_idxs] = elem_defs_with_7_nodes
    cleaned_elems = elems.T
    return cleaned_elems


def find_6_node_elements(
    elems: np.ndarray, nodes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Find elements with six nodes.

    Find element definitions which contain two undefined nodes, then create a grid which
    marks which elements share nodes with these degenerate nodes. For each row $i$ of
    the matrix $A$, the NaN node in the $j$th position is shared among $A_{ij}$
    elements.

    Args:
        elems (np.ndarray): Element definitions with each column being an element.
        nodes (np.ndarray): Cartesian nodal locations with each row being an element.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The grid, and the original element IDs of the
            elements which only have six defined nodes.
    """
    missing_point_mask = get_missing_node_mask(elems, nodes)
    n_missing_points = missing_point_mask.sum(axis=1)
    degenerate_cell_ids = np.flatnonzero(n_missing_points == 2)
    n_degenerate = degenerate_cell_ids.size
    incomplete_element_adjacency = np.zeros((n_degenerate, n_degenerate), dtype=int)
    degen_cell_defs = elems[:, degenerate_cell_ids]
    for iter_idx, cell_id in enumerate(degenerate_cell_ids):
        this_degen_cell_def = elems[:, cell_id]
        contains_bad_nodes = np.isin(degen_cell_defs.T, this_degen_cell_def).T
        incomplete_element_adjacency[iter_idx, :] = contains_bad_nodes.sum(axis=0)
    return incomplete_element_adjacency, degenerate_cell_ids


def get_missing_node_mask(elements: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Return a boolean array marking NaN node numbers.

    Groups the node definitions by element, then searches for NaN values.

    Args:
        elements (np.ndarray): The element definition with rows containing node indexes.
        nodes (np.ndarray): The node definition with one node per row.

    Returns:
        np.ndarray: A boolean array the same shape as `elements` where true entries
            represent node indexes with missing points.
    """
    point_ids = elements.flatten()
    point_locations = nodes[point_ids, :].T
    return np.isnan(point_locations).any(axis=0).reshape(-1, 8, order="F")


def collapse_internal_corners(
    element_defs: np.ndarray,
    node_locations: np.ndarray,
    degenerate_elem_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    new_nodes = node_locations.copy()

    for corner_elem_id in degenerate_elem_ids:
        corner_elem_node_ids = element_defs[:, corner_elem_id]
        corner_nodes = node_locations[corner_elem_node_ids]
        has_nan = np.isnan(corner_nodes).any(axis=1)
        first_nan_node_idx = np.flatnonzero(has_nan)[0]
        first_face = np.arange(4)
        he = HexElement(points=corner_nodes)
        legs, leg_nodes = he.corner_connectivity(first_nan_node_idx, return_nodes=True)
        is_in_face = np.isin(leg_nodes, first_face)
        hyp_idx = he.get_hypotenuse_node(legs[is_in_face])
        leg_idxs = leg_nodes[is_in_face]

        for offset in (0, 4):
            hyp_idx, leg_idxs = hyp_idx + offset, [x + offset for x in leg_idxs]
            new_loc = he.interpolate_final_face_node(
                hypotenuse_node_idx=hyp_idx,
                leg_node_idxs=leg_idxs,
                contraction_factor=0.0,
            )
            new_nodes[corner_elem_node_ids[leg_idxs]] = new_loc

    element_defs = np.delete(element_defs, degenerate_elem_ids, axis=1)
    return element_defs, new_nodes


def fix_stairs(
    elems: np.ndarray, nodes: np.ndarray, indStairs: np.ndarray
) -> np.ndarray:
    delete_mask = np.full((elems.shape[1]), False)


    for step_pair_idx, pairwise_step_ids in enumerate(indStairs):
        # #region-----------------------------------------------------------------------
        # find shared nodes and NaN nodes
        step_elem_defs = elems.T[pairwise_step_ids]
        step_elem_points = nodes[step_elem_defs]
        contains_nan = np.isnan(step_elem_points).all(axis=2)
        _, missing_idxs = np.nonzero(contains_nan)
        missing_pids = step_elem_defs.flatten()[np.flatnonzero(contains_nan)].reshape(
            -1, 2
        )
        is_shared = np.isin(*step_elem_defs)
        try:
            missing_idxs = missing_idxs.reshape(2, 2)
        except ValueError:
            has_two_entries = np.isin(indStairs, pairwise_step_ids).sum(axis=0)
            if (has_two_entries == np.array([2, 2])).all():
                continue
            else:
                errmsg = f"At least one of cells {pairwise_step_ids} is not degenrate."
                try:
                    raise DegenerateError(errmsg)
                except DegenerateError:
                    del elems
                    raise
        shared_pids = step_elem_defs[0, is_shared]
        shared_idxs = np.vstack(
            (np.flatnonzero(is_shared), np.flatnonzero(np.isin(*step_elem_defs[::-1])))
        )

        # TODO: If the shared nodes are not adjacent to the missing nodes, this is not a
        # step. It should be skipped. Alternatively, if any of the missing nodes are in
        # the corner opposite to the shared nodes, then this is not a step.
        for idxs, sed, step_cid in zip(shared_idxs, step_elem_defs, pairwise_step_ids):
            step_hex_elem = HexElement(nodes[sed])
            opp_idx = step_hex_elem.get_opposite_node(idxs[0])
            if np.isnan(nodes[sed[opp_idx]]).all():
                delete_mask[step_cid] = True
        if delete_mask[step_cid].any():
            continue
        if not np.allclose(*missing_idxs):
            continue

        # Find other touching elements
        touching_cids = np.setdiff1d(
            np.flatnonzero(np.isin(elems.T, shared_pids).any(axis=1)), pairwise_step_ids
        )
        touching_has_nan = np.isnan(nodes[elems.T[touching_cids]]).all(axis=2)
        touching_cid = touching_cids[np.all(~touching_has_nan, axis=1)]
        touching_cell_def = elems.T[touching_cid].flatten()
        touching_points = nodes[touching_cell_def]
        try:
            if touching_points.size == 0:
                raise DegenerateError(f"Cells {pairwise_step_ids} are not a step.")
        except DegenerateError as e:
            del elems
            raise e

        # find hypotenuse of touching
        touching = HexElement(touching_points)
        touching_shared_idxs = np.flatnonzero(np.isin(touching_cell_def, shared_pids))
        touching_hyp_idxs = np.flipud(
            [touching.get_opposite_node(x) for x in touching_shared_idxs]
        )
        touching_hyp_pids = touching_cell_def[touching_hyp_idxs]

        # find present but free of touching
        is_free = ~np.isin(step_elem_defs, touching_cell_def) & ~np.isin(
            step_elem_defs, missing_pids
        )
        free_pids = step_elem_defs[is_free].reshape(2, 2)
        _, free_idxs = np.nonzero(is_free)
        free_idxs = free_idxs.reshape(2, 2)

        src = np.hstack([free_idxs, missing_idxs, shared_idxs])
        unchanged_idxs = np.vstack([np.setdiff1d(np.arange(8), x) for x in src])
        src = np.hstack((unchanged_idxs, src))
        dst = np.hstack(
            [unchanged_idxs, missing_idxs, shared_idxs]
        )  # , np.tile(touching_hyp_idxs, (2, 1))])
        remaining_idxs = np.vstack([np.setdiff1d(np.arange(8), x) for x in dst])
        dst = np.hstack((dst, remaining_idxs))

        # add missing locations
        for elem_def in step_elem_defs:
            he = HexElement(nodes[elem_def])
            he.fill_missing_nodes()
            nodes[elem_def] = he.mesh.points
        del he

        rows = np.array((0, 1))[..., np.newaxis]

        # Find the neighbors who are attached to the free nodes
        accounted_idxs = np.hstack([missing_idxs, shared_idxs, free_idxs])
        remaining_idxs = np.vstack(
            [np.setdiff1d(np.arange(8), x) for x in accounted_idxs]
        )
        has_free_pid = np.isin(elems.T, free_pids).any(axis=1)
        has_missing_pid = np.isin(elems.T, missing_pids).any(axis=1)
        has_free_no_missing = has_free_pid & ~has_missing_pid
        has_free_yes_missing = has_free_pid & has_missing_pid
        neighbor_cids = np.setdiff1d(
            np.flatnonzero(has_free_no_missing), pairwise_step_ids
        )
        delete_mask[
            np.setdiff1d(np.flatnonzero(has_free_yes_missing), pairwise_step_ids)
        ] = True
        neighbor_defs = elems.T[neighbor_cids]
        first_neighbor_mask = np.isin(neighbor_defs, free_pids[0])
        second_neighbor_mask = np.isin(neighbor_defs, free_pids[1])
        neighbor_defs[first_neighbor_mask] = np.tile(
            missing_pids[0], (1, first_neighbor_mask.sum() // 2)
        ).flatten()
        neighbor_defs[second_neighbor_mask] = np.tile(
            missing_pids[1], (1, second_neighbor_mask.sum() // 2)
        ).flatten()

        # Point those neighbors free nodes to the missing nodes
        # TODO: Need to pick neighbors based on each step, then assing the correct
        # missing pid to all neighbors

        # Remap the cell definitions
        new_def = step_elem_defs.copy()
        new_def[rows, missing_idxs] = step_elem_defs[rows, shared_idxs]
        new_def[rows, shared_idxs] = np.tile(touching_hyp_pids, (2, 1))
        new_def[rows, free_idxs] = step_elem_defs[rows, missing_idxs]

        # Update the nodes to assist
        old_n_points = nodes.shape[0]
        nodes = np.vstack((nodes, nodes[missing_pids.flatten()]))
        for new_pid, old_pid in enumerate(missing_pids.flatten(), old_n_points):
            new_def[new_def == old_pid] = new_pid
            neighbor_defs[neighbor_defs == old_pid] = new_pid

        # assign the new definitions
        elems[:, pairwise_step_ids] = new_def.T
        elems[:, neighbor_cids] = neighbor_defs.T
        nodes[missing_pids] = np.full(3, np.nan)
        # Mark the touching_cids for deletion
        delete_mask[touching_cid] = True

        # #endregion--------------------------------------------------------------------
    elems = elems[:, ~delete_mask]
    return elems, nodes


def new_func(elems, pairwise_step_ids):
    step_elem_defs = elems.T[pairwise_step_ids]
    return step_elem_defs


def fix_tri_stairs(
    elems: np.ndarray, nodes: np.ndarray, idx_ref: np.ndarray, indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # 3-stairs: __
    #             |__
    #                |__
    #                   |__
    #                      |
    #
    # #           long_idx
    # #    |        /
    # #    |       /
    # #    |  3 ---------- 2
    # #    | /|  /        /|
    # #    |/ | /        / |
    # #    0 -|-------- 1  |
    # #    |  7 --------|- 6
    # #    | /          | /
    # #    |/           |/
    # #    4 ---------- 5 ---------- radial_idx

    n_copy = nodes.copy()
    whole_mesh = UGrid.from_cell_def(points=n_copy, cell_def=elems.T)
    elems_of_interest = elems[:, np.vstack((indices, idx_ref)).flatten()].T
    mesh_of_interest = UGrid.from_cell_def(n_copy, cell_def=elems_of_interest)
    stair_elem = elems[:, idx_ref]
    peak_nodes = nodes[stair_elem]
    has_nan = np.isnan(peak_nodes)
    case = np.flatnonzero(has_nan.flatten(order="F"))[0]
    case2 = np.flatnonzero(has_nan.any(axis=1))[0]
    assert case2 == case
    p = np.flatnonzero(np.isin(elems[:, idx_ref], elems[:, indices[0]]))[0]

    def get_superior_inferior(p, case):
        if p == case:
            supe = indices[0]
            infe = indices[1]
        else:
            supe = indices[1]
            infe = indices[0]
        return supe, infe


    if case == 0:
        supe, infe = get_superior_inferior(p, 3)
        # extrapolate_final_edge((0, 1, 3, 2), (4, 0, 6, 2))

    elif case == 1:
        supe, infe = get_superior_inferior(p, 0)
        # extrapolate_final_edge((1, 0, 2, 3), (5, 1, 7, 3))

    elif case == 2:
        supe, infe = get_superior_inferior(p, 1)
        # extrapolate_final_edge((2, 3, 1, 0), (6, 2, 4, 0))

    elif case == 3:
        supe, infe = get_superior_inferior(p, 2)
        # extrapolate_final_edge(
        #     (3, 2, 0, 1), (7, 3, 5, 1)
        # )  # to maintain new edge length
        # enforce_8_nodes((3, 2, 0, 1), (7, 6, 4, 5))
    elem_def = elems.T[infe.squeeze()]
    new_points = HexElement(nodes[elem_def]).fill_missing_nodes().mesh.points
    nodes[elem_def] = new_points

    corrected_mesh = UGrid.from_cell_def(nodes, cell_def=elems_of_interest)
    plotter = pv.Plotter()
    plotter.add_mesh(
        whole_mesh.mesh,
        opacity=0.8,
        culling="back",
    )
    plotter.add_mesh(whole_mesh.mesh.copy(), style="points")
    plotter.add_mesh(mesh_of_interest.mesh, color="r", opacity=0.5)
    plotter.add_mesh(corrected_mesh.mesh, color="b", opacity=0.5, show_edges=True)
    del plotter, whole_mesh, mesh_of_interest, corrected_mesh
    return elems, nodes


def fix_mirror(
    element_defs: np.ndarray,
    node_locations: np.ndarray,
    mirror_elem_ids: np.ndarray,
    uconn_method: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    # Mirror: |__ __|

    for mirror_pair in np.atleast_2d(mirror_elem_ids):
        node_ids = element_defs[:, mirror_pair]
        left_ids, right_ids = node_ids.T
        unique_node_ids = np.unique(node_ids)
        is_missing_node = np.isnan(node_locations[unique_node_ids]).any(axis=1)
        missing_node_ids = unique_node_ids[is_missing_node]
        if uconn_method:
            node_ids = node_ids.T.flatten()
            is_in_left = np.isin(left_ids, right_ids)
            is_in_right = np.isin(right_ids, left_ids)
            is_shared = np.hstack((is_in_left, is_in_right))
            node_ids = node_ids[~is_shared]
            left_locs = node_locations[node_ids[[0, 1, 5, 4]]]  # hex topology
            right_locs = node_locations[node_ids[[2, 3, 7, 6]]]  # hex topology
            node_locations[missing_node_ids[0], :] = left_locs.mean(axis=0)
            node_locations[missing_node_ids[1], :] = right_locs.mean(axis=0)
        else:
            left_points = (
                HexElement(points=node_locations[left_ids])
                .fill_missing_nodes(contraction_factor=0.5)
                .mesh.points
            )
            right_points = (
                HexElement(points=node_locations[right_ids])
                .fill_missing_nodes(contraction_factor=0.5)
                .mesh.points
            )
            for miss_id in missing_node_ids:
                node_locations[miss_id] = np.vstack(
                    (
                        left_points[left_ids == miss_id],
                        right_points[right_ids == miss_id],
                    )
                ).mean(axis=0)
    return element_defs, node_locations


def find_peak_elem_ids(
    elems: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    """Find element IDs where two adjacent six-node elements share a single face.

    While two elements must share a peak, only one of each pair will be returned.

    > Note: See the documentation for `fix_peaks` for a more detailed definition of
      peaks.

    Args:
        elems (np.ndarray): a (8, n) array of node IDs.
        nodes (np.ndarray): an (n, 3) array of nodal locations.

    Returns:
        Union[np.ndarray, None]: a list of element IDs which contain peak elements.
    """
    adjacency, cell_6_point_cids = find_6_node_elements(elems, nodes)

    pairs = np.array(np.nonzero(np.tril(adjacency == 4)))
    if (adjacency == 4).any():
        pairs = np.array(np.nonzero(np.tril(adjacency == 4)))
        is_peak_pair = np.full_like(pairs[0], False, dtype=bool)
        for pair_idx, (first_idx, second_idx) in enumerate(zip(*pairs)):
            first_cid = cell_6_point_cids[first_idx]
            second_cid = cell_6_point_cids[second_idx]
            first_cell_def = elems[:, first_cid]
            paired_cell_def = elems[:, second_cid]
            shared_pids = first_cell_def[np.isin(first_cell_def, paired_cell_def)]
            shared_face_has_nan = np.isnan(nodes[shared_pids]).any()
            if ~shared_face_has_nan:
                is_peak_pair[pair_idx] = True
        return np.unique(cell_6_point_cids[pairs[:, is_peak_pair]])
    else:
        return pairs.T  # compatibility



def fix_peaks(
    elems: np.ndarray,
    nodes: np.ndarray,
    peak_elem_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrapolate missing nodes when two 6-node cells share a middle face.

        *                         *---*---*
        |            ->           |   |   |
    *---*---*                     *---*---*

    To estimate the missing node location, vector addition is performed in plane from
    the opposite (hypotenuse) nodes, adding the two legs.

       O   * a1
           |         O = h + (a0 - h) + (a1 - h)
    a0 *---* h         = a0 + a1 - h

    Args:
        elems (np.ndarray): the (8, n) array of node IDs defining the cells of the mesh.
        nodes (np.ndarray): the (m, 3) array of nodal locations.
        peak_elem_ids (np.ndarray): Element IDs for elements containing a peak.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the updated (elements, nodes) definitions.
    """
    for peak_elem_id in peak_elem_ids:
        node_ids = elems.T[peak_elem_id]
        points = nodes[node_ids]
        filled_points = HexElement(points=points).fill_missing_nodes().mesh.points
        nodes[node_ids] = filled_points

    return elems, nodes


def find_step_elements(
    elems: np.ndarray,
    nodes: np.ndarray,
    new_algorithm: bool = False,
) -> np.ndarray:
    degen_adjacency, degen_elem_ids = find_6_node_elements(
        elems, nodes
    )  # was originally calculated before mirror
    if not new_algorithm:
        y, x = np.nonzero(degen_adjacency == 2)
        step_elem_idxs = np.unique(np.sort((x, y), axis=0), axis=1).T
        step_elem_ids = degen_elem_ids[step_elem_idxs]
        return step_elem_ids

    if (degen_adjacency == 2).any():
        pairs = np.array(np.nonzero(np.tril(degen_adjacency == 2)))
        is_peak_pair = np.full_like(pairs[0], False, dtype=bool)
        for pair_idx, (first_idx, second_idx) in enumerate(zip(*pairs)):
            first_cid = degen_elem_ids[first_idx]
            second_cid = degen_elem_ids[second_idx]
            first_cell_def = elems[:, first_cid]
            paired_cell_def = elems[:, second_cid]
            shared_pids = first_cell_def[np.isin(first_cell_def, paired_cell_def)]
            shared_face_has_nan = np.isnan(nodes[shared_pids]).any()
            if ~shared_face_has_nan:
                is_peak_pair[pair_idx] = True
        return np.unique(degen_elem_ids[pairs[:, is_peak_pair]])


def find_tri_stairs(step_elem_ids: np.ndarray) -> np.ndarray:
    first_cid = step_elem_ids[:, 0]
    second_cid = step_elem_ids[:, 1]
    has_both = np.isin(first_cid, second_cid)
    if has_both.any():
        has_both_idx = np.flatnonzero(has_both)[0]
        x, y = np.fliplr(
            np.nonzero(step_elem_ids == step_elem_ids.flatten(order="F")[has_both_idx])
        )
        tri_stairs = step_elem_ids[x, :]
        return tri_stairs
    else:
        return np.array([(), ()]).T
