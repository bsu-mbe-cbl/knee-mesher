from __future__ import annotations

import warnings
from typing import Dict, Tuple, Union

import numpy as np
import pyvista as pv
import trimesh
import vtk
from numpy.typing import ArrayLike
from vtk import VTK_HEXAHEDRON, VTK_QUAD  # noqa


def by_value(*pv_arrays: pv.pyvista_ndarray) -> np.ndarray:
    """Return a copy of a pyvista array not referencing the same object in memory."""
    corrected = [np.array(x, copy=True) for x in pv_arrays]
    return tuple(corrected) if len(corrected) > 1 else corrected[0]


class HexMesh:
    def __init__(
        self,
        cells: np.ndarray,
        nodes: np.ndarray,
        cell_type: int = None,
        name: str = None,
    ) -> None:
        cells = np.atleast_2d(cells)
        eights = np.full((cells.shape[0], 1), 8, dtype=cells.dtype)
        if cell_type is None:
            cell_type = np.full(eights.shape[0], vtk.VTK_HEXAHEDRON)
        cells = np.hstack((eights, cells))
        self.mesh = pv.UnstructuredGrid(cells.ravel(), cell_type, nodes)
        self.name = name
        self._initialize_plotter()

    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.mesh = self.mesh.copy()
        result._initialize_plotter()
        return result

    @property
    def top_bottom_point_ids(self):
        top_node_ids = np.unique(self.mesh.cells_dict[VTK_HEXAHEDRON][:, :4])
        bot_node_ids = np.unique(self.mesh.cells_dict[VTK_HEXAHEDRON][:, 4:])
        return top_node_ids, bot_node_ids

    @property
    def perimeter_cells(self):
        single_layer_grid = self.mesh
        surf = single_layer_grid.extract_surface(algorithm=None)
        cell_ids, face_counts = np.unique(
            surf["vtkOriginalCellIds"], return_counts=True
        )
        if face_counts.min() == 1:
            print("Not a single layer mesh, perimeter is undnefined.")
            return None
        else:
            perim_cell_ids = cell_ids[face_counts > 2]
            perim_cells = single_layer_grid.extract_cells(perim_cell_ids)
            perim_cells["face_counts"] = face_counts[face_counts > 2]
            return perim_cells

    @property
    def perimeter_edges(self):
        single_layer_grid = self.mesh
        perim_cells = self.perimeter_cells
        mesh_cid_str = "mesh_cids"
        mesh_pid_str = "mesh_pids"
        count = 0
        while mesh_cid_str in self.mesh.array_names:
            mesh_cid_str = f"mesh_cids_{count}"
            count += 1
        count = 0
        while mesh_pid_str in self.mesh.array_names:
            mesh_pid_str = f"mesh_pids_{count}"
            count += 1
        perim_cells.cell_data[mesh_cid_str] = perim_cells["vtkOriginalCellIds"]
        perim_cells.point_data[mesh_pid_str] = perim_cells["vtkOriginalPointIds"]

        # Extract surface
        surf = single_layer_grid.extract_surface(algorithm=None)
        surf.cell_data[mesh_cid_str] = surf["vtkOriginalCellIds"]
        surf.point_data[mesh_pid_str] = surf["vtkOriginalPointIds"]

        # region #FIXME: debug section
        # Cross-check the surface cells with top and bottom point ids
        bone_side_cids = self.mesh.cells_dict[VTK_HEXAHEDRON][:, :4]
        cart_side_pids = self.mesh.cells_dict[VTK_HEXAHEDRON][:, 4:]
        face_pids = surf.faces.reshape(-1, 5)[:, 1:]
        orig_face_pids = surf.point_data[mesh_pid_str][face_pids]
        is_bone_side_cell = np.isin(orig_face_pids, bone_side_cids).all(axis=1)
        is_cart_side_cell = np.isin(orig_face_pids, cart_side_pids).all(axis=1)
        # The following line casts the surface to UnstructuredGrid, and is important for
        # point ID matching down the chain.
        boundary_surf_cells = surf.extract_cells(np.arange(surf.n_cells))  # important!
        boundary_surf_cells.remove_cells(
            (is_bone_side_cell | is_cart_side_cell), inplace=True
        )
        boundary_surf_edges = boundary_surf_cells.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=True,
        )
        boundary_surf_edges["vtkOriginalPointIds"] = boundary_surf_edges.point_data.pop(
            mesh_pid_str
        )
        boundary_surf_edges["vtkOriginalCellIds"] = boundary_surf_edges.cell_data.pop(
            mesh_cid_str
        )

        boundary_surf_cells["vtkOriginalPointIds"] = boundary_surf_cells.point_data.pop(
            mesh_pid_str
        )
        boundary_surf_cells["vtkOriginalCellIds"] = boundary_surf_cells.cell_data.pop(
            mesh_cid_str
        )
        return boundary_surf_edges, boundary_surf_cells

    def _initialize_plotter(self, **kwargs) -> None:
        self.plotter = pv.Plotter()
        self.plotter.add_mesh(self.mesh, show_edges=True, label=self.name, **kwargs)
        self.plotter.add_axes()

    def _initialize_surface(
        self,
        force: bool = False,
    ) -> None:
        if "surface" not in self.__dict__ or force:
            self.surface = self.mesh.extract_surface(algorithm=None)

    def add_surface_normals(
        self,
        nodes: bool = True,
        faces: bool = False,
    ) -> None:
        mesh = self.mesh
        self._initialize_surface()
        surface = self.surface
        original_ids = surface["vtkOriginalPointIds"]
        if nodes:
            mesh_point_normals = np.full_like(mesh.points, 0)
            mesh_point_normals[original_ids] = surface.point_normals
            mesh.point_data["surface_point_normals"] = mesh_point_normals
            arrows = mesh.glyph(
                scale="surface_point_normals", orient="surface_point_normals", factor=2
            )
            self.plotter.add_mesh(arrows, color="yellow")
            self.mesh = mesh
        if faces:
            point_data = surface.cell_centers()
            point_data.vectors = surface.cell_normals * 2
            self.plotter.add_mesh(point_data.arrows, color="white")

    def show_boundary_edges(self):
        p = pv.Plotter()
        p.add_mesh(self.mesh, opacity=0.5)
        p.add_mesh(self.perimeter_cells, opacity=0.9)
        p.add_mesh(self.perimeter_edges[0], color="r")
        p.show()

    def show(
        self,
        node_normals: bool = False,
        face_normals: bool = False,
        add_mesh_kwargs: Dict = None,
        **kwargs,
    ) -> None:
        if add_mesh_kwargs:
            self._initialize_plotter(**add_mesh_kwargs)
        self.add_surface_normals(nodes=node_normals, faces=face_normals)
        self.plotter.show(**kwargs)
        self._initialize_plotter()

    def apply_graduated_depth_smoothing(self, n_iter=50):
        # smooth surface layer by surface layer, reducing the number of laplacian
        # smoothing iterations with each increasingly deep layer.
        ug = self.mesh.copy()
        ug.cell_data["mesh_cids"] = np.arange(ug.n_cells)
        ug.point_data["mesh_pids"] = np.arange(ug.n_points)
        point_array = ug.points.copy()

        scaler = 0.8
        n_iter = n_iter / scaler
        while ug.n_cells > 0:
            # extract the surface points
            surf = ug.extract_surface(algorithm=None)
            n_iter = int(n_iter * scaler)
            try:
                surf.smooth_taubin(
                    n_iter=n_iter, inplace=True, normalize_coordinates=True
                )
            except AttributeError:
                surf.smooth(n_iter=n_iter, inplace=True)

            point_ids = surf.point_data["mesh_pids"]
            point_array[point_ids] = surf.points

            # extract the cells not containing these points
            keep_pids_mask = ~np.isin(ug.point_data["mesh_pids"], point_ids)
            ug = ug.extract_points(keep_pids_mask, adjacent_cells=False)

        self.mesh.points = point_array.copy()

    def cell_quality(
        self,
        quality_metric: str = "scaled_jacobian",
        metric_min: Union[float, int] = None,
        metric_max: Union[float, int] = None,
        return_bad_cells: bool = False,
        show_plot: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        jac = self.mesh.cell_quality(quality_metric)
        jac[quality_metric] = jac[quality_metric]
        jac.set_active_scalars(quality_metric)
        is_bad_cell = np.full_like(jac[quality_metric], False, dtype=bool)
        is_bad_cell = (
            jac[quality_metric] < metric_min if metric_min is not None else is_bad_cell
        )
        is_bad_cell = (
            jac[quality_metric] > metric_max if metric_max is not None else is_bad_cell
        )
        opacity = 1 if not is_bad_cell.any() else 0.5
        if show_plot:
            p = pv.Plotter()
            p.add_mesh(jac, show_edges=True, opacity=opacity)
            if is_bad_cell.any():
                bad_jac = jac.extract_cells(is_bad_cell)
                p.add_mesh(bad_jac)
                p.add_mesh(bad_jac.extract_all_edges(), color="y", opacity=0.9)
                p.add_axes()
            p.show()
        return (
            jac[quality_metric]
            if not return_bad_cells
            else (jac[quality_metric], is_bad_cell)
        )


class HexElement:
    r"""Represent a single, first-order hexahedral element.

    Here is the topological element definition, to maintain a positive volume.

    .. code-block::

             (external)
                  z
                  |
                  |
                  4 ---------- 7
                 /|           /|
                / |          / |
               5 -|-------- 6  |
               |  0 --------|- 3 ---------- y (axial)
               | /          | /
               |/           |/
               1 ---------- 2
              /
             /
            x
        (radial)

    Attributes:
        mesh (pv.UnstructuredGrid): The underlying pyvista object representing hex
            element topology.
        nodal_connectivity (np.ndarray): An array of nodal connectivity between node
            IDs.

    """

    def __init__(
        self,
        points: ArrayLike = None,
        cells: ArrayLike = None,
        cell_type: int = None,
    ) -> None:
        """Initialize a HexElement using 8 nodes and a single cell definition.

        Args:
            points (ArrayLike, optional): The (n, 3) nodal locations. Defaults to None.
            cells (ArrayLike, optional): A (n + 1, ) cell definition of node IDs
                (indexes). Defaults to None.
            cell_type (int, optional): An integer mapping to vtk cell types. When None,
                the element type is assumed to be vtk.VTK_HEXAHEDRON. Defaults to None.

        """
        if cells is None:
            cells = np.array(
                [8, 0, 1, 2, 3, 4, 5, 6, 7]
            )  # first one says how many nodes
        else:
            if len(cells) == 8:
                cells = np.append(np.array([8]), np.array(cells))
        self.cell_def = cells[1:]
        cells = np.array([8] + list(range(8)))
        if cell_type is None:
            cell_type = np.array([vtk.VTK_HEXAHEDRON])
        if points is None:
            points = np.array(
                [
                    [0, 0, 0],  # 0
                    [1, 0, 0],  # 1
                    [1, 1, 0],  # 2
                    [0, 1, 0],  # 3
                    [0, 0, 1],  # 4
                    [1, 0, 1],  # 5
                    [1, 1, 1],  # 6
                    [0, 1, 1],  # 7
                ],
                dtype=float,
            )
        else:
            points = points[self.cell_def]
        self.mesh = pv.UnstructuredGrid(cells, cell_type, points)
        self._build_face_node_id_maps()

    def _build_face_node_id_maps(self):
        base = np.full(8, False)
        base[np.array((0, 1, 4, 5))] = True
        self.face_masks = dict(
            front=base,
            back=np.roll(base, 2),
            left=np.roll(base, -1),
            right=np.roll(base, 1),
            top=np.hstack((np.full(4, True), np.full(4, False))),
            bottom=np.hstack((np.full(4, False), np.full(4, True))),
        )

    def get_face_mask(self, *faces):
        if len(faces) == 1 and faces[0].lower() == "all":
            return np.vstack(list(self.face_masks.values()))
        masks = np.full((len(faces), 8), False)
        for face_idx, face in enumerate(faces):
            masks[face_idx] = self.face_masks[face]
        return masks

    @property
    def nodal_connectivity(self) -> np.ndarray:
        return self.mesh.extract_all_edges().lines.reshape((-1, 3))[:, 1:]

    def corner_connectivity(
        self,
        idx: int,
        return_nodes: bool = False,
        list_corner_first: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if np.isnan(self.mesh.points).any():
            connectivity = HexElement().nodal_connectivity
        else:
            connectivity = self.nodal_connectivity
        has_idx = np.any(connectivity == idx, axis=1)
        leg_segments = connectivity[has_idx]

        if list_corner_first:
            needs_flip = np.nonzero(leg_segments == idx)[1].astype(bool)
            leg_segments[needs_flip] = leg_segments[needs_flip, ::-1]

        all_leg_nodes = leg_segments[leg_segments != idx]
        if not return_nodes:
            return leg_segments
        else:
            return leg_segments, all_leg_nodes

    def get_hypotenuse_node(
        self,
        leg_segments: ArrayLike,
    ):
        leg_segments = np.array(leg_segments)
        u_nodes, u_counts = np.unique(leg_segments, return_counts=True)
        corner_node = u_nodes[u_counts.argmax()]
        leg_node_idxs = leg_segments[leg_segments != corner_node][:2]

        next_leg_node_idxs = np.hstack(
            [self.corner_connectivity(x, return_nodes=True)[1] for x in leg_node_idxs]
        )
        potential_hypotenuse = next_leg_node_idxs[next_leg_node_idxs != corner_node]
        u_nodes, u_counts = np.unique(potential_hypotenuse, return_counts=True)
        hypotenuse_node = u_nodes[np.argmax(u_counts)]
        return hypotenuse_node

    def get_opposite_node(
        self,
        idx: int,
    ):
        corner_idx = idx
        leg_segments, leg_nodes = self.corner_connectivity(
            corner_idx, return_nodes=True
        )
        hypotenuse_node = self.get_hypotenuse_node(leg_segments[:2])
        hyp_leg_segments, hyp_leg_nodes = self.corner_connectivity(
            hypotenuse_node, return_nodes=True
        )
        is_opposite = np.all(np.array([hyp_leg_nodes != x for x in leg_nodes]), axis=0)
        opposite_node = hyp_leg_nodes[is_opposite][0]
        return opposite_node

    def plot(self):
        p = pv.Plotter()
        p.add_mesh(
            self.mesh,
            show_edges=True,
        )
        p.add_axes()
        p.add_point_labels(
            self.mesh.points,
            self.cell_def,
        )
        p.show()

    def fill_missing_nodes(
        self,
        face_mask_query: str = "all",
        contraction_factor: float = 0.0,
    ) -> HexElement:
        """Estimate missing nodal locations from faces with three established nodes.

        For any face with a single node missing, the location of the missing node can be
        estimated using vector addition. If the missing node shares multiple faces where
        this is possible, we can average the estimates. Sometimes, missing nodes occur
        when an automatic meshing algorithm traces the edges of a hole in the
        reconstruction. In this case, it is beneficial to constrict or expand the
        estimated location toward or away from the hypotenuse nodes before averaging.

        Example:
            HexElement().fill_missing_nodes(
                face_mask_query=['top', 'right'],
                constriction_factor=0.8
            )

        Args:
            face_mask_query (str, optional): Limit the possible 3-node faces to base
                estimated locations on. Defaults to "all".
            constriction_factor (float, optional): the amount to reduce the estimated
                hypotenuses by. Defaults to 0.0.

        Returns:
            HexElement: the updated HexElement object, for method chaining.

        """
        self.mesh.points = self._calculate_missing_node_locations(
            face_mask_query=face_mask_query, contraction_factor=contraction_factor
        )
        return self

    def _calculate_missing_node_locations(
        self,
        face_mask_query: str = "all",
        contraction_factor: float = 0.0,
    ) -> np.ndarray:
        points = self.mesh.points.copy()
        is_missing = np.isnan(points).any(axis=1)
        missing_idxs = np.flatnonzero(is_missing)
        for missing_idx in missing_idxs:
            points[missing_idx] = self.interpolate_node(
                interpolated_node_idx=missing_idx,
                face_query=face_mask_query,
                contraction_factor=contraction_factor,
            )
        return points

    def interpolate_node(
        self,
        interpolated_node_idx: int,
        face_query: str = "all",
        contraction_factor: float = 0.0,
    ) -> np.ndarray:
        points = self.mesh.points
        face_masks = self.get_face_mask(face_query.lower())
        estimated_points = np.tile(points, (face_masks.shape[0], 1, 1))
        for mask_idx, face_mask in enumerate(face_masks):
            face_point_ids = np.arange(8)[face_mask]
            is_missing = np.isnan(points[face_mask]).any(axis=1)
            if np.count_nonzero(is_missing) == 1:
                legs, adj_ids = self.corner_connectivity(
                    interpolated_node_idx, return_nodes=True
                )
                is_in_face = np.isin(adj_ids, face_point_ids)
                adj_ids = adj_ids[is_in_face]
                hyp_id = self.get_hypotenuse_node(legs[is_in_face])
                estimated = self.interpolate_final_face_node(
                    hyp_id, adj_ids, contraction_factor=contraction_factor
                )
                estimated_points[mask_idx, interpolated_node_idx] = estimated
        with warnings.catch_warnings():  # slices are filled in the calling loop
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            return np.nanmean(estimated_points, axis=0)[interpolated_node_idx]

    def interpolate_final_face_node(
        self,
        hypotenuse_node_idx: int,
        leg_node_idxs: ArrayLike,
        contraction_factor: float = 0.0,
    ) -> np.ndarray:
        leg_node_idxs = np.array(leg_node_idxs)
        hyp_point, *leg_points = self.mesh.points[
            np.hstack([hypotenuse_node_idx, leg_node_idxs.flatten()])
        ]
        estimated_point = np.sum(leg_points, axis=0) - hyp_point
        return estimated_point - contraction_factor * (estimated_point - hyp_point)

    def face_normals(
        self,
        nodes: ArrayLike,
        use_uconn_method: bool = False,
    ):
        nodes = np.array(nodes)
        if "surface" not in self.__dict__:
            self.surface = self.mesh.extract_surface(algorithm=None)

        if use_uconn_method:
            if nodes.shape[-1] != 4:
                raise AssertionError(f"Found {len(nodes)} in *nodes when expecting 4.")
            # FIXME: corrected n4 = n1 ./ norm(n4) in uconn.faceNormal:67
            surface = self.surface
            original_ids = surface["vtkOriginalPointIds"]
            faces = np.sort(original_ids[surface.faces.reshape(-1, 5)[:, 1:]], axis=1)
            nodes = np.sort(np.atleast_2d(nodes), axis=1)
            normals = np.zeros((nodes.shape[0], 3))
            for face_idx, face in enumerate(nodes):
                has_face = (face == faces).all(axis=1)
                actual_normal = surface.face_normals[has_face]
                cross_products = np.zeros((4, 3))
                for node_idx, n in enumerate(face):
                    connectivity = self.corner_connectivity(n, list_corner_first=True)
                    in_plane_legs = connectivity[
                        np.isin(connectivity, face).all(axis=1)
                    ]
                    points = self.mesh.points[np.fliplr(in_plane_legs)]
                    vecs = np.diff(points, axis=1)
                    cross_product = trimesh.unitize(np.cross(*vecs))
                    angle_between = trimesh.geometry.vector_angle(
                        np.vstack((cross_product, actual_normal))
                    )
                    cross_products[node_idx] = (
                        -cross_product if angle_between > np.pi / 2 else cross_product
                    )
                normals[face_idx] = cross_products.mean(axis=0)
                angle_between = trimesh.geometry.vector_angle(
                    np.vstack((cross_products.mean(axis=0), actual_normal))
                )
                assert_msg = f"Face normals angle {angle_between} exceeds 90 degrees."
                assert angle_between < np.pi / 2, assert_msg

            return normals.mean(axis=0)

        else:
            surface = self.surface
            original_ids = surface["vtkOriginalPointIds"]
            faces = np.sort(original_ids[surface.faces.reshape(-1, 5)[:, 1:]], axis=1)
            nodes = np.sort(np.atleast_2d(nodes), axis=1)
            has_match = (nodes[:, np.newaxis] == faces).all(axis=2)
            match_idxs = np.nonzero(has_match.T)[0]
            surface.face_normals[match_idxs]
            return np.squeeze(surface.face_normals[match_idxs])


def _remove_orphan_point_cells(quad_surf_mesh: pv.PolyData) -> bool:
    cell_def = quad_surf_mesh.cells_dict[VTK_QUAD]
    pids, counts = np.unique(cell_def, return_counts=True)
    is_done = _check_if_all_boundary(quad_surf_mesh)
    if not is_done:
        single_cell_pids = pids[counts == 1]
        critical_cids = np.flatnonzero(np.isin(cell_def, single_cell_pids).any(axis=1))
        quad_surf_mesh.remove_cells(critical_cids, inplace=True)
        is_done = _check_if_all_boundary(quad_surf_mesh)
    return is_done


def _check_if_all_boundary(quad_surf_mesh: pv.PolyData) -> bool:
    cell_def = quad_surf_mesh.cells_dict[VTK_QUAD]
    pids, counts = np.unique(cell_def, return_counts=True)
    is_done = True if (counts == 2).all() else False
    return is_done


def _remove_bridging_faces(quad_surf_mesh: pv.PolyData, min_connections: int) -> bool:
    # Find any remaining points that connect min_connections or more faces and
    # remove cells that contain only one of these points
    cell_def = quad_surf_mesh.cells_dict[VTK_QUAD]
    pids, counts = np.unique(cell_def, return_counts=True)
    four_cell_pids = pids[counts >= 4]
    critical_cids = np.flatnonzero(np.isin(cell_def, four_cell_pids).any(axis=1))
    crit_cell_def = cell_def[critical_cids]
    has_four_cell_pid = np.isin(crit_cell_def, four_cell_pids)
    is_extraneous = np.count_nonzero(has_four_cell_pid, axis=1) == 1
    extraneous_cids = critical_cids[is_extraneous]
    quad_surf_mesh.remove_cells(extraneous_cids, inplace=True)
    is_done = _check_if_all_boundary(quad_surf_mesh)
    return is_done


def _remove_single_element_peaks(perim_surf_cells: pv.PolyData) -> bool:
    cell_def = perim_surf_cells.cells_dict[VTK_QUAD]
    pids, counts = np.unique(cell_def, return_counts=True)
    three_cell_pids = pids[counts >= 3]
    critical_cids = np.flatnonzero(np.isin(cell_def, three_cell_pids).any(axis=1))
    crit_cells = perim_surf_cells.extract_cells(critical_cids)
    if crit_cells.n_cells > 0:
        crit_edges = crit_cells.extract_feature_edges(
            boundary_edges=False,
            feature_edges=True,
            manifold_edges=True,
            non_manifold_edges=True,
        )
        crit_boundaries = crit_cells.extract_feature_edges(
            feature_edges=False,
            boundary_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        # Find critical cells that contain a single boundary edge, then remove
        # them
        has_bound_and_edge = np.isin(
            crit_boundaries["vtkOriginalCellIds"],
            crit_edges["vtkOriginalCellIds"],
        )
        crit_cell_remove_cids = crit_boundaries["vtkOriginalCellIds"][
            has_bound_and_edge
        ]
        extraneous_cids = crit_cells["vtkOriginalCellIds"][crit_cell_remove_cids]
        perim_surf_cells.remove_cells(extraneous_cids, inplace=True)
    return _check_if_all_boundary(perim_surf_cells)
