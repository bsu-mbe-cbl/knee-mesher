from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from scipy.signal import savgol_filter  # noqa
from trimesh import Trimesh
from typing_extensions import Self
from vtk import VTK_HEXAHEDRON, VTK_QUAD  # noqa

import kneemesher.meshing.hexa_creation as hc
import kneemesher.meshing.raytracing as rays
from kneemesher.errors import (
    BlendingError,
    CartilageMeshError,
    LimitReachedError,
    OptimizationError,
)
from kneemesher.meshing.progress import WorkProgress, report, stage
from kneemesher.metadata import CartilageType, TissueType
from kneemesher.preprocessing import clamp_triangle_count_and_smooth
from kneemesher.pv_helpers import (
    UnstructuredGrid,
    _by_value,
    without_activation,
)
from kneemesher.utilities import (
    import_matlab_data,
    with_debug_kwarg,  # noqa: F401
)

resource_dir = Path.cwd().joinpath("tests", "resources")
matfile_dir = resource_dir.joinpath("mat-files")


if TYPE_CHECKING:
    from kneemesher._typing import (
        ArrayLike,
        PolyData,
        Self,
        UnstructuredGrid,
        npt,
    )


TISSUE_COLORS = dict(
    background=[0, 0, 0, 255],
    patella_cartilage=[255, 127, 0, 255],  # orange
    femur_cartilage=[51, 160, 44, 255],  # green
    tibia_bone=[202, 178, 214, 255],  # light purple
    tibia_cartilage_lateral=[166, 206, 227, 255],  # light blue
    tibia_cartilage_medial=[251, 154, 153, 255],  # light red
    patella_bone=[253, 191, 111, 255],  # light orange
    femur_bone=[178, 223, 138, 255],  # light green
    fibula_bone=[178, 223, 138, 255],  # light green
    meniscus_lateral=[31, 120, 180, 255],  # blue
    meniscus_medial=[227, 26, 28, 255],  # red
)


TISSUE_COLORS_FOR_PUBLISHING = dict(
    background=[255, 255, 255, 255],
    femur_bone=[227, 218, 201, 255],  # bone
    femur_cartilage=[200, 200, 235, 255],  # cartilage
    meniscus_lateral=[178, 212, 242, 255],  # yellow
    meniscus_medial=[178, 212, 242, 255],  # yellow
    patella_bone=[227, 218, 201, 255],  # bone
    patella_cartilage=[111, 184, 210, 255],  # cartilage
    tibia_bone=[227, 218, 201, 255],  # bone
    fibula_bone=[227, 218, 201, 255],  # bone
    tibia_cartilage_lateral=[111, 184, 210, 255],  # cartilage
    tibia_cartilage_medial=[111, 184, 210, 255],  # cartilage
)


@dataclass
class TissueNameConverter:
    femur_bone: str = "FEM_BONE_G"
    femur_cartilage: str = "FEM_CART_G"
    fibula_bone: str = "FIB_BONE_G"
    meniscus_lateral: str = "MEN_LAT_G"
    meniscus_medial: str = "MEN_MED_G"
    patella_bone: str = "PAT_BONE_G"
    patella_cartilage: str = "PAT_CART_G"
    tibia_bone: str = "TIB_BONE_G"
    tibia_cartilage_lateral: str = "TIB_CART_LAT_G"
    tibia_cartilage_medial: str = "TIB_CART_MED_G"

    def __post_init__(self) -> None:
        self.tissue_dict = {
            "femur_cartilage": self.femur_cartilage,
            "patella_cartilage": self.patella_cartilage,
            "tibia_cartilage_medial": self.tibia_cartilage_medial,
            "tibia_cartilage_lateral": self.tibia_cartilage_lateral,
            "meniscus_medial": self.meniscus_medial,
            "meniscus_lateral": self.meniscus_lateral,
            "femur_bone": self.femur_bone,
            "patella_bone": self.patella_bone,
            "tibia_bone": self.tibia_bone,
            "fibula_bone": self.fibula_bone,
        }

    def values_as_keys(self, dictionary) -> Dict:
        return_dict = {}
        for k, v in self.tissue_dict.items():
            return_dict[v] = dictionary.get(k, None)
        return {k: v for k, v in return_dict.items() if v is not None}

    def keys_as_keys(self, dictionary) -> Dict:
        return_dict = {}
        for k, v in self.tissue_dict.items():
            return_dict[k] = dictionary.get(k, None)
        return {k: v for k, v in return_dict.items() if v is not None}


class TissueMesh(UnstructuredGrid):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reconstruction_filename = None
        self._reconstruction_tri_target = None
        self._reconstruction_smooth_iters = None
        self._cached_reconstruction = None
        self._add_persistent_pc_id_data()

    def update_cell_def(self, cell_def: ArrayLike, points: ArrayLike = None) -> Self:
        points = self.point_locs if points is None else points
        new_mesh = self.__class__.from_cell_def(points=points, cell_def=cell_def)
        new_mesh.set_state(self.get_state())
        new_mesh._add_persistent_pc_id_data()
        return new_mesh

    def get_state(self) -> Dict:
        """Get state dictionary of this cartilage mesh, for copying to new meshes."""
        state = self.__dict__.copy()
        return state

    def set_state(self, state: Dict):
        """Set state dictionary of this cartilage mesh, for copying from existing mesh."""
        self.__dict__.update(state)

    def copy(self, *args, **kwargs) -> Self:
        """Create a copy of this mesh."""
        new_obj = super().copy(*args, **kwargs)
        new_obj.set_state(self.get_state())
        return new_obj

    def plot_with_reconstruction(self) -> None:
        """Plot the cartilage mesh with its STL reconstruction."""
        if self.reconstruction is not None:
            add_mesh_kwargs = self.add_mesh_kwargs(opacity=0.5)
            p = self.plotter_with_mesh(add_mesh_kwargs=add_mesh_kwargs)
            p.add_mesh(pv.wrap(self.reconstruction), opacity=0.5)
            p.show()

    def attach_reconstruction(
        # self, filename: Union[str, Path], n_triangles: int, smooth_iters: int, **kwargs
        self,
        filename: str | Path,
        n_triangles: int,
        smooth_iters: int,
        **kwargs,
    ) -> None:
        """Attach a tissue reconstruction STL file to TissueMesh.

        The reconstruction geometry is loaded lazily. Make a subsequent
        cache_reconstruction method call to keep the geometry in memory.

        Args:
            filename: the path to the tissue geometry STL file.
            n_triangles: if the number of triangles in the tissue reconstruction exceeds
                this value, then decimate the mesh. Defaults to None.
            smooth_iters: the number of Laplacian smoothing iterations to apply to the
                reconstruction geometry.

        """
        self._reconstruction_filename = filename
        self._reconstruction_tri_target = n_triangles
        self._reconstruction_smooth_iters = smooth_iters

    def cache_reconstruction(self) -> None:
        """Cache tissue reconstruction in memory for faster loading or plotting."""
        self._cached_reconstruction = None
        self._cached_reconstruction = self.reconstruction

    @property
    def reconstruction(self) -> Trimesh | PolyData:
        """Tissue reconstruction geometry as a trimesh.TriMesh object."""
        if self._cached_reconstruction is not None:
            return self._cached_reconstruction
        elif self._reconstruction_filename is not None:
            return clamp_triangle_count_and_smooth(
                filename=self._reconstruction_filename,
                n_triangles=self._reconstruction_tri_target,
                smooth_iters=self._reconstruction_smooth_iters,
            )
        else:
            return None

    @reconstruction.setter
    def reconstruction(self, reconstruction: Union[Trimesh, PolyData]) -> None:
        self._reconstruction_filename = None
        self._reconstruction_tri_target = None
        self._reconstruction_smooth_iters = None
        self._cached_reconstruction = reconstruction

    @property
    def cell_def(self) -> npt.NDArray[np.int64]:
        """Return a copy of (n_cells, 8) cell definitions."""
        return _by_value(self.cells_dict[VTK_HEXAHEDRON])

    @cell_def.setter
    def cell_def(self, cdef: ArrayLike) -> None:
        eights = np.full((cdef.shape[0], 1), 8, dtype=cdef.dtype)
        self.cells = np.hstack((eights, cdef)).flatten()

    @property
    def point_locs(self) -> npt.NDArray[np.float64]:
        """Return a (n_points, 3) copy of point locations."""
        return _by_value(self.points)

    @property
    def wrapped_reconstruction(self) -> PolyData:
        """Tissue reconstruction geometry as a pyvista.PolyData object."""
        if self.reconstruction is not None:
            return pv.wrap(self.reconstruction)
        else:
            return None

    @property
    def _pc_id_strs(self) -> Tuple[str, str]:
        """Return internal label strings for extraction operations."""
        pid_str = "_".join((self.label, "pids"))
        cid_str = "_".join((self.label, "cids"))
        return pid_str, cid_str

    @without_activation
    def _add_persistent_pc_id_data(self) -> None:
        """Create point and cell ID arrays that persist through multiple operations."""
        pid_str, cid_str = self._pc_id_strs
        self.point_data[pid_str] = np.arange(self.n_points)
        self.cell_data[cid_str] = np.arange(self.n_cells)

    @stage("Repairing degenerate cells")
    def repair_degenerate_cells(self) -> Self:
        """Repair degenerate elements found at the edges of the cartilage mesh.

        Degenerate cells occur when fewer than 8 points are defined within a hexahedral
        cell. This often occurs following a grid-based raytracing pass along the edges
        of cartilage geometry.

        Before the repair, the mesh will appear as separate surfaces. After repairing,
        the surfaces will be stitched together.
        """
        cell_def = self.cell_def
        points = self.point_locs
        points, cell_def = hc.squeeze_hex_mesh(points, cell_def.T)
        self.points = points.copy()
        return self.update_cell_def(cell_def=cell_def, points=points)

    @classmethod
    @stage("Smoothing hexahedral cells")
    def smooth_hexahedral_cells(
        cls,
        points: ArrayLike,
        cell_def: ArrayLike,
        smoothing_operation: str,
        flag: bool = False,
    ) -> np.ndarray:
        return hc.smooth_hex_mesh(
            elems=cell_def,
            nodes=points,
            smoothing_operation=smoothing_operation,
            flag=flag,
        )

    @stage("Optimizing scaled Jacobians")
    def optimize_scaled_jacobians(
        self,
        target_jacobian: float = 0,
        allow_jacobian_break: bool = True,
        in_plane_only: bool = False,
        inplace: bool = False,
    ) -> Self:
        """Relax skewed cells until all are above a target scaled jacobian value.

        Args:
            target_jacobian: The minimum scaled Jacobian allowed. Maximum is 1.0
                Defaults to 0.
            allow_jacobian_break: Deprecated. Defaults to True.
            inplace: Whether to update this CartilageMesh in place. Defaults to False.

        Returns:
            An update CartilageMesh object.

        """
        # Compute the neighbors of each vertex
        # Hexa is made of 2D quads, each traced counterclockwise to make positive faces
        new_mesh = self if inplace else self.copy()
        points = new_mesh.point_locs
        cell_def = new_mesh.cell_def
        if not in_plane_only:
            neighbors = np.vstack(
                (
                    cell_def[:, 0:4].reshape(-1, 2),  # top
                    cell_def[:, np.array((3, 0))],
                    cell_def[:, 4:8].reshape(-1, 2),  # bottom
                    cell_def[:, np.array((7, 4))],
                    cell_def[:, np.array((1, 5, 5, 6, 6, 2, 2, 1))].reshape(
                        -1, 2
                    ),  # right
                    cell_def[:, np.array((0, 4, 4, 7, 7, 3, 3, 0))].reshape(
                        -1, 2
                    ),  # left
                    cell_def[:, np.array((0, 4, 4, 5, 5, 1, 1, 0))].reshape(
                        -1, 2
                    ),  # front
                    cell_def[:, np.array((3, 7, 7, 6, 6, 2, 2, 3))].reshape(
                        -1, 2
                    ),  # back
                )
            )
        else:
            neighbors = np.vstack(
                (
                    cell_def[:, 0:4].reshape(-1, 2),  # top
                    cell_def[:, np.array((3, 0))],
                    cell_def[:, 4:8].reshape(-1, 2),  # bottom
                    cell_def[:, np.array((7, 4))],
                )
            )
        neighbors = np.unique(np.vstack((neighbors, np.fliplr(neighbors))), axis=0)

        needs_optimization = True
        loop_counter = 0
        while needs_optimization:
            scaled = new_mesh.scaled_jacobian
            jacobian = new_mesh.scaled_jacobian_determinants
            is_suboptimal = jacobian < target_jacobian
            indexes = np.unique(cell_def[is_suboptimal])
            if indexes.size == 0:
                needs_optimization = False

            else:
                # For each index substitute the node by the average of it's neighbors
                disp = np.zeros_like(points)
                if target_jacobian == 0:
                    for idx in indexes:
                        is_match = neighbors[:, 0] == idx
                        i_neighbor = neighbors[is_match, 1]
                        disp[idx, :] = points[i_neighbor].mean(axis=0) - points[idx, :]

                else:
                    suboptimal_idx = np.flatnonzero(scaled < target_jacobian)
                    for idx in suboptimal_idx:
                        suboptimal_elems = cell_def[idx]
                        disp = disp + hc.relax_skewed_elements(points, suboptimal_elems)

                    if np.linalg.norm(disp.sum(axis=0)) == 0:
                        needs_optimization = False

                shifted_points = points + disp
                scaled = UnstructuredGrid.from_cell_def(
                    points=shifted_points, cell_def=cell_def
                ).get_cell_quality("scaled_jacobian")
                if ~np.any(scaled < 0) or not allow_jacobian_break:
                    points = shifted_points
                    new_mesh.points = points.copy()
                else:
                    needs_optimization = False
            loop_counter += 1
            report("Optimizing scaled Jacobians", loop_counter, unit="iterations")
            try:
                if loop_counter > 100:
                    raise OptimizationError(
                        f"Optimizing scaled Jacobian attempts exceeded {100}."
                    )
            except OptimizationError:
                del new_mesh
                raise

        return new_mesh

    @without_activation
    def get_cell_quality(
        self,
        quality_measure: str = "scaled_jacobian",
        save_cell_data: bool = False,
        n_layers_or_slice_locs: Optional[int] = None,
    ) -> np.ndarray:
        """Calculate cell quality values.

        Unsupported cell types will have an entry of -1.

        Args:
            quality_measure: String representing quality measure. Defaults to
                "scaled_jacobian".
            save_cell_data: Whether to save the quality measure to this object's cell
                data. If True, the cell_data name will be set to `quality_measure`.
                Defaults to False.
            n_depth_layer: final number of layers defining cartilage thickness.

        Returns:
            The cell quality array.

        """
        # FIXME: This if-statement relies on a CartilageMesh method, but is called from
        # TissueMesh. Should eliminate the if() part, and the n_layers_or_slice_locs
        # parameter at this stage, then rewrite the CartilageMesh method to accept it.
        if n_layers_or_slice_locs is not None:
            grid_copy = self.subdivide_depth(
                n_layers_or_slice_locs=n_layers_or_slice_locs
            ).cell_quality(quality_measure)
        else:
            grid_copy = self.cell_quality(quality_measure)
        quality = _by_value(grid_copy[quality_measure])
        if save_cell_data:
            self.cell_data[quality_measure] = quality
        return quality

    @stage("Checking element quality")
    def find_low_quality_cells(
        self,
        metric_min: Union[float, int] = 0,
        metric_max: Union[float, int] = 1,
        *,
        n_layers_or_slice_locs: Optional[int] = None,
        quality_measure: str = "scaled_jacobian",
    ) -> np.ndarray:
        """Find low quality cells.

        Args:
            metric_min: The minimal acceptable quality value. Defaults to 0.
            metric_max: The maximum acceptable quality value. Defaults to 1.
            quality_measure: The quality measure. Defaults to "scaled_jacobian".

        Returns:
            Cell IDs with quality outside of the [metric_min, metric_max] quality range.

        """
        quality = self.get_cell_quality(
            quality_measure, n_layers_or_slice_locs=n_layers_or_slice_locs
        )
        is_bad_cell = np.full_like(quality, False, dtype=bool)
        is_bad_cell[quality < metric_min] = True
        is_bad_cell[quality > metric_max] = True
        return np.flatnonzero(is_bad_cell)

    def plot_cell_quality(
        self,
        *args,
        quality_measure: str = "scaled_jacobian",
        force_plot: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Plot cell quality measures.

        Extra arguments are passed to CartilageMesh.find_low_quality_cells method. If
        any low quality cells are found, the offending cells are emphasized.

        Args:
            quality_measure: The geometric cell quality measure. Defaults to
                "scaled_jacobian".
            force_plot: Show the plot, even if there are no bad cells. Defaults to
                False.

        Returns:
            The output of CartilageMesh.find_low_quality_cells.

        """
        bad_cids = self.find_low_quality_cells(*args, **kwargs)
        grid_copy = self.cell_quality(quality_measure)
        if grid_copy is None:
            raise CartilageMeshError(
                "Cannot plot cell quality. cell_quality returned None."
            )
        grid_copy[quality_measure] = grid_copy[quality_measure]
        grid_copy.set_active_scalars(quality_measure)

        opacity = 1 if not bad_cids.any() else 0.5
        p = pv.Plotter()
        p.add_mesh(grid_copy, show_edges=True, opacity=opacity)
        p.add_axes()
        if bad_cids.any():
            bad_jac = grid_copy.extract_cells(bad_cids)
            p.add_mesh(bad_jac)
            p.add_mesh(bad_jac.extract_all_edges(), color="y", opacity=0.9)
            p.show()
        if force_plot:
            p.show()
        return bad_cids

    @property
    def scaled_jacobian_determinants(self) -> np.ndarray:
        """Per cell scaled Jacobian determinants. scaled_jacobian == min(axis-1)."""
        points = self.point_locs
        cell_def = self.cell_def
        vertices = np.moveaxis(points[cell_def[:, :], :], 1, 0)
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
        return corner_node_jacobian_determinants

    @property
    def scaled_jacobian(self) -> np.ndarray:
        """Per cell scaled Jacobian mesh quality shortcut, returned as an array."""
        return self.get_cell_quality(quality_measure="scaled_jacobian")

    def _inplane_subdivide_along_axis(self, axis: int) -> Self:
        assert axis == 0 or axis == 1 or axis == 2, f"axis {axis} not in (0, 1, 2)."
        # TODO: Make this a package constant
        front_idx = np.array((0, 1, 5, 4)) if axis == 0 else np.array((1, 2, 6, 5))
        back_idx = np.array((3, 2, 6, 7)) if axis == 0 else np.array((0, 3, 7, 4))
        if axis == 2:
            front_idx = np.arange(8)[:4]
            back_idx = np.arange(8)[4:]
        cell_def = self.cell_def
        all_def = self.points[self.cells_dict[VTK_HEXAHEDRON]]
        front = all_def[:, front_idx, :]
        back = all_def[:, back_idx, :]

        subdivided = np.linspace(front, back, 3)  # shape (3, 3808, 4, 3)
        new_points = subdivided.reshape(-1, 3)
        new_point_ids = np.arange(new_points.shape[0]).reshape(3, -1)

        # Fill in the new cell_def
        n_cells = self.n_cells
        new_def = np.full_like(np.tile(cell_def, (2, 1)), -1)
        new_def[:n_cells, front_idx] = new_point_ids[0].reshape(-1, 4)
        new_def[:n_cells, back_idx] = new_point_ids[1].reshape(-1, 4)
        new_def[n_cells:, front_idx] = new_point_ids[1].reshape(-1, 4)
        new_def[n_cells:, back_idx] = new_point_ids[2].reshape(-1, 4)

        # Remove duplicate points
        _, unq_idx, inv_idx = np.unique(
            np.round(new_points, 3), axis=0, return_index=True, return_inverse=True
        )
        new_points = new_points[unq_idx]

        new_def = inv_idx[new_def]
        return self.update_cell_def(points=new_points, cell_def=new_def)


class CartilageMesh(TissueMesh):
    """UnstructuredGrid specialized for flat layers of hexahderal elements."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._is_depth_subdivided = False
        self.reverse_bone_joint_side = False
        self._bone_reconstruction_filename = None
        self._cached_bone_reconstruction = None
        self.tissue_type: TissueType = TissueType.CARTILAGE
        self.cartilage_type: CartilageType = CartilageType.ARTICULAR

        try:
            self._mark_bone_joint_interfaces()
        except CartilageMeshError:
            pass
        try:
            self._mark_perimeter_cells()
        except CartilageMeshError:
            pass  # uninitialized mesh

    @classmethod
    def from_inp_file(
        cls,
        filename: Union[Path, str],
        n_layers: int,
        reverse_bone_joint_side: bool = False,
    ) -> Self:
        """Create cartilage mesh from exported input file.

        If the mesh has been depthwise subdivided, it is important to use the correct
        number of layers for the n_layers parameter. Otherwise the mesh will not be
        correct.

        Args:
            filename: the abaqus INP filename containing the exported CartilageMesh.
            n_layers: the number of element layers spanning the depth of the cartilage
                from bone-side to joint-side.
            reverse_bone_joint_side: whether to reverse the bone and joint side of thed
                cartilage depth. Defaults to False.

        Returns:
            The cartilage mesh object.

        """
        mesh = pv.read(Path(filename), file_format="abaqus")
        if mesh is None:
            raise CartilageMeshError(f"Could not read {filename}.")
        cell_def = mesh.cells_dict[VTK_HEXAHEDRON]
        cartilage_mesh = cls()
        cartilage_mesh.reverse_bone_joint_side = reverse_bone_joint_side
        if n_layers > 1:
            cartilage_mesh._is_depth_subdivided = True
        cartilage_mesh = cartilage_mesh.update_cell_def(
            cell_def=cell_def, points=mesh.points
        )
        if n_layers > 1:
            cartilage_mesh.cell_data["layer_idx"] = np.arange(n_layers).repeat(
                cartilage_mesh.n_cells / n_layers
            )
            cartilage_mesh._mark_bone_joint_interfaces()
            cartilage_mesh.compute_p2p_cartilage_depths()
        return cartilage_mesh

    def update_cell_def(self, cell_def: ArrayLike, points: ArrayLike = None) -> Self:
        """Update the mesh with new point and cell definitions.

        Args:
            cell_def: an (n_cell, 8) array of point IDs where each line defines a single
                first-order hexahedral element within the cartilage mesh.
            points: an (n_points, 3) array of point locations. Defaults to None.

        Returns:
            A copy of this cartilage mesh with updated cell and point definitions.

        """
        points = self.point_locs if points is None else points
        new_mesh = self.__class__.from_cell_def(points=points, cell_def=cell_def)
        new_mesh.set_state(self.get_state())
        new_mesh._add_persistent_pc_id_data()
        try:
            new_mesh._mark_bone_joint_interfaces()
        except CartilageMeshError:
            pass
        try:
            new_mesh._mark_perimeter_cells()
        except CartilageMeshError:
            pass
        return new_mesh

    def plot_with_bone(self) -> None:
        """Plot the cartilage mesh with any attached mesh."""
        if self.bone_reconstruction is not None:
            p = self.plotter_with_mesh()
            p.add_mesh(pv.wrap(self.bone_reconstruction))
            p.show()

    def plot_orientation(self) -> None:
        """Plot with bone-side and joint-side nodes highlighted."""
        add_mesh_kwargs = self.add_mesh_kwargs(opacity=0.5)
        p = self.plotter_with_mesh(add_mesh_kwargs=add_mesh_kwargs)
        bone_side = self.extract_points(
            self.field_data["bone_side_pids"], include_cells=False
        )
        joint_side = self.extract_points(
            self.field_data["joint_side_pids"], include_cells=False
        )
        p.add_mesh(bone_side, color="r", label="bone_side")
        p.add_mesh(joint_side, color="b", label="joint_side")
        p.add_legend()
        p.show()

    def attach_bone_reconstruction(self, filename: Union[str, Path, PolyData]) -> None:
        """Attach a bone reconstruction filename for plotting and blending.

        A bone reconstruction is necessary for plotting the cartilage mesh with its
        bone, and is also used for blending the cartilage-to-bone interface.

        Args:
            filename: the filename containing the bone reconstructions. Typically an STL
                file.

        """
        if not isinstance(filename, str) or not isinstance(filename, Path):
            self.bone_reconstruction = filename
        else:
            self._bone_reconstruction_filename = filename

    @property
    def bone_reconstruction(self) -> Trimesh | PolyData:
        """Return bone reconstruction triangular surface mesh."""
        if self._cached_bone_reconstruction is not None:
            return self._cached_bone_reconstruction
        elif self._bone_reconstruction_filename is not None:
            return clamp_triangle_count_and_smooth(
                filename=self._bone_reconstruction_filename,
                n_triangles=8000,  # reasonable number of triangles
                smooth_iters=1,  # single smoothing pass after decimating triangles
            )
        else:
            return None

    @bone_reconstruction.setter
    def bone_reconstruction(self, reconstruction: Trimesh | PolyData) -> None:
        self._bone_reconstruction_filename = None
        self._cached_bone_reconstruction = reconstruction

    @without_activation
    def _mark_perimeter_cells(self) -> None:
        """Create field data for perimeter cells in a single layer mesh."""
        if self._is_depth_subdivided:
            raise CartilageMeshError(
                "Cannot update perimeter cells or points after depth subdivision."
            )
        elif self.cells_dict in [None, {}]:
            return
        surf = self.extract_surface(algorithm=None)
        if surf is None:
            raise CartilageMeshError("Cannot extract surface from empty mesh.")
        surf = surf.copy()
        cids, face_counts = np.unique(surf["vtkOriginalCellIds"], return_counts=True)
        perim_cell_ids = cids[face_counts > 2]
        self.add_field_data(perim_cell_ids, "perimeter_cids")

    @property
    def perimeter_cells(self) -> UnstructuredGrid:
        """Return a copy of perimeter cells."""
        if self._check_array_names("perimeter_cids"):
            pid_str, cid_str = self._pc_id_strs
            extracted_cells = self.extract_cells(self.field_data["perimeter_cids"])
            if extracted_cells is None:
                raise CartilageMeshError("No perimeter cells found.")
            perim_cells = extracted_cells.copy()
            perim_cells.cell_data[cid_str] = perim_cells["vtkOriginalCellIds"].copy()
            perim_cells.point_data[pid_str] = perim_cells["vtkOriginalPointIds"].copy()
            return perim_cells

    @without_activation
    def _mark_perimeter_edges(self) -> None:
        """Add field_data arrays marking laminar edges of the Cartilage mesh.

        Creates "boundary_edge_pids", "joint_side_boundary_edge_pids", and
        "bone_side_boundary_edge_pids".
        """
        orig_pid_key, orig_cid_key = self._pc_id_strs

        # Extract surface
        single_layer_grid = self.copy()
        surf = single_layer_grid.extract_surface(algorithm=None)
        if surf is None:
            raise CartilageMeshError("Cannot extract surface from empty mesh.")
        surf.cell_data[orig_cid_key] = surf["vtkOriginalCellIds"]
        surf.point_data[orig_pid_key] = surf["vtkOriginalPointIds"]

        if not self._check_array_names(
            "bone_side_pids", "joint_side_pids", array_type="field"
        ):
            self._mark_bone_joint_interfaces()

        bone_side_pids = self.field_data["bone_side_pids"]
        joint_side_pids = self.field_data["joint_side_pids"]
        face_pids = surf.faces.reshape(-1, 5)[:, 1:]
        orig_face_pids = surf.point_data[orig_pid_key][face_pids]
        is_bone_side_cell = np.isin(orig_face_pids, bone_side_pids).all(axis=1)
        is_cart_side_cell = np.isin(orig_face_pids, joint_side_pids).all(axis=1)
        # The following line casts the surface to UnstructuredGrid, and is important for
        # point ID matching down the chain.
        boundary_surf_faces = surf.extract_cells(np.arange(surf.n_cells))  # important!
        boundary_surf_faces.remove_cells(
            (is_bone_side_cell | is_cart_side_cell), inplace=True
        )
        self.field_data["boundary_edge_pids"] = boundary_surf_faces.point_data[
            orig_pid_key
        ]
        intersect_pids = np.intersect1d(
            self.field_data["boundary_edge_pids"], joint_side_pids
        )
        self.field_data["joint_side_boundary_edge_pids"] = intersect_pids
        intersect_pids = np.intersect1d(
            self.field_data["boundary_edge_pids"], bone_side_pids
        )
        self.field_data["bone_side_boundary_edge_pids"] = intersect_pids
        del surf, bone_side_pids, joint_side_pids, single_layer_grid, face_pids

    @property
    def is_interior_point_mask(self) -> np.ndarray:
        """Return boolean (n_points, ) array marking interior points."""
        if not self._check_array_names("bone_side_boundary_edge_pids"):
            self._mark_perimeter_edges()
            self._mark_perimeter_cells()
        is_int_pt = np.full(self.n_points, True)
        is_int_pt[self.field_data["joint_side_boundary_edge_pids"]] = False
        is_int_pt[self.field_data["bone_side_boundary_edge_pids"]] = False
        return is_int_pt

    @property
    def is_joint_side_edge_mask(self) -> np.ndarray:
        """Return boolean (n_points, ) array marking joint-side-edge points."""
        if not self._check_array_names("joint_side_boundary_edge_pids"):
            self._mark_perimeter_edges()
            self._mark_perimeter_cells()
        is_js_edge_pt = np.full(self.n_points, False)
        is_js_edge_pt[self.field_data["joint_side_boundary_edge_pids"]] = True
        return is_js_edge_pt

    @property
    def is_bone_side_edge_mask(self) -> np.ndarray:
        """Return boolean (n_points, ) array marking bone-side-edge points."""
        if not self._check_array_names("bone_side_boundary_edge_pids"):
            self._mark_perimeter_edges()
            self._mark_perimeter_cells()
        is_bs_edge_pt = np.full(self.n_points, False)
        is_bs_edge_pt[self.field_data["bone_side_boundary_edge_pids"]] = True
        return is_bs_edge_pt

    @property
    def perimeter_edges(self) -> UnstructuredGrid:
        """Return (boundary surface edges, boundary surface faces)."""
        if not self._check_array_names("bondary_edge_pids", array_type="field"):
            self._mark_perimeter_edges()

        self.point_data["cmpe_is_joint_side"] = self.is_joint_side_point_mask
        self.point_data["cmpe_is_bone_side"] = self.is_bone_side_point_mask
        orig_pid_key, orig_cid_key = self._pc_id_strs
        surf = self.copy().extract_surface(algorithm=None)

        boundary_surf_faces = surf.extract_points(
            np.isin(surf[orig_pid_key], self.field_data["boundary_edge_pids"]),
            adjacent_cells=False,
        )
        # Double check that we didn't get any tangential faces
        cell_def = boundary_surf_faces.cells_dict[VTK_QUAD]
        all_js = np.isin(
            cell_def,
            np.flatnonzero(boundary_surf_faces.point_data["cmpe_is_joint_side"]),
        ).all(axis=1)
        all_bs = np.isin(
            cell_def,
            np.flatnonzero(boundary_surf_faces.point_data["cmpe_is_bone_side"]),
        ).all(axis=1)
        boundary_surf_faces.remove_cells(all_bs | all_js, inplace=True)

        boundary_surf_edges = boundary_surf_faces.copy().extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=True,
        )
        boundary_surf_edges["vtkOriginalPointIds"] = boundary_surf_edges.point_data[
            orig_pid_key
        ]
        boundary_surf_edges["vtkOriginalCellIds"] = boundary_surf_edges.cell_data[
            orig_cid_key
        ]

        boundary_surf_faces["vtkOriginalPointIds"] = boundary_surf_faces.point_data[
            orig_pid_key
        ]
        boundary_surf_faces["vtkOriginalCellIds"] = boundary_surf_faces.cell_data[
            orig_cid_key
        ]
        del surf
        [self.point_data.remove(x) for x in self.point_data.keys() if "cmpe_" in x]
        return boundary_surf_edges, boundary_surf_faces

    def _mark_bone_joint_interfaces(self) -> None:
        """Create field data arrays of bone- and joint-side point IDs.

        Raises:
            CartilageMeshError: when the cartilage mesh does not have a cell definition.
                This should be caught and handled by other functions.

        """
        if self._is_depth_subdivided and self._check_array_names(
            "layer_idx", array_type="cell"
        ):
            n_layers = self.cell_data["layer_idx"].max() + 1
            is_bone_side_cell = self.cell_data["layer_idx"] == 0
            is_joint_side_cell = self.cell_data["layer_idx"] == (n_layers - 1)
            if not self.reverse_bone_joint_side:
                js_idxs = np.unique(
                    self.cells_dict[VTK_HEXAHEDRON][is_joint_side_cell, 4:]
                )
                bs_idxs = np.unique(
                    self.cells_dict[VTK_HEXAHEDRON][is_bone_side_cell, :4]
                )
            else:
                bs_idxs = np.unique(
                    self.cells_dict[VTK_HEXAHEDRON][is_joint_side_cell, 4:]
                )
                js_idxs = np.unique(
                    self.cells_dict[VTK_HEXAHEDRON][is_bone_side_cell, :4]
                )
            js_idxs = np.flatnonzero(np.isin(self.cell_def, js_idxs))
            bs_idxs = np.flatnonzero(np.isin(self.cell_def, bs_idxs))
            js_pids = self.cell_def.flat[js_idxs]
            bs_pids = self.cell_def.flat[bs_idxs]

            _, idxs = np.unique(js_pids, return_index=True)
            unique_js_pids = js_pids[np.sort(idxs)]
            _, idxs = np.unique(bs_pids, return_index=True)
            unique_bs_pids = bs_pids[np.sort(idxs)]

            self.add_field_data(unique_bs_pids, "bone_side_pids")
            self.add_field_data(unique_js_pids, "joint_side_pids")

        elif self.cells_dict not in [None, {}]:
            if not self.reverse_bone_joint_side:
                js_idxs = np.unique(self.cells_dict[VTK_HEXAHEDRON][:, 4:])
                bs_idxs = np.unique(self.cells_dict[VTK_HEXAHEDRON][:, :4])
            else:
                bs_idxs = np.unique(self.cells_dict[VTK_HEXAHEDRON][:, 4:])
                js_idxs = np.unique(self.cells_dict[VTK_HEXAHEDRON][:, :4])
            js_idxs = np.flatnonzero(np.isin(self.cell_def, js_idxs))
            bs_idxs = np.flatnonzero(np.isin(self.cell_def, bs_idxs))
            js_pids = self.cell_def.flat[js_idxs]
            bs_pids = self.cell_def.flat[bs_idxs]

            _, idxs = np.unique(js_pids, return_index=True)
            unique_js_pids = js_pids[np.sort(idxs)]
            _, idxs = np.unique(bs_pids, return_index=True)
            unique_bs_pids = bs_pids[np.sort(idxs)]

            self.add_field_data(unique_bs_pids, "bone_side_pids")
            self.add_field_data(unique_js_pids, "joint_side_pids")
        else:
            raise CartilageMeshError()

    @property
    def is_joint_side_point_mask(self) -> np.ndarray:
        """Return boolean (n_points, ) array marking joint-side points."""
        if self._check_array_names("joint_side_pids", array_type="field"):
            is_js_pt = np.full(self.n_points, False)
            is_js_pt[self.field_data["joint_side_pids"]] = True
            return is_js_pt
        else:
            raise CartilageMeshError("No joint-side point IDs found.")

    @property
    def is_bone_side_point_mask(self):
        """Return boolean (n_points, ) array marking bone-side points."""
        if self._check_array_names("bone_side_pids", array_type="field"):
            is_bs_pt = np.full(self.n_points, False)
            is_bs_pt[self.field_data["bone_side_pids"]] = True
            return is_bs_pt
        else:
            raise CartilageMeshError("No bone-side point IDs found.")

    @without_activation
    @stage("Fusing to bone")
    def fuse_to_bone(
        self,
        only_edge: bool = False,
        match_edge_translation: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Move joint-side points to nearest intersection of bone reconstruction.

        Args:
            only_edge: Only perform this operation on the edgemost points. Defaults to
                False.
            match_edge_translation: For the cartilage edges, shift the bone and joint
                edge by an equal amount. Defaults to False.
            dry_run: Instead of moving points, return a point_data array of
            displacements. Defaults to False.

        Raises:
            CartilageMeshError: when there is no attached bone reconstruction.

        """
        if self.bone_reconstruction is None:
            raise CartilageMeshError("No bone reconstruction is attached.")
        bone = pv.wrap(self.bone_reconstruction)
        _, closest_bone_point_locs = bone.find_closest_cell(
            self.points, return_closest_point=True
        )
        deltas = closest_bone_point_locs - self.point_locs
        deltas[self.is_joint_side_point_mask] = np.zeros(3)
        if only_edge:
            deltas[self.is_interior_point_mask] = np.zeros(3)

        if match_edge_translation:
            deltas[self.joint_interface_ids] = deltas[self.bone_interface_ids]

        self.point_data["tmp_deltas"] = deltas

        if dry_run:
            del self.point_data["tmp_deltas"]
            return deltas
        else:
            self.points += self.point_data["tmp_deltas"]
            del self.point_data["tmp_deltas"]

    @without_activation
    @stage("Expanding laminar edges")
    def expand_laminar_edges(
        self,
        max_node_shift: Union[int, float] = None,
        include_centers: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> Self:
        """Expand the edges of the cartilage mesh.

        During mesh optimization, the cartilage mesh repeatedly undergoes Laplacian
        smoothing. This smoothing operation tends to shrink portions of the mesh with
        surface features (large changes in surface angle). Since the edge of the
        cartilage is made up of right angles, it shrinks the most. To correct for this,
        we can translate the nodes defining the edges until they intersect the
        reconstruction geometry once more.

        The direction of point translations are calculated by extracting only the
        perimeter edges which form ribbons of quadrilateral cells, and calculating the
        resulting point normals. To maintain element quality, bone-side and joint-side
        nodes connected by an edge are moved in pairs, using the maximum raytraced
        distance.

        Since the cartilage reconstruction is likely to extend the furthest at the
        midpoint of its depth, we can introduce additional rays at the center of the
        bone- joint-side pairs when determining the magnitude of point displacements.
        Points can be restricted from moving too far during this process.

        Args:
            max_node_shift: maximum distance to translate a point. Defaults to None.
            include_centers: include rays at the midpoint of the bone- joint-side
                surfaces. Defaults to True.
            inplace: if True, update the cartilage mesh in place, return an updated copy
                otherwise. Defaults to False.

        Returns:
            The cartilage mesh with edges translated to intersect the reconstruction.

        """
        pid_str, cid_str = self._pc_id_strs
        reconstruction = self.reconstruction
        surf = self.extract_surface(algorithm=None)
        avg_surf_cell_area = surf.area / surf.n_cells
        avg_side_len = np.sqrt(avg_surf_cell_area)
        del surf
        max_node_shift = avg_side_len * 2 if max_node_shift is None else max_node_shift

        # Get border elements
        self.point_data["is_joint_side"] = self.is_joint_side_point_mask
        self.point_data["is_bone_side"] = self.is_bone_side_point_mask
        perim_edges, perim_ribbon = self.perimeter_edges  # noqa

        # Add face  and point rays
        perim_tri_mesh = perim_ribbon.extract_surface(algorithm=None)
        perim_tri_mesh.compute_normals(inplace=True)
        point_normals = perim_tri_mesh.point_data["Normals"]
        face_normals = perim_tri_mesh.cell_data["Normals"]
        perim_ribbon.cell_data["Normals"] = np.full_like(face_normals, np.nan)
        perim_ribbon.point_data["Normals"] = np.full_like(point_normals, np.nan)
        perim_ribbon.cell_data["Normals"][perim_tri_mesh["vtkOriginalCellIds"]] = (
            face_normals
        )
        perim_ribbon.point_data["Normals"][perim_tri_mesh["vtkOriginalPointIds"]] = (
            point_normals
        )

        # Add center origins and directions
        depth_lines = perim_ribbon.extract_feature_edges(
            feature_edges=False,
            manifold_edges=True,
            non_manifold_edges=False,
            boundary_edges=False,
        )
        centers = depth_lines.cell_centers()
        centers["norms"] = depth_lines.point_data["Normals"][::2]
        perim_ribbon.set_active_vectors("Normals")

        depth_lines["is_joint_side"] = depth_lines["is_joint_side"].astype(bool)
        depth_lines["is_bone_side"] = depth_lines["is_bone_side"].astype(bool)

        line_pair_ids = depth_lines.lines.reshape(-1, 3)[:, 1:].copy()
        is_bone_side_edge = np.isin(
            line_pair_ids, np.flatnonzero(depth_lines["is_bone_side"])
        )
        needs_flip = is_bone_side_edge[:, 0]
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
            distances.reshape(n_lines, -1, order="F").max(axis=1),
            max_distances.shape[1],
        )
        displacement_vectors = max_distances[..., np.newaxis] * directions

        depth_lines.point_data["displacements"] = depth_lines.point_data[
            "Normals"
        ].copy()
        depth_lines.point_data["displacements"][line_pair_ids[:, 0]] = (
            displacement_vectors[:n_lines]
        )
        depth_lines.point_data["displacements"][line_pair_ids[:, 1]] = (
            displacement_vectors[n_lines : 2 * n_lines]
        )
        depth_lines.set_active_vectors("displacements")
        self.point_data["displacements"] = np.zeros((self.n_points, 3), dtype=float)
        self.point_data["displacements"][depth_lines[pid_str]] = depth_lines.point_data[
            "displacements"
        ]

        disp_vectors = self.smooth_hexahedral_cells(
            points=self.point_data["displacements"],
            cell_def=self.cell_def,
            smoothing_operation="surface",
            flag=False,
        )

        if kwargs.get("debug", False):
            self.set_active_vectors("displacements", preference="point")
            p = self.plotter_with_mesh(
                add_mesh_kwargs=self.add_mesh_kwargs(color="tan", opacity=0.8)
            )
            p.add_mesh(self.arrows, color="r")
            p.add_mesh(self.wrapped_reconstruction, color="grey", opacity=0.4)
            p.show()
            del p

            p = pv.Plotter()
            perim_ribbon.set_active_vectors("Normals", "point")
            point_arrows = perim_ribbon.arrows
            centers.set_active_vectors("norms", "point")
            p.add_mesh(self, opacity=0.4, color="tan", show_edges=False)
            p.add_mesh(centers.arrows, color="y")
            p.add_mesh(perim_ribbon.arrows, color="r")
            p.add_mesh(pv.PolyData(depth_lines.points[line_pair_ids[:, 0]]), color="b")
            p.add_mesh(pv.PolyData(depth_lines.points[line_pair_ids[:, 1]]), color="g")
            p.add_mesh(perim_ribbon, color="tan", opacity=1, show_edges=True)
            p.add_axes()
            p.show()
            del p

        del (
            # perim_ribbon ,
            # perim_edges,
            # face_normals,
            # depth_lines,
            # perim_tri_mesh,
            self.point_data["is_joint_side"],
            self.point_data["is_bone_side"],
            self.point_data["displacements"],
        )

        if inplace:
            self.points += disp_vectors
            return self
        else:
            new_mesh = self.copy()
            new_mesh.points += disp_vectors
            return new_mesh

    def optimize_iteratively(
        self,
        max_iters: int = 15,
        include_expand: bool = True,
        max_point_shift: float = 2,
        upper_jacobian_limit: float = 0.5,
        include_smooth: bool = True,
        smoothing_operation: str = "surface",
        inplace: bool = False,
    ) -> Self:
        """Optimize a CartilageMesh iteratively, until a minimum scaled Jacobian is met.

        Args:
            max_iters: The maximum number of iterations allowed. Most meshes will finish
                within 5. Defaults to 20.
            include_expand: Include a call to expand_laminar_edges during the
                optimization. Should be included if smoothing. Defaults to True.
            max_point_shift: Maximum distance a point may be shifted during expansion.
                Meaningless if inclde_expand is False. Defaults to 2.
            upper_jacobian_limit: The minimum scaled Jacobian target. Optimization
                proceeds until every cell greater than or equal to this value. Defaults
                to 0.5.
            include_smooth: Include a mesh smoothing step in each iteration. To combat
                mesh shrinkage, set include_expand to True. Defaults to True.
            smoothing_operation: The type of mesh smoothing performed.
                Defaults to "surface".
            inplace: Whether the mesh should be updated in place, or if a copy should be
                returned. Defaults to False.

        Raises:
            OptimizationError: when optimization loop has been performed max_iters
                times.

        Returns:
            The optimized mesh.

        """
        new_mesh = self if inplace else self.copy()
        # TODO: Replace this with a method call
        s = new_mesh.scaled_jacobian
        is_bad_jacobian = s < upper_jacobian_limit
        loop_counter = 0
        with WorkProgress(desc="Optimizing mesh") as pbar:
            while loop_counter < max_iters and is_bad_jacobian.any():
                if include_smooth:
                    new_mesh.points = new_mesh.smooth_hexahedral_cells(
                        points=new_mesh.points,
                        cell_def=new_mesh.cell_def,
                        smoothing_operation=smoothing_operation,
                    )
                if include_expand:
                    new_mesh.expand_laminar_edges(
                        max_node_shift=max_point_shift, inplace=True
                    )
                new_mesh.optimize_scaled_jacobians(
                    target_jacobian=upper_jacobian_limit, inplace=True
                )
                # TODO: Replace this with a method call
                s = new_mesh.scaled_jacobian
                is_bad_jacobian = s < upper_jacobian_limit
                loop_counter += 1
                pbar.update(1)

        try:
            if loop_counter >= max_iters:
                raise OptimizationError(
                    f"Mesh optimization did not finish after {max_iters} iterations."
                )
            return new_mesh
        except OptimizationError:
            del new_mesh
            raise

    @stage("Subdividing in plane")
    def subdivide_in_plane(self) -> Self:
        """Return CatilageMesh with each element split into four in-plane elements.

        > Note: Does not affect cartilage depth, and should be performed before
            cartilage depth subdivision.

        Returns:
            The updated mesh.

        """

        # TODO: Needs to account for selecting edges and interface pids
        new_mesh = self.copy()
        new_mesh = new_mesh._inplane_subdivide_along_axis(axis=0)
        new_mesh = new_mesh._inplane_subdivide_along_axis(axis=1)
        return new_mesh

    @without_activation
    @stage("Subdividing depth")
    def subdivide_depth(self, n_layers_or_slice_locs: int = 4) -> Self:
        """Subdivides the cartilage mesh along the depth axis.

        Args:
            n_layers_or_slice_locs (int or array-like): The number of layers to
                subdivide the mesh into. If an integer is provided, the mesh will be
                divided into that many equally spaced layers.  If an array-like object
                is provided, the mesh will be divided into layers at the specified slice
                locations.

        Example:
            # divide into 4 evenly spaced element layers
            >>> CartilageMesh.subdivide_depth(4) # divide into 4 layers

            # divide into 4 layers delimited at 0%, 10%, 50%, 90%, and 100$ of the depth
            >>> CartilageMesh.subdivide_depth([0.1, 0.5, 0.9]) # divide into 4 layers

        Returns:
            A new cartilage mesh object with the specified depth subdivision.

        """
        if self._is_depth_subdivided:
            raise CartilageMeshError("Requires single layer mesh. Already subdivided.")
        if n_layers_or_slice_locs == 1:
            return self.copy()

        # subdivide the single cell into four cells
        points = np.moveaxis(self.point_locs[self.cell_def], 1, 0)
        if isinstance(n_layers_or_slice_locs, int):
            points = np.linspace(
                points[:4], points[4:], n_layers_or_slice_locs + 1
            )  # HexElem topology
            n_element_layers = n_layers_or_slice_locs
        elif isinstance(n_layers_or_slice_locs, (list, tuple, np.ndarray)):
            n_layers_or_slice_locs = np.array(n_layers_or_slice_locs)
            n_layers_or_slice_locs = np.hstack((0, n_layers_or_slice_locs, 1))
            starts, stops = points[:4], points[4:]
            vecs = stops - starts
            magnitudes = trimesh.util.row_norm(vecs.reshape(-1, 3))
            step_sizes = np.atleast_2d(magnitudes).T * n_layers_or_slice_locs
            unit_vecs = trimesh.util.unitize(vecs.reshape(-1, 3)).reshape(vecs.shape)
            points = np.zeros((n_layers_or_slice_locs.size, *starts.shape))
            for step_idx, step in enumerate(step_sizes.T):
                points[step_idx] = (
                    starts + np.atleast_3d(step.reshape(vecs.shape[:2])) * unit_vecs
                )
            # Adjust parameter to match integer version for downstream element splitting
            n_element_layers = n_layers_or_slice_locs.size - 1

        else:
            raise CartilageMeshError(
                "n_layers_or_slice_locs must be an integer or array-like."
            )

        # Create the layer indexes before updating self
        layer_idx = np.hstack(
            [x * np.ones(self.n_cells, dtype=int) for x in range(n_element_layers)]
        ).astype(int)
        # assign IDs to all the points in the cloud, and reshape the cloud for indexing
        point_cloud_dims = points.shape[:-1]
        node_ids = np.arange(np.prod(point_cloud_dims)).reshape(point_cloud_dims)
        new_node_locs = points.reshape(-1, 3)

        # split the cloud of point IDs into element pairs
        new_elems = np.hstack(
            [np.vstack(node_ids[a : a + 2]) for a in np.arange(n_element_layers)]
        ).T

        # remove coincident nodes, and adjust the element definitions
        new_node_locs, inverse_idx = np.unique(
            new_node_locs.round(2), axis=0, return_inverse=True
        )
        new_elems = inverse_idx[new_elems]
        new_mesh = self.copy()
        new_mesh._is_depth_subdivided = True
        new_mesh = new_mesh.update_cell_def(cell_def=new_elems, points=new_node_locs)
        new_mesh.cell_data["layer_idx"] = layer_idx

        # Add field data for each layer
        for idx in range(n_element_layers):
            new_mesh.add_field_data(np.flatnonzero(layer_idx == idx), f"layer_{idx}")

        # Actually mark the new cells
        new_mesh._mark_bone_joint_interfaces()
        new_mesh._mark_perimeter_edges()

        return new_mesh

    @property
    def bone_interface_ids(self) -> np.ndarray:
        """Bone-side surface point IDs."""
        if self._check_array_names("bone_side_pids", array_type="field"):
            return self.field_data["bone_side_pids"]
        else:
            raise CartilageMeshError("No bone-side point IDs found.")

    @property
    def joint_interface_ids(self) -> np.ndarray:
        """Joint-side surface point IDs."""
        if self._check_array_names("joint_side_pids", array_type="field"):
            return self.field_data["joint_side_pids"]
        else:
            raise CartilageMeshError("No joint-side point IDs found.")

    def _check_array_names(self, *names: str, array_type: str = "field") -> bool:
        """Check if all array names are present in this object.

        Args:
            array_type: the type of array to check. Can be {'cell', 'point', 'field'}.
                Defaults to "field".

        Raises:
            CartilageMeshError: if the array_type is not known.

        Returns:
            True if all arrays are present, False, otherwise.

        """
        at = array_type.lower()
        if at == "field":
            keys = self.field_data.keys()
        elif at == "cell":
            keys = self.cell_data.keys()
        elif at == "point":
            keys = self.point_data.keys()
        else:
            raise CartilageMeshError(
                f"Unknown array_type {array_type}. Choices are ('cell', 'point', 'field'."
            )
        has_names = [None] * len(names)
        for idx, name in enumerate(names):
            has_names[idx] = True if name in keys else False

        return all(has_names)

    @property
    def p2p_cartilage_depths(self) -> np.ndarray:
        """Joint-side to bone-side point-to-point cartilage depths in an array."""
        if self._check_array_names("joint_side_pids", "bone_side_pids"):
            js_locs = self.points[self.field_data["joint_side_pids"]]
            bs_locs = self.points[self.field_data["bone_side_pids"]]
            return np.linalg.norm(js_locs - bs_locs, axis=1)
        else:
            raise CartilageMeshError("No joint-side or bone-side point IDs found.")

    @without_activation
    def compute_p2p_cartilage_depths(self):
        """Calculate point-to-point cartilage depths.

        The cartilage depths will populate a point_data array, with the intermediate
        points set to zero, while the joint/bone side points will contain identical
        depth.
        """
        depths = np.full(self.n_points, 0.0)
        depths[self.field_data["joint_side_pids"]] = self.p2p_cartilage_depths
        depths[self.field_data["bone_side_pids"]] = self.p2p_cartilage_depths
        self.point_data["p2p_depth"] = depths

    # @deprecated
    def extract_perimeter_edges(self):
        pass

    # @deprecated
    def extract_perimeter_cells(self):
        pass

    # @with_debug_kwarg
    @without_activation
    @stage("Blending bone interface")
    def blend_to_bone_interface(
        self,
        grade: float = 1,
        weight_vectors_towards_bone: float = 0.2,
        bone_to_joint_displacement_weights: Tuple[float, float] = (1, 0),
        feature_angle_range: Tuple[float, float] = (25, 30),
        include_bone_fusion: bool = True,
        n_layers_or_slice_locs: Optional[int] = None,
        dry_run: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Blend the cartilage mesh with the bone.

        Treating the mesh as a single layer of cubes leads to sharp edges at the laminar
        edges of the mesh, which result in contacting cartilage meshes catching on each
        other. We can shift the edge nodes to create an incline at the edges of the
        mesh.

        Args:
            grade: the target slope for the cartilage edges as a percent of depth.
                Defaults to 1.
            weight_vectors_towards_bone: weight bone-side displacement vectors. Defaults
                to 0.2.
            bone_to_joint_displacement_weights: a tuple of values representing the
                amount of displacement for the bone-side and joint-side edges,
                respectively. Defaults to (1, 0).
            feature_angle_range: a minimum and maximum feature angle denoting too
                smooth, or too sharp for the blended edges. Defaults to (25, 30).
            include_bone_fusion: fuse the bone-side cartilage edges to the nearest bone
                surface before smoothing the blend. Defaults to True.
            dry_run: return the proposed displacements as a point_data array, instead of
                moving the points in place. Defaults to False.

        Raises:
            CartilageMeshError: if the mesh has already been subdivide in depth.

        Returns:
            The proposed displacements in a point_data array, if dry_run is True.

        """
        min_feature_angle, max_feature_angle = feature_angle_range
        if self._is_depth_subdivided:
            raise CartilageMeshError("Cannot blend after depth subdivision.")
        init_point_locs = self.point_locs

        # Calculate vector directions
        self._compute_edge_normals_for_blending(
            weight_towards_bone=weight_vectors_towards_bone
        )

        # Calculate distance scalers
        self.compute_p2p_cartilage_depths()
        scalers = self.point_data["p2p_depth"] * grade
        quantile = np.quantile(scalers, 0.85)
        scalers[scalers > quantile] = quantile
        scalers[self.is_interior_point_mask] = 0

        scalers[self.is_bone_side_edge_mask] *= bone_to_joint_displacement_weights[0]
        scalers[self.is_joint_side_edge_mask] *= bone_to_joint_displacement_weights[1]

        self.point_data["tmp_disp_vectors"] = (
            scalers[..., np.newaxis] * self.point_data["edge_normals"]
        )
        if kwargs.get("debug", False):
            pv.set_plot_theme("document")
            p = pv.Plotter(window_size=(2100, 991))
            if self.bone_reconstruction is not None:
                p.add_mesh(self.bone_reconstruction, opacity=0.4)

            auto_close = True  # False
            self.plot_with_arrows(
                external_plotter=p,
                active_vector_name="tmp_disp_vectors",
                add_mesh_kwargs=self.add_mesh_kwargs(opacity=0.8, culling="backface"),
                auto_close=auto_close,
            )
            # p.screenshot('screenshot.png', transparent_background=False)

        # Fuse the points with the bone
        if include_bone_fusion and self.bone_reconstruction is not None:
            self.points += self.point_data["tmp_disp_vectors"]
            fusion_displacements = self.fuse_to_bone(
                only_edge=True, dry_run=True, match_edge_translation=False
            )
            self.points = init_point_locs.copy()
            self.point_data["tmp_disp_vectors"] += fusion_displacements

        self._relax_tight_loops()
        self._sharpen_to_feature_angle(min_feature_angle=min_feature_angle)
        self._savgolay_filter_edge_components(
            n_layers_or_slice_locs=n_layers_or_slice_locs
        )
        self._smooth_bone_side_blend()
        self._smooth_joint_side_blend(max_feature_angle=max_feature_angle)
        disp_vecs = self.point_data["tmp_disp_vectors"]
        _ = [self.point_data.remove(x) for x in self.array_names if "tmp_" in x]
        if dry_run:
            return disp_vecs
        else:
            self.points += disp_vecs

    def _savgolay_filter_edge_components(
        self,
        init_window: int = 3,
        n_iters: int = 1,
        max_successful_window_growths: int = 3,
        n_layers_or_slice_locs: Optional[float] = None,
    ) -> None:
        """Smooth cartilage edges using Savitsky-Golay filtering of the edge components.

        Savitsky-Golay filtering includes and extends the moving average. Closed loops
        that are too small will not be smoothed, to account for small holes within the
        cartilage. If the filtering process results in additional cells failing scaled
        Jacobian checks, then the points are reset to their original locations.

        If the filtering succeeds, the window of points is allowed to grow to see if
        additional smoothing would be possible.

        Args:
            init_window: the initial number of neighboring points to account for during
            filtering. Must be an odd number. Defaults to 3.
            n_iters: the number of times to apply an identical filter. Defaults to 1.
            max_successful_window_growths: the maximum number of times the window of
                neighboring points is allowed to grow before leaving the function. While
                a larger window will result in smoother curves, shrinkage will develop.
                Defaults to 3.

        Raises:
            BlendingError: if filtering exclusively fails three times, and the initial
                mesh does not pass cell quality checks of the scaled Jacobian.

        """
        assert init_window % 2 == 1, f"Window must be odd, but {init_window} passed."
        init_n_bad = np.count_nonzero(
            self.find_low_quality_cells(
                metric_min=0,
                metric_max=1,
                quality_measure="scaled_jacobian",
                n_layers_or_slice_locs=n_layers_or_slice_locs,
            )
        )
        if n_iters < 1 and init_n_bad > 0:
            raise BlendingError("Elements of bad quality remain.")

        # Get perimeter edges
        pid_str, _ = self._pc_id_strs
        perim_edges, _ = self.perimeter_edges
        perim_edges.points += perim_edges.point_data["tmp_disp_vectors"]


        conn = perim_edges.connectivity()
        n_failed, n_passed, growth_counter = 0, 0, 0
        init_disp_vectors = self.point_data["tmp_disp_vectors"].copy()
        current_best_disp_vectors = init_disp_vectors.copy()
        orig_point_locs = self.point_locs
        with WorkProgress(desc="Smoothing edges") as pbar:
            while n_passed < max_successful_window_growths:
                # recalculate the window and polyorder
                window = init_window + 2 * growth_counter
                polyorder = 2 if init_window == 3 else 3

                # extract the closed loop
                for rid in np.unique(conn["RegionId"]):
                    line = conn.extract_cells(
                        conn["RegionId"] == rid
                    ).extract_surface(algorithm=None)
                    max_window_size = line.n_points // 4 * 2 + 2

                    # check if it is too small, or the window is too large
                    if line.n_points < 7:
                        continue
                    elif line.n_points // 4 < window:
                        window = max_window_size - 3
                        polyorder = 2 if window == 3 else polyorder

                    # apply the savitsky-golay filter to each triple component
                    sorted_pids = line.strip(join=True).lines[1:-1]
                    pts = line.points[sorted_pids].copy()
                    smoothed_points = pts.copy()

                    for _ in range(n_iters):
                        smoothed_points = savgol_filter(
                            smoothed_points, window, polyorder, axis=0, mode="wrap"
                        )

                    # assign the smoothed point locations
                    deltas = smoothed_points - pts
                    self.point_data["tmp_disp_vectors"][line.point_data[pid_str]] += (
                        deltas
                    )

                # check the Jacobians
                self.points += self.point_data["tmp_disp_vectors"]
                n_bad = np.count_nonzero(
                    self.find_low_quality_cells(
                        metric_min=0,
                        metric_max=1,
                        quality_measure="scaled_jacobian",
                        n_layers_or_slice_locs=n_layers_or_slice_locs,
                    )
                )
                self.points = orig_point_locs.copy()

                if n_bad - init_n_bad > 0:
                    n_failed += 1
                else:
                    n_passed += 1
                    current_best_disp_vectors = self.point_data["tmp_disp_vectors"]
                self.point_data["tmp_disp_vectors"] = init_disp_vectors.copy()

                # Check if smoothing has failed too many times
                if (n_failed > 3) or (window > max_window_size):
                    if n_passed > 0:
                        break
                    elif init_n_bad == 0:
                        break
                    else:
                        del line, conn
                        raise BlendingError("Failed to smooth blended edges.")

                # increment the counter
                growth_counter += 1
                pbar.update(1)

        self.point_data["tmp_disp_vectors"] = current_best_disp_vectors

    def _advancing_front_selection(
        self,
        n_cells_depth: int = 1,
        joint_side: bool = True,
        bone_side: bool = True,
    ) -> np.ndarray:
        """Select layers of surface cells internal from the cartilage edges.

        Args:
            n_cells_depth: number of layers from edges to select. Defaults to 1.
            joint_side: include joint-side cells in the selection. Defaults to True.
            bone_side: include bone-side cells in the selection. Defaults to True.

        Returns:
            Point IDs for the selection.

        """
        pid_str, _ = self._pc_id_strs
        self.point_data["tmp_is_interior"] = self.is_interior_point_mask
        self.point_data["tmp_is_bs_edge"] = self.is_bone_side_edge_mask
        self.point_data["tmp_is_js_edge"] = self.is_joint_side_edge_mask
        self.point_data["tmp_is_js"] = self.is_joint_side_point_mask
        self.point_data["tmp_is_bs"] = self.is_bone_side_point_mask

        surf = self.extract_surface(algorithm=None)
        is_edge = surf.point_data["tmp_is_bs_edge"].astype(bool) | surf.point_data[
            "tmp_is_js_edge"
        ].astype(bool)

        if not bone_side:
            is_edge[surf.point_data["tmp_is_bs_edge"].astype(bool)] = False

        if not joint_side:
            is_edge[surf.point_data["tmp_is_js_edge"].astype(bool)] = False

        edge_cells = surf.extract_points(is_edge)
        selection_pids = edge_cells.point_data[pid_str]
        for _ in range(n_cells_depth):
            edge_cells = edge_cells.extract_points(
                edge_cells.point_data["tmp_is_interior"].astype(bool),
                include_cells=False,
            )
            selection_pids = edge_cells.point_data[pid_str]
            is_selection = np.isin(surf.point_data[pid_str], selection_pids)
            edge_cells = surf.extract_points(is_selection)

        return selection_pids

    def _smooth_bone_side_blend(self) -> None:
        """Smooth surface point locations of the bone interface with an advancing front.

        The number of Laplacian smoothing iterations will be halved each time the front
        advances.
        """
        pid_str, _ = self._pc_id_strs
        selection_pids = self._advancing_front_selection(1, joint_side=False)
        surf = self.extract_surface(algorithm=None)
        is_selection = np.isin(surf.point_data[pid_str], selection_pids)
        edge_cells = surf.extract_points(is_selection)
        selection_reduction = (surf.n_cells - edge_cells.n_cells) / surf.n_cells
        iters = selection_reduction * 550
        selection_pids = edge_cells.point_data[pid_str]
        with WorkProgress(
            np.arange(1, 5), desc="Smoothing bone side blend", unit="fronts"
        ) as fronts:
            for front_idx in fronts:
                selection_pids = self._advancing_front_selection(
                    front_idx, joint_side=False
                )
                smoothing_successful, iters = self._smooth_edge_surface(
                    selected_pids=selection_pids, init_smooth_iters=iters
                )
                if not smoothing_successful:
                    break
                iters //= 2

    def _select_joint_side_feature_edges(
        self, max_feature_angle: float = 30
    ) -> pv.PolyData:
        """Select joint-side feature edges exceeding a maximum feature angle.

        Args:
            max_feature_angle: The maximum feature angle for the joint-side edge. Defaults to 30.

        Returns:
            The selected feature edges PolyData object.

        """
        pid_str, _ = self._pc_id_strs
        selection_pids = np.flatnonzero(self.is_joint_side_edge_mask)
        surf = self.extract_surface(algorithm=None)
        surf.points += surf.point_data["tmp_disp_vectors"]
        is_selection = np.isin(surf.point_data[pid_str], selection_pids)
        edge_cells = surf.extract_points(is_selection)

        feature_edges = edge_cells.extract_feature_edges(
            feature_angle=max_feature_angle,
            feature_edges=True,
            boundary_edges=False,
            non_manifold_edges=False,
            manifold_edges=False,
        )
        lines = feature_edges.lines.reshape(-1, 3)[:, 1:]
        bs_pids = np.flatnonzero(feature_edges["tmp_is_bs"].astype(bool))
        is_vertical_line = np.isin(lines, bs_pids).any(axis=1)
        feature_edges = feature_edges.extract_cells(~is_vertical_line)
        return feature_edges

    def _smooth_joint_side_blend(self, max_feature_angle: float = 30) -> None:
        """Smooth the joint-side edges using an advancing front.

        Excludes any blended edges that are less than a provided maximum feature angle,
        which are already smooth enough.

        Args:
            max_feature_angle: The maximum angle dictating smooth enough. Defaults to 30.

        """
        pid_str, _ = self._pc_id_strs
        feature_edges = self._select_joint_side_feature_edges(
            max_feature_angle=max_feature_angle
        )
        surf = self.extract_surface(algorithm=None)
        surf.points += surf.point_data["tmp_disp_vectors"]
        init_volume = surf.volume
        m = self.copy()
        m.points += m.point_data["tmp_disp_vectors"]
        m_surf = m.extract_surface(algorithm=None)
        m_surf.set_active_scalars(None)
        with WorkProgress(desc="Smoothing joint side blend") as pbar:
            while feature_edges.n_cells > 0:
                needs_smooth_pids = feature_edges.point_data[pid_str][
                    feature_edges.point_data["tmp_is_js_edge"].astype(bool)
                ]
                for _ in np.arange(1, 6):
                    smoothing_successful, iters = self._smooth_edge_surface(
                        selected_pids=needs_smooth_pids, init_smooth_iters=20
                    )
                    surf = self.extract_surface(algorithm=None)
                    surf.points += surf.point_data["tmp_disp_vectors"]
                    vol_proportion = surf.volume / init_volume
                    if not smoothing_successful or vol_proportion < 0.95:
                        break
                    iters = int(iters * 0.75)
                    is_selection = np.isin(surf.point_data[pid_str], needs_smooth_pids)
                    edge_cells = surf.extract_points(is_selection)
                    needs_smooth_pids = edge_cells.point_data[pid_str][
                        edge_cells.point_data["tmp_is_interior"].astype(bool)
                    ]
                if not smoothing_successful or vol_proportion < 0.95:
                    break
                feature_edges = self._select_joint_side_feature_edges(
                    max_feature_angle=max_feature_angle
                )
                pbar.update(1)
        pass

    def _smooth_edge_surface(
        self,
        selected_pids: ArrayLike,
        init_smooth_iters: int,
        n_layers_or_slice_locs: Optional[int] = 4,
    ) -> Tuple[bool, int]:
        """Smooth the interior points of a surface.

        The surface is created by passing point IDs to the CartilageMesh.extract_points
        method. Smoothing is then performed on the interior points of the extracted
        surface. If the smoothing results in an increase in cells with scaled Jacobians
        leess than zero, retry using fewer smoothing iterations.

        Args:
            selected_pids: the point IDs used in extracting the surface.
            init_smooth_iters: the number of Laplacian smoothing iterations to attempt
                the first time.

        Returns:
            A tuple containing the status of the smoothing operation, as well as the
            final number of iterations.

        """
        pid_str, _ = self._pc_id_strs
        init_point_locs = self.point_locs
        self.points += self.point_data["tmp_disp_vectors"]
        init_n_bad = np.count_nonzero(
            self.find_low_quality_cells(
                metric_min=0,
                metric_max=1,
                quality_measure="scaled_jacobian",
                n_layers_or_slice_locs=n_layers_or_slice_locs,
            )
        )
        surf = self.extract_surface(algorithm=None)
        edge_cells = surf.extract_points(
            np.isin(surf.point_data[pid_str], selected_pids)
        )
        iter_choice = np.linspace(init_smooth_iters, 0, 10, dtype=int)
        smoothing_successful = False
        for n_iters in iter_choice:
            smoothed = pv.PolyData(edge_cells.points.copy(), edge_cells.cells).smooth(
                # ).smooth_taubin(
                n_iter=n_iters,
                boundary_smoothing=False,  # , normalize_coordinates=True
            )

            # Check if we're done
            allclose = np.allclose(edge_cells.points, smoothed.points, atol=1e-2)
            self.points[edge_cells.point_data[pid_str]] = smoothed.points
            n_bad = np.count_nonzero(
                self.find_low_quality_cells(
                    metric_min=0,
                    metric_max=1,
                    quality_measure="scaled_jacobian",
                    n_layers_or_slice_locs=n_layers_or_slice_locs,
                )
            )
            self.points = init_point_locs.copy() + self.point_data["tmp_disp_vectors"]
            if allclose or n_bad > init_n_bad:
                break
            else:
                self.point_data["tmp_disp_vectors"][edge_cells.point_data[pid_str]] += (
                    smoothed.points - edge_cells.points
                )
                smoothing_successful = True
                break
        self.points = init_point_locs.copy()
        return smoothing_successful, n_iters

    # @with_debug_kwarg
    def _sharpen_to_feature_angle(
        self, min_feature_angle: float = 25, **kwargs
    ) -> None:
        """Reduce blending displacements of overly smooth joint-side edge segments.

        When the initial displacements cause the joint-side edge to become overly
        smooth, the scaled Jacobians of the cells will be less than zero. If a segment
        is overly smooth, as defined by a minimum angle, the displacements will be
        relaxed until the angle is sharpened sufficiently.

        Args:
            min_feature_angle: The minimum angle for a join-side edge segment to be
                sharp enough. Defaults to 25.

        """
        pid_str, _ = self._pc_id_strs
        self.point_data["tmp_is_bs_edge"] = self.is_bone_side_edge_mask
        self.point_data["tmp_is_js_edge"] = self.is_joint_side_edge_mask
        self.point_data["tmp_is_js"] = self.is_joint_side_point_mask
        self.point_data["tmp_is_bs"] = self.is_bone_side_point_mask

        joint_surf = self.extract_surface(algorithm=None)
        joint_surf.points += joint_surf.point_data["tmp_disp_vectors"]
        orig_disp_vecs = self.point_data["tmp_disp_vectors"].copy()
        too_smooth_edges = self._find_smooth_edges_to_sharpen(joint_surf)
        with WorkProgress(desc="Sharpening edges") as pbar:
            while too_smooth_edges.n_cells > 0:
                # Select the top nodes which share the same cell IDs of the offending
                # edges
                bone_edges = joint_surf.extract_points(
                    joint_surf.point_data["tmp_is_bs_edge"].astype(bool),
                ).extract_all_edges()

                shares_pid = np.isin(
                    bone_edges.point_data[pid_str],
                    too_smooth_edges.point_data[pid_str],
                )

                # Select only edges connected to the overly smoothed point
                bone_edge_lines = bone_edges.lines.reshape(-1, 3)[:, 1:]
                has_too_smooth_point = np.isin(
                    bone_edge_lines,
                    np.flatnonzero(shares_pid),
                ).any(axis=1)

                bone_edges.remove_cells(~has_too_smooth_point, inplace=True)

                # Remove the dangling edges on the joint-side interface
                bone_edge_lines = bone_edges.lines.reshape(-1, 3)[:, 1:]
                is_on_edge = np.isin(
                    bone_edge_lines,
                    np.flatnonzero(bone_edges.point_data["tmp_is_js_edge"]),
                ).all(axis=1)
                bone_edges.remove_cells(is_on_edge, inplace=True)

                critical_pids = bone_edges.point_data[pid_str]

                # Back off the critical PIDs
                self.point_data["tmp_disp_vectors"][critical_pids] -= (
                    0.1 * orig_disp_vecs[critical_pids]
                )

                joint_surf = self.extract_surface(algorithm=None)
                joint_surf.points += joint_surf.point_data["tmp_disp_vectors"]
                too_smooth_edges = self._find_smooth_edges_to_sharpen(
                    joint_surf, min_feature_angle=min_feature_angle
                )
                pbar.update(1)

                if kwargs.get("debug", False):
                    auto_close = True  # False
                    joint_edge_cells = joint_surf.extract_points(
                        joint_surf.point_data["tmp_is_js_edge"].astype(bool)
                    )
                    p = pv.Plotter()
                    p.add_mesh(joint_surf, color="tan", opacity=1, show_edges=True)
                    p.add_mesh(
                        too_smooth_edges,
                        color="r",
                        label="Needs Sharpening",
                        line_width=10,
                    )
                    p.show(auto_close=auto_close)
                    # p.screenshot("screenshot.png")

    def _find_smooth_edges_to_sharpen(
        self, joint_surf: PolyData, min_feature_angle: float = 25
    ) -> PolyData:
        """Select joint-side edge feature angles below a minimum.

        Args:
            joint_surf: a surface mesh extracted from the CartilageMesh.
            min_feature_angle: select any segments with feature angle under this value. Defaults to 25.

        Returns:
            PolyData with the overly smooth edges.

        """
        pid_str, _ = self._pc_id_strs
        joint_edge_cells = joint_surf.extract_points(
            joint_surf.point_data["tmp_is_js_edge"].astype(bool)
        )

        # Check if any of the feature edges have gotten too small on the joint-side
        # edge
        sharp_enough_edges = joint_edge_cells.extract_feature_edges(
            feature_angle=min_feature_angle,
            feature_edges=True,
            boundary_edges=False,
            non_manifold_edges=False,
            manifold_edges=False,
        )

        # Remove any surviving bone-side points
        is_js_edge = sharp_enough_edges.point_data["tmp_is_js_edge"].astype(bool)
        sharp_lines = sharp_enough_edges.lines.reshape(-1, 3)[:, 1:]
        is_not_on_edge = np.isin(sharp_lines, np.flatnonzero(~is_js_edge)).any(axis=1)
        sharp_enough_edges.remove_cells(is_not_on_edge, inplace=True)

        # Select the too-small edges
        too_smooth_edges = joint_edge_cells.extract_all_edges()
        smooth_lines = too_smooth_edges.lines.reshape(-1, 3)[:, 1:]
        is_js_edge = too_smooth_edges.point_data["tmp_is_js_edge"].astype(bool)
        is_not_on_edge = np.isin(smooth_lines, np.flatnonzero(~is_js_edge)).any(axis=1)
        remove_pids = np.flatnonzero(
            np.isin(
                too_smooth_edges.point_data[pid_str],
                sharp_enough_edges.point_data[pid_str],
            )
        )
        is_sharp_enough = np.isin(smooth_lines, remove_pids).all(axis=1)
        too_smooth_edges.remove_cells(is_not_on_edge | is_sharp_enough, inplace=True)
        return too_smooth_edges

    def _relax_tight_loops(self) -> None:
        """Reduce the edge displacement vectors when a loop is too tight.

        If a hole is small, we could introduce self-intersections during blending. To
        avoid this, we treat each closed loop as a circular circumference. We can
        calculate the area of the circle before and after blending. If the area
        reduction is too large, we can reduce the blending displacements by 80%.
        """
        pid_str, _ = self._pc_id_strs
        perim_edges, _ = self.perimeter_edges
        conn = perim_edges.connectivity()
        rids = np.unique(conn["RegionId"])
        with WorkProgress(rids, desc="Relaxing tight loops", unit="regions") as rids:
            for rid in rids:
                line = conn.extract_cells(conn["RegionId"] == rid)
                init_area = line.length**2 / np.pi / 4
                line_new = line.copy()
                line_new.points = line.points + line.point_data["tmp_disp_vectors"]
                new_area = line_new.length**2 / np.pi / 4
                if new_area / init_area < 0.2:
                    line["tmp_disp_vectors"] *= 0.2
                    self.point_data["tmp_disp_vectors"][line.point_data[pid_str]] = (
                        line["tmp_disp_vectors"]
                    )

    def _compute_edge_normals_for_blending(
        self, weight_towards_bone: float = 0.2
    ) -> None:
        """Compute normals along the bone/joint side edges of the cartilage mesh.

        Normals on the bone-side edge will expand the laminar edges, while normals on
        the joint-side edge will contract the laminar edges.

        The bone-side normals can be weighted towards the bone, up to a maximum angle of
        45 degrees.

        Args:
            weight_towards_bone: weigh the bone-side edge normals heavier towards the
                bone surface, up to a maximum of 45 degrees when equal to 1.0. Defaults
                to 0.2.

        """
        pid_str = self._pc_id_strs[0]
        # Get the cartilage depths for each point
        self.compute_p2p_cartilage_depths()

        # Extract the surface, and calculate normals
        self.point_data["tmp_is_bs_edge"] = self.is_bone_side_edge_mask
        self.point_data["tmp_is_js_edge"] = self.is_joint_side_edge_mask

        surf = self.extract_surface(algorithm=None)
        surf_top_edge = surf.extract_points(
            surf.point_data["tmp_is_bs_edge"].astype(bool)
        )

        # Calculate normals 45 degree towards bone
        ste_pd = pv.PolyData(surf_top_edge.points, surf_top_edge.cells)
        ste_pd.compute_normals(point_normals=True, cell_normals=False, inplace=True)
        surf_top_edge.point_data["edge_normals"] = ste_pd.point_normals
        is_not_edge = ~surf_top_edge.point_data["tmp_is_bs_edge"].astype(
            bool
        ) & ~surf_top_edge.point_data["tmp_is_js_edge"].astype(bool)
        surf_top_edge.point_data["edge_normals"][is_not_edge] = np.zeros(3)

        # Calculate straight out of laminar edge normals
        perim_edges, perim_cells = self.perimeter_edges
        perim_surf = perim_cells.extract_surface(algorithm=None)
        perim_surf.compute_normals(point_normals=True, cell_normals=False, inplace=True)

        # Preallocate edge normal arrays
        self.point_data["edge_normals"] = np.zeros((self.n_points, 3))
        self.point_data["edge_normals_2"] = self.point_data["edge_normals"].copy()

        # Assign the edge normals
        self.point_data["edge_normals"][surf_top_edge[pid_str]] = (
            surf_top_edge.point_data["edge_normals"]
        )
        self.point_data["edge_normals_2"][perim_surf[pid_str]] = perim_surf.point_data[
            "Normals"
        ]

        # Weight the normals
        self.point_data["edge_normals"] = np.average(
            np.stack(
                (self.point_data["edge_normals"], self.point_data["edge_normals_2"])
            ),
            axis=0,
            weights=(weight_towards_bone, 1 - weight_towards_bone),
        )

        # Reverse and unitize the joint-side normals
        self.point_data["edge_normals"][self.point_data["tmp_is_js_edge"]] *= -1
        self.point_data["edge_normals"] = trimesh.unitize(
            self.point_data["edge_normals"]
        )

        # Cleanup
        [
            self.point_data.remove(x)
            for x in ("edge_normals_2", "tmp_is_bs_edge", "tmp_is_js_edge")
        ]

    @without_activation
    @stage("Smoothing surface")
    def smooth_side(self, side: str, n_iter: int = 20) -> None:
        """Smooth the bone- or joint-side surface of the cartilage mesh.

        Only the surface faces are smoothed, edgemost surface points will not translate.

        Args:
            side: the side to smooth, either 'bone_side' or 'joint_side'.
            n_iter: the number of Laplacian smoothing iterations to smooth. Defaults to
                20.

        Raises:
            CartilageMeshError: if the side parameter is not recognized.

        """
        pid_str, _ = self._pc_id_strs
        if side.lower() not in ["joint_side", "bone_side"]:
            raise CartilageMeshError(
                f"Side: {side} did not match either 'joint_side' or 'bone_side'."
            )
        self.point_data["tmp_js_mask"] = self.is_joint_side_point_mask
        self.point_data["tmp_bs_mask"] = self.is_bone_side_point_mask
        self.point_data["tmp_interior_mask"] = self.is_interior_point_mask
        surf = self.extract_surface(algorithm=None)
        is_interior = surf.point_data["tmp_interior_mask"].astype(bool)
        if side.lower() == "joint_side":
            rough_surf = surf.extract_points(
                is_interior & surf.point_data["tmp_js_mask"].astype(bool)
            )
        else:
            rough_surf = surf.extract_points(
                is_interior & surf.point_data["tmp_bs_mask"].astype(bool)
            )
        smoothed = pv.PolyData(rough_surf.points.copy(), rough_surf.cells).smooth(
            n_iter=n_iter,
            boundary_smoothing=False,  # , normalize_coordinates=True
        )
        self.points[rough_surf.point_data[pid_str]] = smoothed.points
        [
            self.point_data.remove(x)
            for x in ("tmp_js_mask", "tmp_bs_mask", "tmp_interior_mask")
        ]

    def blend_with_retries(
        self,
        minimum_grade_scaling: float = 0.3,
        n_layers_or_slice_locs: Optional[int] = 4,
        **blend_to_bone_interface_kwargs,
    ) -> None:
        kwargs = blend_to_bone_interface_kwargs
        init_dry_run = kwargs.pop("dry_run", False)
        include_bone_fusion = kwargs.pop("include_bone_fusion", True)
        init_grade = kwargs.pop("grade", 1)
        grade_scalers = np.round(np.linspace(1, 0, 11), 2)
        init_point_locs = self.point_locs

        first_attempt = True
        grade_scalers = np.hstack((minimum_grade_scaling, grade_scalers))
        with WorkProgress(grade_scalers, desc="Blending", unit="grades") as pbar:
            for gs in pbar:
                grade = np.round(gs * init_grade, 2)
                pbar.set_description(f"Blending with grade {grade}.")
                if gs < minimum_grade_scaling:
                    raise LimitReachedError(
                        f"Subceeded minimum grade scaler {minimum_grade_scaling}."
                    )
                self.points = init_point_locs.copy()
                try:
                    proposed_displacements = self.blend_to_bone_interface(
                        grade=grade,
                        dry_run=True,
                        include_bone_fusion=include_bone_fusion,
                        **kwargs,
                    )
                    self.points += proposed_displacements
                    if self.find_low_quality_cells(
                        metric_min=0,
                        metric_max=1,
                        quality_measure="scaled_jacobian",
                        n_layers_or_slice_locs=n_layers_or_slice_locs,
                    ).any():
                        first_attempt = False
                        raise BlendingError
                    elif not first_attempt:
                        break
                    else:
                        first_attempt = False
                except BlendingError:
                    pass
                    if gs > minimum_grade_scaling:
                        continue
                    else:
                        self.points = init_point_locs.copy()
                        raise LimitReachedError(
                            f"Subceeded minimum grade scaler {minimum_grade_scaling}."
                        )
        if init_dry_run:
            self.points = init_point_locs.copy()
            return proposed_displacements

    def blended_cell_quality(self, depth: int = 3) -> pd.DataFrame:
        """Return a pandas describe dataframe of aggregate edge cell quality.

        Args:
            depth: The number of layers from the edge to examine. Defaults to 3.

        Returns:
            Pandas describe dataframe with <depth> columns of scaled jacobian data.

        """
        new_mesh = self.copy()
        new_mesh.get_cell_quality(save_cell_data=True)
        new_mesh.set_active_scalars("scaled_jacobian")
        perim_cells = new_mesh.perimeter_cells
        ring_jacobians = pd.Series(
            perim_cells.cell_data["scaled_jacobian"], name="layer_0"
        )

        pid_str, cid_str = self._pc_id_strs
        series_collector = [None] * depth
        series_collector[0] = ring_jacobians
        for layer_idx in np.arange(1, depth):
            previous_pids = perim_cells.point_data[pid_str]
            previous_cids = perim_cells.cell_data[cid_str]
            new_cids = new_mesh.extract_points(previous_pids).cell_data[cid_str]
            new_cids = np.setdiff1d(new_cids, previous_cids)
            perim_cells = new_mesh.extract_cells(new_cids)
            series_collector[layer_idx] = pd.Series(
                perim_cells.cell_data["scaled_jacobian"], name=f"layer_{layer_idx}"
            )

        return pd.concat([x.describe() for x in series_collector], axis=1)


class GridCartilageMesh(CartilageMesh):
    def __init__(
        self,
        *args,
        raytrace_pattern: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raytrace_function = None
        if self.raytrace_pattern is not None:
            self.raytrace_pattern = raytrace_pattern

    @property
    def raytrace_pattern(self) -> Optional[str]:
        """Select either 'capsule', or 'plane' based raytracing grids."""
        if self.raytrace_function == self.capsule_raytrace:
            return "capsule"
        elif self.raytrace_function == self.planar_raytrace:
            return "plane"
        else:
            return None

    @raytrace_pattern.setter
    def raytrace_pattern(self, type_str: str) -> None:
        if type_str.lower() == "capsule":
            self.raytrace_function = self.capsule_raytrace
        elif type_str.lower() == "plane":
            self.raytrace_function = self.planar_raytrace
        else:
            raise NotImplementedError(f"Unknown raytracing type {type_str}")

    def capsule_raytrace(
        self,
        axis_points: ArrayLike,
        radial_resolution: int,
        longitudinal_resolution: int,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Raytrace initial point locations using a capsule shaped grid of rays.

        A capsule is a cylinder with two hemispherical caps on each end. The capsule is
        defined by an axis of two cartesian axis_points, a longitudinal resolution, and
        a radial resolution.

        Args:
            axis_points: the two cartesian locations defining the capsule axis.
            radial_resolution: the approximate element size around the capsule shaft.
            longitudinal_resolution: the approximate element size along the capsule
                shaft.

        Returns:
            Arrays of bone- and joint-side ray intersections.

        """
        directions, origins, n_radial_nodes, n_axial_nodes = rays.femur_cartilage(
            *axis_points,
            radial_res=radial_resolution,
            axial_res=longitudinal_resolution,
        )
        bone_side_points, joint_side_points = rays.find_surface_vertices(
            directions, origins, self.reconstruction
        )
        return _by_value(
            bone_side_points, joint_side_points, (n_radial_nodes, n_axial_nodes)
        )

    def _initialize_capsule_mesh(
        self, points: ArrayLike, dims: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_radial_nodes, n_axial_nodes = dims
        n_points = points.shape[0]
        cells = hc.initialize_hexahedral_plane(n_radial_nodes, n_points)
        UGrid = UnstructuredGrid.from_cell_def(points=points, cell_def=cells.T).clean()
        points = UGrid.points.copy()
        cells = UGrid.cells_dict[VTK_HEXAHEDRON].copy().T
        return points, cells.T

    def planar_raytrace(
        self,
        i_resolution: float,
        j_resolution: float,
        plane_center: Optional[ArrayLike] = None,
        plane_normal: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        reconstruction = self.wrapped_reconstruction
        if plane_center is None or plane_normal is None:
            _, plane_center, plane_normal = pv.fit_plane_to_points(
                reconstruction.points, return_meta=True
            )
        if np.dot(trimesh.unitize(plane_normal), np.array([0, 0, 1])) < 0:
            plane_normal *= -1
        projected = reconstruction.project_points_to_plane(
            origin=plane_center, normal=plane_normal
        )
        bounds = np.array(projected.bounds).reshape(-1, 2)
        plane_side_length = np.linalg.norm(bounds, "fro")  # frobenius
        resolution = (i_resolution, j_resolution)
        i_res, j_res = np.ceil(plane_side_length / resolution).astype(int)
        plane = pv.Plane(
            center=plane_center,
            direction=plane_normal,
            i_size=plane_side_length,
            j_size=plane_side_length,
            i_resolution=i_res,
            j_resolution=j_res,
        )
        plane.point_data.clear()

        if kwargs.get("debug", False):
            p = pv.Plotter()
            p.add_mesh(reconstruction, opacity=0.8)
            p.add_mesh(plane, opacity=0.8, color="r", show_edges=True)
            p.add_axes()
            p.show()

        # Create a grid of rays
        terminations = plane.points.copy()
        plane.translate(-plane_side_length * 0.5 * plane_normal, inplace=True)
        origins = plane.points
        directions = trimesh.unitize(terminations - origins)

        # Perform raytracing
        reconstruction = self.reconstruction
        bone_side_points, joint_side_points = rays.find_surface_vertices(
            directions, origins, reconstruction
        )
        return bone_side_points, joint_side_points, i_res, j_res

    @stage("Initializing grid and raytracing")
    def initialize_mesh(self, *ray_args, **ray_kwargs) -> Self:
        if self.raytrace_pattern == "capsule" and self.raytrace_function is not None:
            bsp, jsp, dims = self.raytrace_function(*ray_args, **ray_kwargs)
            points, cell_def = self._initialize_capsule_mesh(
                points=np.vstack((bsp, jsp)), dims=dims
            )
        elif self.raytrace_pattern == "plane" and self.raytrace_function is not None:
            bsp, jsp, i_res, j_res = self.raytrace_function(*ray_args, **ray_kwargs)
            points = np.vstack((bsp, jsp))
            cells = hc.initialize_hexahedral_plane(
                i_res + 1, 2 * (j_res + 1) * (i_res + 1)
            )
            cell_def = cells.T
        else:
            raise NotImplementedError(
                "raytrace_pattern must be set to either 'capsule' or 'plane'."
            )
        return self.update_cell_def(cell_def=cell_def, points=points)


class SectorCartilageMesh(CartilageMesh):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def raytrace(
        self,
        elem_counts: Union[Tuple[int], np.ndarray] = None,
        elem_sizes: Union[Tuple[float], np.ndarray] = None,
        bounding_box_scaling_factor: float = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        reconstruction = self.reconstruction
        centroid = reconstruction.centroid
        offset = [0, 0, np.max(reconstruction.extents)]
        ray_origin_plane_point = centroid + offset

        directions, origins, elem_counts = rays.tibia_cartilage(
            ray_origin_plane_point=ray_origin_plane_point,
            reconstruction=reconstruction,
            elem_counts=elem_counts,
            elem_sizes=elem_sizes,
            bounding_box_scaling_factor=bounding_box_scaling_factor,
            **kwargs,
        )

        bone_side_points, joint_side_points = rays.find_surface_vertices(
            directions,
            origins,
            reconstruction,
            max_degree_redirection=10,
            neighborradius=5,
            **kwargs,
        )
        return bone_side_points, joint_side_points, elem_counts

    @stage("Initializing sectors and raytracing")
    def initialize_mesh(
        self,
        elem_counts: Union[Tuple[int], np.ndarray] = None,
        elem_sizes: Union[Tuple[float], np.ndarray] = None,
        bounding_box_scaling_factor: float = None,
        **kwargs,
    ) -> Self:
        bsp, jsp, updated_elem_counts = self.raytrace(
            elem_counts=elem_counts,
            elem_sizes=elem_sizes,
            bounding_box_scaling_factor=bounding_box_scaling_factor,
            **kwargs,
        )
        if not np.array_equal(updated_elem_counts, elem_counts):
            logging.info("Updated element counts while meshing tibial cartilage.")
        ratio_v, ratio_h, radius = updated_elem_counts
        points, cells = hc.compute_hexa_tibial(
            bsp,
            jsp,
            ratio_v + 1,
            ratio_h + 1,
            radius + 1,
            **kwargs,
        )
        cell_def = cells
        return self.update_cell_def(cell_def=cell_def, points=points)


class MeniscusMesh(TissueMesh):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_subdivided = False
        self.n_in_plane_subdivide = 0
        self.n_axial_subdivide = 0
        self.centers = None
        self.tissue_type: TissueType = TissueType.CARTILAGE
        self.cartilage_type: CartilageType = CartilageType.VERMIFORM

    @stage("Subdividing meniscus")
    def subdivide(self, inplace: bool = False, directions: list[int] = None) -> Self:
        if directions is None:
            directions = [0, 1]
        directions = tuple(directions)
        new_mesh = self if inplace else self.copy()
        for dir_ in directions:
            new_mesh = new_mesh._inplane_subdivide_along_axis(dir_)
            new_mesh.n_in_plane_subdivide += 1
        new_mesh.is_subdivided = True
        return new_mesh

    @property
    def n_layers(self) -> int:
        cells_per_layer = 10 * 2 ** (self.n_in_plane_subdivide + self.n_axial_subdivide)
        return self.n_cells // cells_per_layer

    def get_layer_cids(self, idxs) -> np.ndarray:
        idxs = np.array(idxs).flatten()
        cids = np.tile(np.arange(10), (idxs.size, 1)).T + 10 * idxs
        binary_exponent = self.n_axial_subdivide + self.n_in_plane_subdivide
        for x in np.arange(binary_exponent, self.n_axial_subdivide, -1):
            cids = np.stack((cids, cids + self.n_cells // 2**x), axis=2)
        cids = cids.reshape(10, -1).T.reshape(idxs.size, -1)
        return np.squeeze(cids)

    # @with_debug_kwarg
    @without_activation
    @stage("Expanding meniscus profile")
    def expand_profile(
        self,
        max_node_shift: float = 5,
        both_directions=False,
        inplace: bool = False,
        expand_tips: bool = True,
        **kwargs,
    ) -> Self:
        """Expand the laminar profile of the mesh by shifting nodes.

        Expand the nodes on the laminar edge of the cartilage mesh using raytracing with
        the reconstruction. This step resists shrinkage due to smoothing.

        Parameters
        ----------
            max_node_shift (float): The maximum amount by which a node can be shifted.
            both_directions (bool): Flag indicating whether to perform raytracing in both directions.
            inplace (bool): Flag indicating whether to modify the mesh in-place or create a new mesh.
            expand_tips (bool): Flag indicating whether to expand the tips of the profile.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            Self: The expanded mesh.

        Raises
        ------
            None

        """
        new_mesh = self.copy()
        pid_str, cid_str = self._pc_id_strs
        reconstruction = self.reconstruction

        if kwargs.get("debug", False):
            mat_dir = matfile_dir.joinpath("meniscus")
            mat_dict = import_matlab_data(
                mat_dir / "expandHexMesh_Meniscus_64_pre_rays_averaging.mat"
            )
            new_mesh = new_mesh.copy()
            new_mesh.reconstruction = trimesh.Trimesh(
                vertices=mat_dict["v"], faces=mat_dict["f"] - 1
            )
            new_mesh = new_mesh.update_cell_def(
                points=mat_dict["nodes"], cell_def=mat_dict["hex"] - 1
            )

        # Create ray directions
        surf = new_mesh.extract_surface(algorithm=None)
        surf.compute_normals(inplace=True)

        # select the faces on the ends of the horseshoe shape
        first_cids = np.arange(10)
        last_cids = np.arange(-10, 0, 1)
        binary_exp = self.n_axial_subdivide + self.n_in_plane_subdivide
        for x in np.arange(binary_exp, self.n_axial_subdivide, -1):
            first_cids = np.hstack((first_cids, first_cids + new_mesh.n_cells // 2**x))
            last_cids = np.hstack((last_cids, last_cids - new_mesh.n_cells // 2**x))
        last_cids = new_mesh.n_cells + last_cids
        first_cells = new_mesh.cell_def[
            first_cids
            # 0:faces_per_slice
        ]  # 10 is the number of quad patches
        last_cells = new_mesh.cell_def[last_cids]

        first_face_pids = first_cells[:, :4].flatten()
        last_face_pids = last_cells[:, 4:].flatten()

        if kwargs.get("debug", False):
            i_cids = new_mesh.get_layer_cids([0, new_mesh.n_layers - 1])
            i_cells = new_mesh.cell_def[i_cids[0]]
            first_face_pids = i_cells[:, :4].flatten()
            second_face_pids = i_cells[:, 4:]
            first_cell_mesh = new_mesh.extract_cells(i_cids)
            p = pv.Plotter()
            p.add_mesh(first_cell_mesh, show_edges=True, opacity=0.5)
            p.add_point_labels(
                new_mesh.points[np.unique(first_face_pids)], np.unique(first_face_pids)
            )
            p.add_point_labels(
                new_mesh.points[np.unique(second_face_pids)],
                np.unique(second_face_pids),
            )
            p.add_axes()

        first_mesh = new_mesh.update_cell_def(
            points=new_mesh.points, cell_def=first_cells
        )
        last_mesh = new_mesh.update_cell_def(
            points=new_mesh.points, cell_def=last_cells
        )

        # Select the points that are allowed to move, and the ones that remain frozen
        outer_pids = surf.point_data[pid_str]
        is_mobile_pids = ~np.isin(
            outer_pids, np.hstack((first_face_pids, last_face_pids))
        )
        mobile_pids = surf.extract_points(
            is_mobile_pids, include_cells=True, adjacent_cells=True
        ).point_data[pid_str]
        frozen_pids = np.setdiff1d(outer_pids, mobile_pids)

        is_frozen_pids = np.isin(outer_pids, frozen_pids)
        is_mobile_pids = np.isin(outer_pids, mobile_pids)

        # average the point normals with a ray from the center of the hex cell to the
        # point.
        # Get the layers
        layer_cids = new_mesh.get_layer_cids(np.arange(new_mesh.n_layers))

        # Get the third cell
        center_cids = layer_cids[:, 3]

        # Get the center (front face, 1) pid
        center_pids = new_mesh.cell_def[center_cids, 1]

        # Handle the final (back face, 5) pid
        back_face_center_point = new_mesh.cell_def[center_cids[-1], 5]
        center_pids = np.hstack((center_pids, back_face_center_point))
        center_locs = new_mesh.point_locs[center_pids]
        center_locs_reps = np.tile(center_locs, (17, 1, 1)).reshape(-1, 3, order="F")
        if kwargs.get("debug", False):
            p.add_mesh(pv.PolyData(center_locs), color="r")
            p.show()

        # get the front face pids
        face_pids = new_mesh.cell_def[layer_cids, :4]

        # handle the last face
        last_face_pids = new_mesh.cell_def[layer_cids[-1], 4:]
        face_pids = np.stack((*face_pids, last_face_pids))

        # resist the first and last faces shrinking
        if expand_tips:
            surf = new_mesh.extract_surface(algorithm=None)
            terminal_faces = surf.extract_points(
                np.isin(surf.point_data[pid_str], face_pids[0]), adjacent_cells=False
            ).extract_surface(algorithm=None)
            terminal_faces.compute_normals(inplace=True)
            origins = terminal_faces.points
            directions = trimesh.unitize(terminal_faces.point_data["Normals"])
            tri_idx, ray_idx, locations = reconstruction.ray.intersects_id(
                ray_origins=origins,
                ray_directions=directions,
                return_locations=True,
                multiple_hits=False,
            )
            disp = np.zeros_like(terminal_faces.points)
            disp_vecs = locations - origins[ray_idx]
            disp_vec_dist = np.quantile(trimesh.util.row_norm(disp_vecs), 0.3)
            disp[:] = trimesh.unitize(disp_vecs).mean(axis=0) * disp_vec_dist
            new_mesh.points[
                np.isin(
                    new_mesh.point_data[pid_str], terminal_faces.point_data[pid_str]
                )
            ] += disp

            terminal_faces = surf.extract_points(
                np.isin(surf.point_data[pid_str], face_pids[-1]), adjacent_cells=False
            ).extract_surface(algorithm=None)
            terminal_faces.compute_normals(inplace=True)
            origins = terminal_faces.points
            directions = trimesh.unitize(terminal_faces.point_data["Normals"])
            tri_idx, ray_idx, locations = reconstruction.ray.intersects_id(
                ray_origins=origins,
                ray_directions=directions,
                return_locations=True,
                multiple_hits=False,
            )
            disp = np.zeros_like(terminal_faces.points)
            disp_vecs = locations - origins[ray_idx]
            disp_vec_dist = np.quantile(trimesh.util.row_norm(disp_vecs), 0.3)
            disp[:] = trimesh.unitize(disp_vecs).mean(axis=0) * disp_vec_dist
            new_mesh.points[
                np.isin(
                    new_mesh.point_data[pid_str], terminal_faces.point_data[pid_str]
                )
            ] += disp

        # get the surface
        surf = new_mesh.extract_surface(algorithm=None)
        surf.compute_normals(inplace=True)
        moving_surf = surf.extract_points(
            is_mobile_pids, include_cells=True, adjacent_cells=False
        ).extract_surface(algorithm=None)
        moving_surf.compute_normals(inplace=True)

        # Assign the location at pid[1] to the points
        hex_cids = moving_surf.cell_data[cid_str]
        moving_surf_pids = moving_surf.point_data[pid_str]
        moving_surf.point_data["slice_centers"] = np.zeros_like(moving_surf.points)
        for face in face_pids:
            center_pid = face[3, 1]
            center_loc = new_mesh.point_locs[center_pid]
            needs_center = np.isin(moving_surf_pids, face)
            moving_surf.point_data["slice_centers"][needs_center] = center_loc
        moving_surf.point_data["origin_rays"] = trimesh.unitize(
            moving_surf.points - moving_surf.point_data["slice_centers"]
        )
        origins = moving_surf.points
        ray_directions = np.stack(
            (
                moving_surf.point_data["origin_rays"],
                moving_surf.point_data["Normals"],
            )
        ).sum(axis=0)
        ray_directions = trimesh.unitize(ray_directions)

        # #TODO: Perform outward raytracing with the reconstruction
        tri_idx, ray_idx, locations = reconstruction.ray.intersects_id(
            ray_origins=origins,
            ray_directions=ray_directions,
            return_locations=True,
            multiple_hits=False,
        )

        # check that all rays intersected backfaces
        ray_vecs = locations - origins[ray_idx]
        hit_normals = reconstruction.face_normals[tri_idx]
        angles = np.rad2deg(
            trimesh.geometry.vector_angle(
                np.stack((hit_normals, trimesh.unitize(ray_vecs)), axis=1)
            )
        )
        is_front_face_intersection = angles > 90
        locations = locations[~is_front_face_intersection]
        ray_idx = ray_idx[~is_front_face_intersection]
        tri_idx = tri_idx[~is_front_face_intersection]

        displacements = np.zeros_like(moving_surf.points)
        displacements[ray_idx] = locations - moving_surf.points[ray_idx]

        # #TODO: If an outward ray returned NaN, try the inward direction
        needs_flip = ~np.isin(np.arange(moving_surf.n_points), ray_idx)
        if both_directions and needs_flip.any():
            origins = origins[needs_flip]
            ray_directions = ray_directions[needs_flip]
            ray_directions *= -1
            tri_idx, ray_idx, locations = reconstruction.ray.intersects_id(
                ray_origins=origins,
                ray_directions=ray_directions,
                return_locations=True,
                multiple_hits=False,
            )
            # check that all rays intersected front_faces
            ray_vecs = locations - origins[ray_idx]
            hit_normals = reconstruction.face_normals[tri_idx]
            angles = np.rad2deg(
                trimesh.geometry.vector_angle(
                    np.stack((hit_normals, trimesh.unitize(ray_vecs)), axis=1)
                )
            )
            is_front_face_intersection = angles > 90
            locations = locations[is_front_face_intersection]
            ray_idx = ray_idx[is_front_face_intersection]
            tri_idx = tri_idx[is_front_face_intersection]

            flipped_disp = displacements[needs_flip]
            flipped_disp[ray_idx] = locations - origins[ray_idx]
            displacements[needs_flip] = flipped_disp
        is_too_far = trimesh.util.row_norm(displacements) > max_node_shift
        if is_too_far.any():
            displacements[is_too_far] = np.zeros(
                3
            )  # TODO: Check if this should be set to the max
        new_mesh.point_data["displacement_vecs"] = np.zeros_like(new_mesh.points)
        new_mesh.point_data["displacement_vecs"][moving_surf.point_data[pid_str]] = (
            displacements
        )

        # Plots for debugging
        if kwargs.get("debug", False):
            # new_mesh.point_data["displacement_vecs"] *= 20
            new_mesh.set_active_vectors("displacement_vecs")
            p = pv.Plotter()
            p.add_mesh(new_mesh.wrapped_reconstruction, opacity=0.4)
            new_mesh.plot_with_arrows(external_plotter=p)
            mb = pv.MultiBlock(
                [
                    surf.extract_points(
                        is_frozen_pids, include_cells=False, adjacent_cells=False
                    ),
                    surf.extract_points(
                        is_mobile_pids, include_cells=False, adjacent_cells=False
                    ),
                ]
            )

            p = pv.Plotter()
            for b, lgnd in zip(mb, ["frozen", "mobile"]):
                p.add_mesh(b, show_edges=True, color=np.random.rand(3), label=lgnd)
            p.add_mesh(surf, style="wireframe")
            p.add_mesh(first_mesh, opacity=0.4, color="r", label="first")
            p.add_mesh(last_mesh, opacity=0.4, color="b", label="last")
            p.add_axes()
            p.add_legend()
            p.show()
            # new_mesh.point_data["displacement_vecs"] /= 20


        new_mesh.point_data["displacement_vecs"] = new_mesh.smooth_hexahedral_cells(
            points=new_mesh.point_data["displacement_vecs"],
            cell_def=new_mesh.cell_def,
            smoothing_operation="total",
        )
        new_mesh.points += new_mesh.point_data["displacement_vecs"]
        jacobian = new_mesh.scaled_jacobian_determinants
        is_suboptimal = jacobian < 0
        indexes = np.unique(new_mesh.cell_def[is_suboptimal])
        new_mesh.points[indexes] -= new_mesh.point_data["displacement_vecs"][indexes]


        if inplace:
            self.points = new_mesh.point_locs.copy()
            return self
        else:
            del new_mesh.point_data["displacement_vecs"]
            return new_mesh

    def optimize_iteratively(
        self,
        max_iters: int = 15,
        include_expand: bool = True,
        max_point_shift: float = 5,
        upper_jacobian_limit: float = 0.5,
        include_smooth: bool = True,
        inplace: bool = False,
    ) -> Self:
        """Optimize a CartilageMesh iteratively, until a minimum scaled Jacobian is met.

        Args:
            max_iters: The maximum number of iterations allowed. Most meshes will finish
                within 5. Defaults to 20.
            include_expand: Include a call to expand_laminar_edges during the
                optimization. Should be included if smoothing. Defaults to True.
            max_point_shift: Maximum distance a point may be shifted during expansion.
                Meaningless if inclde_expand is False. Defaults to 2.
            upper_jacobian_limit: The minimum scaled Jacobian target. Optimization
                proceeds until every cell greater than or equal to this value. Defaults
                to 0.5.
            include_smooth: Include a mesh smoothing step in each iteration. To combat
                mesh shrinkage, set include_expand to True. Defaults to True.
            inplace: Whether the mesh should be updated in place, or if a copy should be
                returned. Defaults to False.

        Raises:
            OptimizationError: when optimization loop has been performed max_iters
                times.

        Returns:
            The optimized mesh.

        """
        new_mesh = self if inplace else self.copy()
        # TODO: Replace this with a method call
        s = new_mesh.scaled_jacobian
        is_bad_jacobian = s < upper_jacobian_limit
        loop_counter = 0
        with WorkProgress(desc="Optimizing meniscus") as pbar:
            while loop_counter < max_iters and is_bad_jacobian.any():
                if include_smooth:
                    new_mesh.points = new_mesh.smooth_hexahedral_cells(
                        points=new_mesh.points,
                        cell_def=new_mesh.cell_def,
                        smoothing_operation="meniscus",
                    )
                if include_expand:
                    new_mesh.expand_profile(
                        max_node_shift=max_point_shift,
                        inplace=True,
                        both_directions=False,
                        expand_tips=True,
                    )
                    new_mesh.points = new_mesh.smooth_hexahedral_cells(
                        points=new_mesh.points,
                        cell_def=new_mesh.cell_def,
                        smoothing_operation="total",
                    )
                new_mesh.optimize_scaled_jacobians(
                    target_jacobian=upper_jacobian_limit,
                    inplace=True,
                    in_plane_only=True,
                )
                mesh_copy = new_mesh.copy()
                front_faces = mesh_copy.cell_def[:, :4]
                back_faces = mesh_copy.cell_def[:, 4:]
                needs_delete = np.isin(back_faces, front_faces).any(axis=1) | np.isin(
                    front_faces, back_faces
                ).any(axis=1)
                mesh_copy.update_cell_def(mesh_copy.cell_def[~needs_delete])

                # TODO: Replace this with a method call
                s = mesh_copy.scaled_jacobian
                is_bad_jacobian = s < upper_jacobian_limit
                loop_counter += 1
                pbar.update(1)

            front_faces = new_mesh.cell_def[:, :4]
            back_faces = new_mesh.cell_def[:, 4:]
            needs_delete = np.isin(back_faces, front_faces).any(axis=1) | np.isin(
                front_faces, back_faces
            ).any(axis=1)
            new_mesh.update_cell_def(new_mesh.cell_def[~needs_delete])
            new_mesh.update_cell_def(new_mesh.cell_def[~needs_delete])

        try:
            if loop_counter >= max_iters:
                raise OptimizationError(
                    f"Mesh optimization did not finish after {max_iters} iterations."
                )
            return new_mesh
        except OptimizationError:
            del new_mesh
            raise

    @stage("Smoothing meniscus slices")
    def smooth_by_slice(
        self, n_iter: int = 200, boundary_smoothing: bool = False, inplace: bool = False
    ) -> Self:
        # loop through the slices
        new_mesh = self.copy()
        pid_str, cid_str = new_mesh._pc_id_strs
        layer_cids = new_mesh.get_layer_cids(np.arange(new_mesh.n_layers))

        # Get the third cell
        center_cids = layer_cids[:, 3]

        # Get the center (front face, 1) pid
        center_pids = new_mesh.cell_def[center_cids, 1]

        # Handle the final (back face, 5) pid
        back_face_center_point = new_mesh.cell_def[center_cids[-1], 5]
        center_pids = np.hstack((center_pids, back_face_center_point))
        center_locs = new_mesh.point_locs[center_pids]

        # get the front face pids
        face_pids = new_mesh.cell_def[layer_cids, :4]

        # handle the last face
        last_face_pids = new_mesh.cell_def[layer_cids[-1], 4:]
        face_pids = np.stack((*face_pids, last_face_pids))

        # get the surface
        for slice_index, (layer_cid, face_pid) in enumerate(
            zip(layer_cids, face_pids)
        ):
            layer = new_mesh.extract_cells(layer_cid).extract_surface(algorithm=None)
            layer_pids = layer.point_data[pid_str]
            needs_smooth = np.isin(layer_pids, face_pid)
            face = layer.extract_points(
                needs_smooth, adjacent_cells=False
            ).extract_surface(algorithm=None)
            face_2 = face.copy()
            face_2.smooth_taubin(
                n_iter=n_iter,
                inplace=True,
                boundary_smoothing=boundary_smoothing,
                normalize_coordinates=True,
            )
            new_mesh.points[face_2.point_data[pid_str]] = face_2.points.copy()
            report(
                "Smoothing meniscus slices", slice_index + 1, len(layer_cids),
                unit="slices",
            )

        if inplace:
            self.points = new_mesh.point_locs.copy()
            return self
        else:
            return new_mesh

    @stage("Creating horn node sets")
    def create_horn_node_set(self, inplace: bool = False, **kwargs):
        new_mesh = self.copy() if inplace else self
        pid_str, cid_str = new_mesh._pc_id_strs
        layer_cids = new_mesh.get_layer_cids(np.arange(new_mesh.n_layers))

        # Get the third cell
        center_cids = layer_cids[:, 3]

        # Get the center (front face, 1) pid
        center_pids = new_mesh.cell_def[center_cids, 1]

        # Handle the final (back face, 5) pid
        back_face_center_point = new_mesh.cell_def[center_cids[-1], 5]
        center_pids = np.hstack((center_pids, back_face_center_point))
        center_locs = new_mesh.point_locs[center_pids]

        # get the front face pids
        face_pids = new_mesh.cell_def[layer_cids, :4]

        # handle the last face
        last_face_pids = new_mesh.cell_def[layer_cids[-1], 4:]
        face_pids = np.stack((*face_pids, last_face_pids))

        # Create the PID sets
        anterior_horn_pids = np.unique(last_face_pids)
        posterior_horn_pids = np.unique(face_pids[0])

        new_mesh.field_data["anterior_horn_pids"] = anterior_horn_pids
        new_mesh.field_data["posterior_horn_pids"] = posterior_horn_pids

        if kwargs.get("debug", False):
            plotter = pv.Plotter()
            plotter.add_mesh(new_mesh.extract_surface(algorithm=None), opacity=0.5)
            anterior_horn = new_mesh.extract_points(
                anterior_horn_pids, include_cells=False
            )
            posterior_horn = new_mesh.extract_points(
                posterior_horn_pids, include_cells=False
            )
            plotter.add_mesh(
                anterior_horn, color="r", point_size=10, label="Anterior Horn"
            )
            plotter.add_mesh(
                posterior_horn, color="b", point_size=10, label="Posterior Horn"
            )
            plotter.add_legend()
            plotter.add_point_labels(
                new_mesh.points[anterior_horn_pids], anterior_horn_pids
            )
            plotter.add_point_labels(
                new_mesh.points[posterior_horn_pids], posterior_horn_pids
            )
            plotter.add_axes()
            plotter.show()

        return new_mesh
