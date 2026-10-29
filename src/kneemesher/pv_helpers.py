"""This module contains helper functions for working with PyVista objects."""

from __future__ import annotations

import functools
import inspect
import logging
from contextlib import contextmanager
from typing import Dict, List, Self, Tuple, Union

from kneemesher.metadata import TissueType  # noqa

logger = logging.getLogger(__name__)

import numpy as np
import pyvista as pv
import vtk
from numpy.typing import ArrayLike
from scipy import spatial
from trimesh import geometry as tm_geometry
from trimesh.transformations import affine_matrix_from_points
from trimesh.util import row_norm, unitize
from vtk import VTK_HEXAHEDRON, VTK_QUAD  # noqa

from kneemesher.errors import GridError
from kneemesher.mesh_repair import RepairBackend, repair_mesh
from kneemesher.utilities import trimesh_wrap
from kneemesher.utilities._utilities import apply_cpd_registration


@contextmanager
def pv_wrapper_context(package: str = "pyvista"):
    """Context manager to temporarily replace PyVista wrappers."""
    import pyvista as pv

    current_polydata = pv._wrappers["vtkPolyData"]
    current_ugrid = pv._wrappers["vtkUnstructuredGrid"]
    try:
        if package == "kneemesher":
            pv._wrappers["vtkPolyData"] = PolyData
            pv._wrappers["vtkUnstructuredGrid"] = pv.UnstructuredGrid
        elif package == "pyvista":
            pv._wrappers["vtkPolyData"] = pv.PolyData
            pv._wrappers["vtkUnstructuredGrid"] = pv.UnstructuredGrid
        yield
    finally:
        pv._wrappers["vtkPolyData"] = current_polydata
        pv._wrappers["vtkUnstructuredGrid"] = current_ugrid


def fit_plane_to_points(*args, **kwargs):
    """Fit a plane to points using PyVista with wrapper handling."""
    import pyvista as pv

    with pv_wrapper_context():
        return pv.fit_plane_to_points(*args, **kwargs)


def handle_inplace(method):
    """Decorate methods to handle inplace logic."""

    def wrapper(self, *args, inplace=False, **kwargs):
        target = self if inplace else self.copy()
        result = method(target, *args, inplace=inplace, **kwargs)
        return target if inplace else result

    return wrapper


def polyhull(poly: pv.PolyData, is_line: bool = False, **kwargs) -> pv.PolyData:
    """Return the convex hull of a PolyData object.

    Args:
        poly: The input PolyData object.
        is_line: Whether the PolyData object represents a line. Defaults to False.
        **kwargs: Additional keyword arguments to pass to
            `pyvista.PolyDataFilters.delaunay_3d`.

    Returns:
        The convex hull of the input PolyData object.

    Raises:
        None.

    """
    poly = poly.copy()
    if is_line:
        # Get the plane of best fit for the slice
        _, _, normal = pv.fit_plane_to_points(poly.points, return_meta=True)

        # Find the transformation that will rotate the plane of best fit to the xy plane
        # of the coordinate system
        rot_matrix = tm_geometry.align_vectors(normal, [0, 0, 1])
        rotated_slice = poly.transform(rot_matrix, inplace=False)
        hull = spatial.ConvexHull(rotated_slice.points[:, :2])
        lines = np.column_stack(
            (2 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        points = hull.points
        points = np.hstack(
            (points, rotated_slice.points[:, -1].mean() * np.ones((points.shape[0], 1)))
        )
        hull_poly = pv.PolyData(points, lines=lines)
        hull_poly = hull_poly.transform(rot_matrix.T, inplace=False)
    else:
        hull = spatial.ConvexHull(poly.points)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        hull_poly = pv.PolyData(hull.points, faces)
    return hull_poly


def ugrid_to_polydata(ugrid, *args, **kwargs):
    """Convert an unstructured grid to polydata.

    Args:
        ugrid: The input unstructured grid.
        *args: Additional positional arguments to be passed to
            `UnstructuredGrid.extract_surface` method.
        **kwargs: Additional keyword arguments to be passed to
            `UnstructuredGrid.aextract_surface` method.

    Returns:
        surface: The converted polydata.

    Raises:
        GridError: If the input unstructured grid contains multiple poly surfaces.

    """
    kwargs.setdefault("algorithm", None)
    surface = ugrid.extract_surface(*args, **kwargs)
    if surface.n_cells != ugrid.n_cells:
        raise GridError("Can only perform with a single poly surface.")

    # iterate through each cell type
    faces = np.array((), dtype=int)
    for ct in ugrid.cells_dict.keys():
        i_cell_def = ugrid.cells_dict[ct].copy()
        n_verts_per_cell = i_cell_def.shape[1]
        # pad all the rows with number of vertices per cell
        i_faces = np.pad(
            i_cell_def,
            ((0, 0), (1, 0)),
            mode="constant",
            constant_values=n_verts_per_cell,
        ).reshape(-1)

    faces = np.hstack((faces, i_faces))
    surface = PolyData(ugrid.points.copy(), faces=faces)
    surface.mesh_type = ugrid.mesh_type
    surface.label = ugrid.label

    # Ensure that the data arrays are still present
    for key in ugrid.point_data.keys():
        surface.point_data[key] = ugrid.point_data[key]
    for key in ugrid.cell_data.keys():
        surface.cell_data[key] = ugrid.cell_data[key]
    for key in ugrid.field_data.keys():
        surface.field_data[key] = ugrid.field_data[key]
    return surface


def split_connected_line_into_segments(line: pv.PolyData) -> pv.PolyData:
    """Split a connected line into individual line segments.

    Args:
        line: A PolyData object representing a connected line.

    Returns:
        A PolyData object with each segment of the line represented as a separate line.

    Raises:
        None.

    """
    if isinstance(line, pv.MultiBlock):
        return pv.MultiBlock([split_connected_line_into_segments(ln) for ln in line])

    # Get the connectivity of the line
    line_connectivity = line.lines.copy()[1:]

    # Split the line into two point segments, where the terminal point of the previous
    # segment becomes the initial point of the next segment
    new_connectivity = np.stack((line_connectivity, np.roll(line_connectivity, -1))).T
    new_connectivity = new_connectivity[:-1]

    # Prepend every row of new_connectivity with a 2 and flatten the array
    new_connectivity = np.insert(new_connectivity, 0, 2, axis=1).flatten()

    # Assign the new connectivity to the line and return it
    line.lines = new_connectivity
    return line


def sort_line_ordering(line: pv.PolyData, **kwargs) -> pv.PolyData:
    """Sort the line ordering of a PolyData object.

    Args:
        line: A PolyData object representing a line.
        **kwargs: Additional keyword arguments.

    Returns:
        A PolyData object with sorted line ordering.

    Raises:
        AssertionError: If the line is not a single, connected line.

    """
    conn = line.connectivity("all")
    multi = pv.MultiBlock()
    for rid in np.unique(conn.point_data["RegionId"]):
        multi.append(conn.connectivity("specified", region_ids=rid))

    if multi.n_blocks > 1:
        for m in multi:
            del m.point_data["RegionId"]
            del m.cell_data["RegionId"]
        line = multi
    if isinstance(line, pv.MultiBlock):
        return pv.MultiBlock([sort_line_ordering(ln, **kwargs) for ln in line])

    last_segment = None

    line_connectivity = line.lines.reshape(-1, 3)[:, 1:]
    new_connectivity = np.full_like(line_connectivity, 0, dtype=int)

    # Get the unique point indices and their counts from line_connectivity
    unique_pids, counts = np.unique(line_connectivity, return_counts=True)

    # Get the indices of the points that are only used once
    unique_pids_used_once = unique_pids[counts == 1]

    # If there are no points that are only used once, save the last segment to it's own
    # variable and remove it from line_connectivity
    if len(unique_pids_used_once) == 0:
        last_segment = line_connectivity[-1]
        line_connectivity = line_connectivity[:-1]
        new_connectivity = new_connectivity[:-1]
        unique_pids, counts = np.unique(line_connectivity, return_counts=True)
        unique_pids_used_once = unique_pids[counts == 1]

    # Confirm that the line is a single, connected line
    assert len(unique_pids_used_once) == 2

    # Get the index of the first point
    first_pid = unique_pids_used_once[0]

    # Get the next line segment connected to the first point
    next_segment = line_connectivity[np.any(line_connectivity == first_pid, axis=1)]

    # Get the second pid from the segment
    second_pid = next_segment[next_segment != first_pid][0]

    # Update the first segment in new_connectivity
    new_connectivity[0] = [first_pid, second_pid]

    # Update the first point
    first_pid = second_pid

    for line_idx, _ in enumerate(new_connectivity):
        if line_idx == 0:
            continue
        # Get the next line segment connected to the first point
        next_segment = line_connectivity[np.any(line_connectivity == first_pid, axis=1)]

        # Get the second pid from the segment using np.isin
        second_pid = next_segment[
            ~np.isin(next_segment, new_connectivity[line_idx - 1])
        ][0]

        # Update the current segment in new_connectivity
        new_connectivity[line_idx] = [first_pid, second_pid]

        # Update the first point
        first_pid = second_pid

    # If there is a last segment, insert it at the end of new_connectivity
    if last_segment is not None:
        # Find the row and column containing the first pid of the last segment
        first_pid, second_pid = last_segment
        row, col = np.unravel_index(
            np.flatnonzero(line_connectivity == first_pid), line_connectivity.shape
        )
        # If the column is 0, reverse the segment
        if col == 0:
            last_segment = last_segment[::-1]

        # Insert the last segment after row in the new_connectivity array
        new_connectivity = np.insert(new_connectivity, row + 1, last_segment, axis=0)

    # Prepend each row in new_connectivity with the number of points in each line
    # segment
    sorted_point_idxs = new_connectivity.copy().flatten()
    sorted_point_idxs = sorted_point_idxs.flatten()
    sorted_point_idxs = np.hstack(
        (sorted_point_idxs[0], sorted_point_idxs[1:-1:2], sorted_point_idxs[-1])
    )

    new_connectivity = np.insert(new_connectivity, 0, 2, axis=1)

    # Update the line with the new connectivity
    line.lines = new_connectivity
    line.field_data["sorted_points"] = line.points[sorted_point_idxs]
    line.field_data["sorted_pids"] = sorted_point_idxs

    return line


def make_line_continuous(line: pv.PolyData, **kwargs) -> pv.PolyData:
    """Make a series of connected line segments into a single line.

    Given a PolyData object representing a line, sorts the points in the line
    in a continuous order and returns a new PolyData object with the sorted
    points.

    Args:
        line: A PolyData object representing a line.
        **kwargs: Additional keyword arguments.
            debug: If True, plot the line with the arc_length scalars.

    Returns:
        A new PolyData object with the sorted points.

    Raises:
        None.

    """
    line = sort_line_ordering(line)

    if isinstance(line, pv.MultiBlock):
        return pv.MultiBlock([make_line_continuous(ln, **kwargs) for ln in line])

    new_connectivity = line.lines.reshape(-1, 3)[:, 1:]

    # Prepend new_connectivity with the number of points
    flattened = new_connectivity.flatten()
    new_connectivity = np.hstack((flattened[0], flattened[1:-1:2], flattened[-1]))
    new_connectivity = np.insert(new_connectivity, 0, new_connectivity.size)

    # Update the line with the new connectivity
    line.lines = new_connectivity
    if kwargs.get("debug", False):
        line.compute_arc_length().plot(scalars="arc_length")

    line.field_data["sorted_points"] = line.points[new_connectivity[1:]]
    return line


def _with_correct_original_ids(method) -> pv.DataSet:
    """Perform method without reordering 'vtkOriginalPointIds' or '...CellIds'."""

    @functools.wraps(method)
    def wrapper(pv_obj: pv.DataSet, *args, **kwargs) -> pv.DataSet:
        orig_pids = None
        orig_cids = None
        if "vtkOriginalPointIds" in pv_obj.point_data.keys():
            orig_pids = _by_value(pv_obj.point_data["vtkOriginalPointIds"])
        if "vtkOriginalCellIds" in pv_obj.cell_data.keys():
            orig_cids = _by_value(pv_obj.cell_data["vtkOriginalCellIds"])
        method_return = method(pv_obj, *args, **kwargs)
        if orig_pids is not None:
            pv_obj["vtkOriginalPointIds"] = orig_pids
        if orig_cids is not None:
            pv_obj["vtkOriginalCellIds"] = orig_cids
        return method_return

    return wrapper


@_with_correct_original_ids
def extract_points(pv_obj: pv.DataSet, *args, **kwargs):
    """Extract points from a pyvista object without side effects."""
    return pv_obj.extract_points(*args, **kwargs)


def _by_value(*pv_arrays: pv.pyvista_ndarray) -> np.ndarray:
    """Return a copy of a pyvista array not referencing the same object in memory."""
    corrected = [np.array(x, copy=True) for x in pv_arrays]
    return tuple(corrected) if len(corrected) > 1 else corrected[0]


def without_activation(method):
    """Decorate methods to keep the same active arrays."""

    @functools.wraps(method)
    def wrapper_without_activation(pv_obj, *args, **kwargs):
        active_scalars = pv_obj.active_scalars_name
        active_vectors = pv_obj.active_vectors_name
        method_return = method(pv_obj, *args, **kwargs)
        pv_obj.set_active_scalars(active_scalars)
        pv_obj.set_active_vectors(active_vectors)
        if isinstance(method_return, type(pv_obj)):
            try:
                method_return.set_active_scalars(active_scalars)
                method_return.set_active_vectors(active_vectors)
            except KeyError:
                pass
        return method_return

    return wrapper_without_activation


def is_called_directly():
    """Check if the current function is called directly."""
    return inspect.stack()[1].function == "<module>"


def _log_transform_operation(method):
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Use an instance attribute to track the counter
        if not hasattr(self, "_transform_operation_counter"):
            if int(pv.__version__.split('.')[1]) < 46:
                self._transform_operation_counter = 0
            else:
                pv.set_new_attribute(self, "_transform_operation_counter", 0)

        self._transform_operation_counter += 1  # Increment the counter
        args = tuple(  # Fix for pv.Transform being unpicklable
            [arg.matrix if isinstance(arg, pv.Transform) else arg for arg in args]
        )
        op_detail = {"operation": method.__name__, "args": args, "kwargs": kwargs}
        inplace = kwargs.pop("inplace", False)
        result = method(self, *args, inplace=inplace, **kwargs)
        if (
            self._transform_operation_counter == 1
        ):  # Trigger only if it's the first time the wrapper is run
            result.transform_stack.append(op_detail)
            result._transform_index += 1
        self._transform_operation_counter -= 1  # Decrement the counter

        # Remove the attribute if the counter reaches zero
        if self._transform_operation_counter == 0:
            del self._transform_operation_counter

        return result

    return wrapper


def apply_log_transform_decorator(cls):
    """Apply the _log_transform_operation decorator to specific methods in a class."""
    methods_to_decorate = [
        "transform",
        "scale",
        "translate",
        "rotate_vector",
        "rotate_x",
        "rotate_y",
        "rotate_z",
    ]
    for method_name in methods_to_decorate:
        if hasattr(cls, method_name):
            original_method = getattr(cls, method_name)
            decorated_method = _log_transform_operation(original_method)
            setattr(cls, method_name, decorated_method)
    return cls


@pv.abstract_class
class HierarchicalMixin:
    def __init__(self, *args, **kwargs):
        """Initialize the HierarchicalMixin class.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        """
        self._parent: Self | None = None
        self._children: List[Self] = []
        super().__init__(*args, **kwargs)

    @property
    def parent(self) -> Self | None:
        """Return the parent of the current object."""
        return self._parent

    @parent.setter
    def parent(self, parent: Self | None):
        """Set the parent of the current object.

        Args:
            parent (HierarchicalMixin | None): The parent object to set.

        """
        if (
            self._parent is not None
            and parent is not None
            and self._parent is not parent
        ):
            raise ValueError("This object already has a parent.")
        if self._parent is not None:
            self._parent._children.remove(self)
        self._parent = parent
        if parent is not None:
            parent._children.append(self)

    @property
    def children(self) -> List[Self]:
        """Return the children of the current object."""
        return self._children

    @children.setter
    def children(self, children: List[Self]):
        """Set the children of the current object.

        Args:
            children (List[HierarchicalMixin]): The list of child objects to set.

        Raises:
            ValueError: If any child already has a parent.

        """
        children = [child for child in children if child is not None]
        for child in children:
            if child.parent is not None and child.parent is not self:
                raise ValueError("Child already has a parent.")
            child.parent = self
        self._children = children

    def add_child(self, child: Self):
        """Add a child to the current object.

        Args:
            child (HierarchicalMixin): The child object to add.

        """
        if not isinstance(child, Self):
            raise TypeError("Child must be an instance of HierarchicalMixin.")
        if child.parent is not None:
            raise ValueError("Child already has a parent.")
        self._children.append(child)
        child.parent = self

    def remove_child(self, child: Self):
        """Remove a child from the current object.

        Args:
            child (HierarchicalMixin): The child object to remove.

        Raises:
            ValueError: If the child is not a child of the current object.

        """
        if child not in self._children:
            raise ValueError("Child is not a child of this object.")
        self._children.remove(child)
        child.parent = None



@pv.abstract_class
class MetaDataMixin:
    def __init__(
        self,
        *args,
        label: str = None,
        color=None,
        mesh_type: TissueType | None = None,
        **kwargs,
    ):
        """Initialize the MetaDataMixin class.

        Args:
            *args: Positional arguments.
            label (str): A label for the object. Defaults to None.
            color (str): The color of the object. Defaults to None.
            **kwargs: Keyword arguments.

        """
        self.label = label
        self.color = color
        self.mesh_type = mesh_type
        self.tissue_landmarks = None
        super().__init__(*args, **kwargs)


@pv.abstract_class
# @apply_log_transform_decorator
class TransformTrackingDataSet:
    def __init__(self, *args, **kwargs):
        self.transform_stack: List[float] = []
        self.transform_dict: Dict[str, np.ndarray] = {}
        self._transform_matrix = np.eye(4)
        self._transform_index = -1
        super().__init__(*args, **kwargs)

    def scale_and_align(
        self,
        target_dataset: pv.DataSet,
        *transform_args,
        voxel_size: float | None = 5,
        use_cuda: bool = False,
        return_matrix: bool = False,
        max_rotation_angle: float | None = None,
        **transform_kwargs,
    ) -> Union[pv.DataSet, Tuple[pv.DataSet, np.ndarray]]:
        """Scale and align the current dataset to a target using CPD registration.

        This method performs Coherent Point Drift (CPD) registration to align the points
        of the current dataset (`self`) with the points of a target dataset. Optionally,
        it can return the transformation matrix used for alignment.

        Args:
            target_dataset (pv.DataSet): The target dataset to align the current dataset to.
            *args: Additional positional arguments passed to the `transform` method.
            voxel_size (float | None, optional): The voxel size for downsampling during
                registration. Defaults to 5.
            use_cuda (bool, optional): Whether to use CUDA for acceleration during
                registration. Defaults to False.
            return_matrix (bool, optional): Whether to return the transformation matrix
                along with the transformed dataset. Defaults to False.
            max_rotation_angle (float | None, optional): The maximum allowed rotation
                angle in degrees. If the computed rotation exceeds this angle, the
                alignment is performed using rigid ICP instead. Defaults to None.
            **kwargs: Additional keyword arguments passed to the `transform` method.

        Returns:
            Union[pv.DataSet, Tuple[pv.DataSet, np.ndarray]]: The transformed dataset.
            If `return_matrix` is True, a tuple containing the transformed dataset and
            the transformation matrix is returned.

            The transformation is composed of scaling, rotation, and translation
            components (in that order).

        """
        source_points = self.points.copy()
        target_points = target_dataset.points.copy()

        _, trans = apply_cpd_registration(
            source_points,
            target_points,
            voxel_size=voxel_size,
            use_cuda=use_cuda,
            return_transformation=True,
        )
        if max_rotation_angle is not None:
            rotation = trans.as_rotation("rotvec")
            rotation_angle = np.linalg.norm(rotation)
            if rotation_angle > np.radians(max_rotation_angle):
                logger.warning(
                    f"{self.label}: Rotation angle {np.degrees(rotation_angle):.2f} exceeds "
                    f"maximum allowed rotation angle of {max_rotation_angle:.2f} degrees."
                )
                _, trans = self.align(target_dataset, return_matrix=True)
        if return_matrix:
            return self.transform(trans, *transform_args, **transform_kwargs), trans
        else:
            return self.transform(trans, *transform_args, **transform_kwargs)

    @property
    def tracking_original(self):
        if (
            "_original_points_x" in self.point_data
            and "_original_points_y" in self.point_data
            and "_original_points_z" in self.point_data
        ):
            return True

    @tracking_original.setter
    def tracking_original(self, value: bool):
        """Set the tracking_original property."""
        if value:
            self._capture_original_points()
        else:
            self.stop_tracking_original_mesh()

    def _copy_mesh_data_set_simple_attributes(self, other_mesh):
        self.tracking_original = other_mesh._track_original
        self.original_mesh = other_mesh.original_mesh
        self.transform_dict = other_mesh.transform_dict
        self.label = other_mesh.label
        self.default_color = other_mesh.default_color
        self._transform_matrix = other_mesh._transform_matrix

    @property
    def original_mesh(self):
        if self.tracking_original:
            original_mesh = self.copy()
            original_mesh.points = self.original_points.copy()
            return original_mesh

    @original_mesh.setter
    def original_mesh(self, mesh):
        """Raise an error explaining that the original mesh cannot be set."""
        raise AttributeError(
            "The original mesh cannot be set directly. Use the `reset_original_mesh` method."
        )

    def reset_original_mesh(self):
        self._capture_original_points()

    def stop_tracking_original_mesh(self):
        """Stop tracking the original mesh."""
        if self.tracking_original:
            self.tracking_original = False
            del self.point_data["_original_points_x"]
            del self.point_data["_original_points_y"]
            del self.point_data["_original_points_z"]

    @property
    def original_points(self):
        """Retrieve the original points from the point data.

        This method extracts the original x, y, and z coordinates stored in the
        point data and combines them into a NumPy array.

        Returns:
            numpy.ndarray: A 2D array where each row represents a point with
            its x, y, and z coordinates.

        """
        original_point_array = np.array(
            [
                self.point_data["_original_points_x"],
                self.point_data["_original_points_y"],
                self.point_data["_original_points_z"],
            ]
        ).T
        return original_point_array

    @original_points.setter
    def original_points(self, points: np.ndarray):
        """Raise an error explaining that the original points cannot be set."""
        raise AttributeError(
            "The original points cannot be set directly. Use the `reset_original_mesh` method."
        )

    def _capture_original_points(self):
        self.point_data.set_array(self.points[:, 0], "_original_points_x")
        self.point_data.set_array(self.points[:, 1], "_original_points_y")
        self.point_data.set_array(self.points[:, 2], "_original_points_z")


    def __enter__(self):
        """Enter the runtime context related to this object.

        Can be used for temporary mesh copies in a `with` statement, to conserve memory.

        Returns:
            self: The current object.

        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up any resources used.

        Args:
            exc_type (type): The type of the exception raised, if any.
            exc_val (Exception): The exception raised, if any.
            exc_tb (traceback): The traceback for the exception raised, if any.

        """
        self.__del__()

    @property
    def transform_matrix(self):
        """Return the transformation matrix of the object."""
        return self._transform_matrix

    @transform_matrix.setter
    def transform_matrix(self, matrix: np.ndarray):
        """Set the transformation matrix of the object."""
        self._transform_matrix = matrix

    def save_transform(self, name: str):
        """Save the current transformation matrix with a given name.

        This method computes the affine transformation matrix between the original
        mesh and the current mesh and saves it in the `transform_dict` attribute
        under the provided name.

        Args:
            name (str): The name to associate with the saved transformation.

        Raises:
            ValueError: If the original mesh is not set or the points do not match in shape.

        """
        if self.original_mesh and self.points.shape == self.original_mesh.points.shape:
            original_points = self.original_mesh.points
            current_points = self.points
            if self.n_points > 8000:
                # Create an index of 8000 unique points
                indices = np.random.choice(
                    np.arange(original_points.shape[0]), 8000, replace=False
                )
                original_points = original_points[indices].copy()
                current_points = current_points[indices].copy()
            transform = affine_matrix_from_points(original_points.T, current_points.T)
            # Save both the transform and the current index
            self.transform_dict[name] = {
                "transform": transform,
                "index": self._transform_index,
            }
        else:
            raise ValueError(
                "Original mesh is not set or the points do not match in shape."
            )

    def apply_saved_transform(self, name: str, inplace: bool = True):
        """Apply a saved transformation from the transform dictionary."""
        if inplace:
            target_mesh = self
        else:
            target_mesh = self.copy()

        target_mesh.points = target_mesh.original_points.copy()

        if name in target_mesh.transform_dict:
            saved_data = target_mesh.transform_dict[name]
            transform = saved_data["transform"]
            target_mesh._transform_index = saved_data["index"]  # Restore index
            target_mesh.transform(transform, inplace=True)
        else:
            raise ValueError(f"Transformation '{name}' not found in transform stack.")
        return target_mesh if not inplace else None

    def reset_transforms(self, inplace: bool = True):
        """Reset the mesh to its original state."""
        if self.original_mesh is not None:
            if inplace:
                self.points = self.original_points.copy()
                self.transform_stack = []
                self._transform_index = -1
            else:
                new_mesh = self.copy()
                new_mesh.points = self.original_points.copy()
                new_mesh.transform_stack = []
                new_mesh._transform_index = -1
                return new_mesh
        else:
            raise ValueError("No original mesh to reset to.")

    def warp_to(
        self,
        target_mesh: pv.DataSet,
        *warp_args,
        inplace: bool = False,
        show_plots: bool = False,
        perform_smoothing: bool = False,
        n_iter: int = 3,
        **warp_kwargs,
    ):
        warped = self.copy() if not inplace else self
        for idx in range(n_iter):
            _, closest_cell_points = target_mesh.find_closest_cell(
                warped.points, return_closest_point=True
            )
            warped.point_data.set_array(
                closest_cell_points - warped.points, "__warp_to_vectors"
            )
            warped = warped.warp_by_vector(
                "__warp_to_vectors",
                *warp_args,
                inplace=inplace,
                **warp_kwargs,
            )
            if perform_smoothing and idx < n_iter - 1:
                warped.smooth_taubin(n_iter=10, inplace=True)

        if show_plots:
            _, closest_cell_points = target_mesh.find_closest_cell(
                warped.points, return_closest_point=True
            )
            warped.point_data.set_array(
                closest_cell_points - warped.points, "__warp_to_vectors"
            )
            pv.MultiBlock([warped, target_mesh]).plot(
                scalars="__warp_to_vectors", multi_colors=False, opacity=0.5
            )

        if "__warp_to_vectors" in warped.point_data:
            del warped.point_data["__warp_to_vectors"]

        return warped


@apply_log_transform_decorator
class PolyData(
    HierarchicalMixin,
    MetaDataMixin,
    TransformTrackingDataSet,
    pv.PolyData,
):
    """A subclass of `pyvista.UnstructuredGrid` that adds additional functionality."""

    def __init__(self, *args, track_original_mesh: bool = False, **kwargs):  # noqa: D107
        super().__init__(*args, **kwargs)
        if track_original_mesh:
            self.tracking_original = True

    @property
    def line_lengths(self):
        """Return the lengths of the lines in the PolyData object."""
        if self.n_lines == 0:
            return np.array([])
        return row_norm(
            np.squeeze(np.diff(self.points[self.lines.reshape(-1, 3)[:, 1:]], axis=1))
        )

    def plot_curvature(self, curv_type: str = "mean", **kwargs):
        """Plot the curvature of the PolyData object."""
        if curv_type == "absolute":
            maximum = self.curvature("maximum")
            minimum = self.curvature("minimum")
            curvature = np.abs(maximum) + np.abs(minimum)
            copied = self.copy()
            copied["curvature"] = curvature
            return copied.plot(scalars="curvature", **kwargs)
        else:
            return super().plot_curvature(curv_type=curv_type, **kwargs)

    def slice_closed_surface(
        self,
        normal="x",
        origin=None,
        tolerance: float = 1e-6,
        inplace: bool = False,
        progress_bar: bool = False,
        *,
        repair_backend: RepairBackend = "auto",
        **kwargs,
    ):
        """Slice a closed surface along a specified normal direction.

        Parameters
        ----------
        - normal (str): The normal direction along which to slice the surface. Default
        is "x".
        - tolerance (float): The tolerance value for determining if a point lies on the
        slicing plane. Default is 1e-6.
        - progress_bar (bool): If True, display a progress bar during the slicing
            process. Default is False.
        - repair_backend (str): Select the backend used to close an open input.
        - inplace (bool): If True, modify the original surface in-place. If False,
            create a new sliced surface. Default is False.
        - normal (str or tuple): The normal direction along which to slice the surface.
            Default is "x".
        - origin (None or tuple): The origin point of the slicing plane. Default is
            None.
        - **kwargs: Additional keyword arguments to be passed to the underlying slicing
            algorithm.

        Returns
        -------
        - extracted (vtkPolyData): The sliced surface as a vtkPolyData object.

        Raises
        ------
        - AssertionError: If the surface is not manifold (i.e., not a closed surface).
        - inplace (bool): If True, modify the original surface in-place. If False,
            create a new sliced surface. Default is False.
        - origin (None or tuple): The origin point of the slicing plane. Default is

        """
        if not self.is_manifold:
            self = PolyData(
                repair_mesh(self, backend=repair_backend, require_closed=True)
            )
        clipped = self.clip_closed_surface(
            normal=normal,
            origin=origin,
            tolerance=tolerance,
            inplace=False,
            progress_bar=progress_bar,
        )
        if clipped.n_cells == 0:
            return clipped
        clipped.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        cell_normals = clipped.cell_data["Normals"]
        repeated = np.repeat(np.atleast_2d(-normal), clipped.n_cells, axis=0)
        angles = tm_geometry.vector_angle(np.stack((cell_normals, repeated), axis=1))
        is_aligned = np.isclose(angles, 0, atol=1e-2)
        extracted = clipped.extract_cells(is_aligned).extract_surface(algorithm=None)
        # compute the distance of the extracted cells to the plane
        projected = PolyData(extracted).project_points_to_plane(
            normal=normal, origin=origin, inplace=False
        )
        distances = projected.points - extracted.points
        distances = row_norm(distances)
        is_on_plane = np.isclose(distances, 0)
        extracted = extracted.extract_points(
            is_on_plane, include_cells=True, adjacent_cells=False
        ).extract_surface(algorithm=None)
        return extracted


@apply_log_transform_decorator
class UnstructuredGrid(
    HierarchicalMixin,
    MetaDataMixin,
    TransformTrackingDataSet,
    pv.UnstructuredGrid,
):
    """A subclass of `pyvista.UnstructuredGrid` that adds additional functionality.

    Args:
        *args: Positional arguments to pass to `pyvista.UnstructuredGrid`.
        label (str): A label for the mesh. Defaults to "mesh".
        color (str or tuple): The default color for the mesh. Defaults to "w".
        **kwargs: Keyword arguments to pass to `pyvista.UnstructuredGrid`.

    Attributes:
        label (str): A label for the mesh.
        default_color (str or tuple): The default color for the mesh.

    Methods:
        from_cell_def(points, cell_def): Create a mesh using an array of points and cell
            definitions.
        extract_points(ind,
            adjacent_cells=True,
            include_cells=True,
            progress_bar=False
            ): Extract points without overwriting vtkOriginalPointIds and ...CellIds.
        plot(*args, **kwargs): Plot the unstructured grid.
        add_mesh_kwargs(color=None, opacity=1, show_edges=True, style="surface",
            **kwargs): Return a dictionary of default `pyvista.Plotter.add_mesh`
            *arguments.
        plotter_with_mesh(*args, external_plotter=None, add_mesh_kwargs=None, **kwargs):
            Return a plotter with this `UnstructuredGrid` object added.
        plot_with_arrows(*args, arrow_colors="r", active_vector_name=None,
            external_plotter=None, add_mesh_kwargs=None, auto_close=True, **kwargs):
            Plot the mesh with active arrows.
        get_cell_quality(quality_measure="scaled_jacobian", save_cell_data=False):
            Calculate cell quality values.

    """

    def __init__(self, *args, label: str = "mesh", color="w", **kwargs) -> None:
        """Initialize an upgraded unstructured grid for use in a hexahedral mesh.

        Args:
            *args: Variable length argument list.
            label (str): A label for the object. Defaults to "mesh".
            color (str): The default color for the object. Defaults to "w".
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        """
        kwargs["label"] = label
        kwargs["color"] = color
        super().__init__(*args, **kwargs)

    @classmethod
    def from_cell_def(
        UGrid,
        points: ArrayLike,
        cell_def: ArrayLike,
    ):
        """Create mesh using (n, 3) array of points and (m, 8) cell definitions.

        Args:
            UGrid: The UnstructuredGrid class.
            points: an (n, 3) array of point locations.
            cell_def: an (m, 8) array of hexahedral nodal connectivity definitions.
                Connections must trace an anticlockwise bottom face, followed by a
                matching anticlockwise top face.

        Returns:
            The initialized unstructured grid.

        """
        points, cell_def = _by_value(points, cell_def)
        cell_def = np.atleast_2d(cell_def)
        eights = np.full((cell_def.shape[0], 1), 8, dtype=cell_def.dtype)
        cell_type = np.full(eights.shape[0], VTK_HEXAHEDRON)
        cell_def = np.hstack((eights, cell_def))
        ug = UGrid(cell_def.ravel(), cell_type, points)
        ug.clear_data()
        return ug

    def to_poly_surface(self, *args, **kwargs):
        """Convert a surface UnstructuredGrid to a PolyData surface.

        The data arrays are preserved, as are the point and cell ordering.

        Args:
            *args: arguments passed to `pyvista.UnstructuredGrid.extract_surface`.
            **kwargs: keyword arguments passed to
                *`pyvista.UnstructuredGrid.extract_surface`.

        Returns:
            surface (pyvista.PolyData): The PolyData surface.

        Raises:
            GridError: If the number of cells in the extracted surface is not equal to
            the number of cells in the grid, which means this grid did not represent a
            surface mesh.

        """
        ugrid = self
        surface = ugrid_to_polydata(ugrid, *args, **kwargs)

        return surface

    @_with_correct_original_ids
    def extract_points(
        self,
        ind,
        adjacent_cells: bool = True,
        include_cells: bool = True,
        progress_bar: bool = False,
    ):
        """Extract points without overwriting vtkOriginalPointIds and ...CellIds."""
        return super().extract_points(
            ind,
            adjacent_cells=adjacent_cells,
            include_cells=include_cells,
            progress_bar=progress_bar,
        )

    @property
    def mesh(self):
        """Return self for backwards compatibility with HexMesh."""
        return self

    def show(self):
        """Alias of plot."""
        self.plot()

    def plot(
        self,
        *args,
        **kwargs,
    ):
        """Plot the unstructured grid."""
        kwargs.setdefault("show_edges", True)
        kwargs.setdefault("show_axes", True)

        return super().plot(
            *args,
            **kwargs,
        )

    def add_mesh_kwargs(
        self,
        color=None,
        opacity: float = 1,
        show_edges: bool = True,
        style: str = "surface",
        **kwargs,
    ):
        """Return a dictionary of default pyvista.Plotter.add_mesh arguments.

        Args:
            color: a string or RGB float array. Defaults to None.
            opacity: values closer to zero appear more transparent. Defaults to 1.
            show_edges: whether to show cell edges. Defaults to True.
            style: the type of plot, 'wireframe' is an option. Defaults to "surface".

        Returns:
            a dictionary of default pyvista.Plotter.add_mesh arguments for easy adding
            to other plotters.

        """
        color = self.default_color if color is None else color
        return dict(
            mesh=self,
            color=color,
            show_edges=show_edges,
            opacity=opacity,
            style=style,
            label=self.label,
            **kwargs,
        )

    def plotter_with_mesh(
        self,
        *args,
        external_plotter: pv.Plotter = None,
        add_mesh_kwargs: Dict = None,
        **kwargs,
    ):
        """Return a plotter with this UnstructuredGrid object added.

        Uses the outputs of add_mesh_kwargs when adding the object.
        """
        p = (
            pv.Plotter(*args, **kwargs)
            if external_plotter is None
            else external_plotter
        )
        p.add_axes()
        if add_mesh_kwargs is None:
            p.add_mesh(**self.add_mesh_kwargs())
        else:
            p.add_mesh(**add_mesh_kwargs)
        return p

    @without_activation
    def plot_with_arrows(
        self,
        *args,
        arrow_colors: str = "r",
        active_vector_name: str = None,
        external_plotter: pv.Plotter = None,
        add_mesh_kwargs: Dict = None,
        auto_close: bool = True,
        **kwargs,
    ):
        """Plot the mesh with active arrows.

        Args:
            arrow_colors: arrow color string. Defaults to "r".
            active_vector_name: choose the active vector name to plot. Defaults to None.
            external_plotter: an external plotter to add the mesh with arrows to. Defaults to None.
            add_mesh_kwargs: a dictionary of keyword arguments for adding the mesh to
                the plotter. Defaults to None.

        """
        p = self.plotter_with_mesh(
            *args,
            external_plotter=external_plotter,
            add_mesh_kwargs=add_mesh_kwargs,
            **kwargs,
        )
        if active_vector_name is not None:
            try:
                self.set_active_vectors(active_vector_name)
            except ValueError:
                pass
        if self.active_vectors_name is not None:
            p.add_mesh(self.arrows, color=arrow_colors)
        p.show(auto_close=auto_close)

    @without_activation
    def get_cell_quality(
        self, quality_measure: str = "scaled_jacobian", save_cell_data: bool = False
    ) -> np.ndarray:
        """Calculate cell quality values.

        Unsupported cell types will have an entry of -1.


        Args:
            quality_measure: String representing quality measure. Defaults to
                "scaled_jacobian".
            save_cell_data: Whether to save the quality measure to this object's cell
                data. If True, the cell_data name will be set to `quality_measure`.
                Defaults to False.

        Returns:
            The cell quality array.

        """
        grid_copy = self.cell_quality(quality_measure)
        quality = _by_value(grid_copy[quality_measure])
        if save_cell_data:
            self.cell_data[quality_measure] = quality
        return quality

    def find_low_quality_cells(
        self,
        metric_min: Union[float, int] = 0,
        metric_max: Union[float, int] = 1,
        *,
        quality_measure: str = "scaled_jacobian",
    ):
        """Find low quality cells.

        Args:
            metric_min: The minimal acceptable quality value. Defaults to 0.
            metric_max: The maximum acceptable quality value. Defaults to 1.
            quality_measure: The quality measure. Defaults to "scaled_jacobian".

        Returns:
            Cell IDs with quality outside of the [metric_min, metric_max] quality range.

        """
        quality = self.get_cell_quality(quality_measure)
        is_bad_cell = np.full_like(quality, False, dtype=bool)
        is_bad_cell[quality < metric_min] = True
        is_bad_cell[quality > metric_max] = True
        return np.flatnonzero(is_bad_cell)

    def plot_cell_quality(
        self, *args, quality_measure: str = "scaled_jacobian", **kwargs
    ) -> np.ndarray:
        """Plot cell quality measures.

        Extra arguments are passed to CartilageMesh.find_low_quality_cells method. If
        any low quality cells are found, the offending cells are emphasized.

        Args:
            quality_measure: The geometric cell quality measure. Defaults to
                "scaled_jacobian".

        Returns:
            The output of CartilageMesh.find_low_quality_cells.

        """
        bad_cids = self.find_low_quality_cells(*args, **kwargs)
        grid_copy = self.cell_quality(quality_measure)
        grid_copy[quality_measure] = grid_copy[quality_measure]
        grid_copy.set_active_scalars(quality_measure)

        opacity = 1 if not bad_cids.any() else 0.5
        p = pv.Plotter()
        p.add_mesh(grid_copy, show_edges=True, opacity=opacity)
        if bad_cids.any():
            bad_jac = grid_copy.extract_cells(bad_cids)
            p.add_mesh(bad_jac)
            p.add_mesh(bad_jac.extract_all_edges(), color="y", opacity=0.9)
            p.add_axes()
            p.show()
        return bad_cids

    def hausdorff_distance(self, compare_mesh, **hd_kwargs):
        """Calculate the Hausdorff distance between two meshes.

        Args:
            self (pyvista.PolyData): The mesh to calculate the Hausdorff distance for.
            compare_mesh (pyvista.PolyData): The mesh to compare against.
            **hd_kwargs: Additional keyword arguments to pass to the
                *`pyvista.PolyDataFilters.hausdorff_distance` method.

        Returns:
            float: The Hausdorff distance between the two meshes.

        """
        return hausdorff_distance(self, target_mesh=compare_mesh, **hd_kwargs)


def hausdorff_distance(
    mesh: UnstructuredGrid,
    target_mesh: UnstructuredGrid,
    return_both: bool = False,
    exclude_internal: bool = True,
):
    """Calculate the Hausdorff distance between two meshes.

    Args:
        mesh: the mesh that needs distances calculated
        target_mesh: the mesh to compare with.
        exclude_internal: perform distance calculations using only surface meshes. Defaults to True.

    Returns:
        a copy of mesh with a hausdorff_distance point data array, as well as a field
        data arrays with average hausdorff distance and the relative distance to the
        target mesh.

    """
    mesh = mesh.copy()
    target_mesh = target_mesh.copy()
    if exclude_internal:
        mesh = mesh.extract_surface(algorithm=None)
        target_mesh = target_mesh.extract_surface(algorithm=None)
    distance = vtk.vtkHausdorffDistancePointSetFilter()
    distance.SetTargetDistanceMethodToPointToCell()
    distance.SetInputData(0, mesh)
    distance.SetInputData(1, target_mesh)
    distance.Update()
    distance_a = pv.wrap(distance.GetOutput(0))
    distance_b = pv.wrap(distance.GetOutput(1))
    distance_b.rename_array(
        "RelativeDistanceBtoA", "relative_distance", preference="field"
    )
    distance_a.rename_array(
        "RelativeDistanceAtoB", "relative_distance", preference="field"
    )

    distances = np.array([distance_a, distance_b])
    idxs = np.argsort(np.hstack([x["relative_distance"] for x in distances]))
    distances = distances[idxs[::-1]]
    for distance in distances:
        distance.rename_array("Distance", "hausdorff_distance", preference="point")
        distance.rename_array(
            "HausdorffDistance", "hausdorff_distance", preference="field"
        )
        distance.set_active_scalars("hausdorff_distance")
    return distances if return_both else distances[0]


def nearest_distance(
    mesh: UnstructuredGrid, target_mesh: UnstructuredGrid, exclude_internal: bool = True
):
    """Find nearest distance between a mesh and target mesh.

    Returns a copy of the mesh with a distances scalar array.

    Args:
        mesh: the mesh that needs distances calculated.
        target_mesh: the mesh to compare to.
        exclude_internal: perform extract_surface before calculating distances. Defaults to True.

    Returns:
        a copy of mesh with calculated distances point data array.

    """
    mesh = mesh.copy()
    target_mesh = target_mesh.copy()
    if exclude_internal:
        mesh = mesh.extract_surface(algorithm=None)
        target_mesh = target_mesh.extract_surface(algorithm=None)
    closest_cells, closest_points = target_mesh.find_closest_cell(
        mesh.points, return_closest_point=True
    )
    d_exact = np.linalg.norm(mesh.points - closest_points, axis=1)
    mesh["distances"] = d_exact
    return mesh


# @with_debug_kwarg
def nearest_joint_side_distance(
    mesh,
    target_mesh,
    tolerance: float = 30,
    use_kdtree: bool = False,
    use_raytrace: bool = True,
    **kwargs,
):
    """Compute the nearest distance between the joint side surface of two meshes.

    Alternative distance measure to the Hausdorff distance.

    Args:
        mesh (pyvista.PolyData): The first mesh to compute the distance from.
        target_mesh (pyvista.PolyData): The second mesh to compute the distance to.
        tolerance (float, optional): The tolerance angle in degrees for the angle
            between the normal vectors of the two meshes. Defaults to 30.
        use_kdtree (bool, optional): Whether to use a KDTree to compute the distance.
            Defaults to False.
        use_raytrace (bool, optional): Whether to use ray tracing to compute the
            distance. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        pyvista.PolyData: A mesh representing the distances and vectors between the two
        input meshes.

    """
    mesh = extract_joint_side_surface(mesh).extract_surface(algorithm=None)
    target_mesh = extract_joint_side_surface(target_mesh).extract_surface(
        algorithm=None
    )
    closest_cells, closest_points = target_mesh.find_closest_cell(
        mesh.points, return_closest_point=True
    )
    if use_kdtree:
        tree = spatial.KDTree(target_mesh.points)
        distances, idx = tree.query(mesh.points)
        vecs = target_mesh.points[idx] - mesh.points

    elif use_raytrace:
        tree = spatial.KDTree(target_mesh.points)
        distances, idx = tree.query(mesh.points)
        tree_vecs = target_mesh.points[idx] - mesh.points

        tmesh = trimesh_wrap(mesh)
        ttarget_mesh = trimesh_wrap(target_mesh)
        origins = mesh.points
        directions = unitize(mesh.point_normals)
        _, ray_idx, locs = ttarget_mesh.ray.intersects_id(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
            return_locations=True,
        )
        vecs = np.full_like(mesh.points, np.nan, dtype=float)
        vecs[ray_idx] = locs - origins[ray_idx]
        directions = unitize(-1 * mesh.point_normals)
        _, ray_idx, locs = ttarget_mesh.ray.intersects_id(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
            return_locations=True,
        )

        vecs_2 = np.full_like(mesh.points, np.nan, dtype=float)
        vecs_2[ray_idx] = locs - origins[ray_idx]

        vecs[np.isnan(vecs)] = vecs_2[np.isnan(vecs)]
        vecs_2_is_least = row_norm(vecs) > row_norm(vecs_2)
        vecs[vecs_2_is_least] = vecs_2[vecs_2_is_least]

        # Handle convex shapes resulting in giant vectors
        too_large = row_norm(vecs) > 5 * row_norm(tree_vecs)
        too_large = np.tile(too_large, (3, 1)).T
        vecs = np.where(too_large, tree_vecs, vecs)
        vecs = np.where(np.isnan(vecs), tree_vecs, vecs)

        # plot the vectors
        if kwargs.get("debug", False):
            mesh["vecs"] = vecs
            mesh.set_active_vectors("vecs")
            mesh.set_active_scalars("vecs")
            p = pv.Plotter()
            p.add_mesh(target_mesh, color="tan", opacity=0.4, show_edges=False)
            p.add_mesh(mesh, nan_color="w", show_edges=True)
            p.add_mesh(mesh.arrows, color="r")
            p.show()
    else:
        vecs = mesh.points - closest_points
    angles_rads = tm_geometry.vector_angle(
        np.stack((unitize(mesh.point_normals), unitize(vecs)), axis=1)
    )
    angles_deg = np.rad2deg(angles_rads)
    outside_of_tolerance = (angles_deg > tolerance) & (angles_deg < (180 - tolerance))
    outside_of_tolerance[np.isnan(vecs).all(axis=1)] = True
    d_exact = row_norm(vecs)
    d_exact[outside_of_tolerance] = np.nan
    vecs[outside_of_tolerance] = np.full(3, np.nan)
    mesh["distances"] = d_exact
    mesh.set_active_scalars("distances")
    mesh["vecs"] = vecs
    mesh.set_active_vectors("vecs")
    mesh.set_active_scalars("distances")
    return mesh


def extract_joint_side_surface(mesh: UnstructuredGrid) -> pv.PolyData:
    """Extract the joint side surface from a mesh.

    Args:
        mesh (UnstructuredGrid): The input mesh (actually a CartilageMesh) with joint
            side surface point IDs defined during raytracing.

    Returns:
        pv.PolyData: The extracted joint side surface.

    """
    mesh.point_data["is_joint_side_pid"] = np.full(mesh.n_points, False)
    mesh.point_data["is_joint_side_pid"][mesh.field_data["joint_side_pids"]] = True
    js_mesh = mesh.extract_surface(algorithm=None)
    js_mesh = js_mesh.extract_points(
        js_mesh.point_data["is_joint_side_pid"].astype(bool),
        adjacent_cells=False,
    )
    return js_mesh


def sorted_nearest_joint_side_distances(
    mesh: UnstructuredGrid, target_mesh: UnstructuredGrid, *args, **kwargs
) -> List[UnstructuredGrid]:
    """Return mesh objects with nearest distance scalar arrays.

    Returns a list of two UnstructuredGrid objects representing the nearest joint side
    distances between the given `mesh` and `target_mesh`, sorted in descending order by
    their maximum relative distance.

    Args:
        mesh (UnstructuredGrid): The mesh to compute the distances from.
        target_mesh (UnstructuredGrid): The target mesh to compute the distances to.
        *args: Variable length argument list to be passed to the
            *`nearest_joint_side_distance` function.
        **kwargs: Arbitrary keyword arguments to be passed to the
            *`nearest_joint_side_distance` function.

    Returns:
        List[UnstructuredGrid]: A list of two UnstructuredGrid objects representing the
        nearest joint side distances between the given `mesh` and `target_mesh`, sorted
        in descending order by their maximum relative distance.

    """
    nearest_target_mesh = nearest_joint_side_distance(
        target_mesh, mesh, *args, **kwargs
    )
    nearest_mesh = nearest_joint_side_distance(mesh, target_mesh, *args, **kwargs)
    meshes = np.array([nearest_mesh, nearest_target_mesh])
    relative_distances = [x["distances"].max() for x in meshes]
    idxs_descending = np.argsort(relative_distances)[::-1]
    meshes = meshes[idxs_descending]
    return meshes


def cylindrical_arc(
    angle_degrees: float = 90,
    radius: float = 1,
    height: float = 10,
    **extrude_rotate_kwargs,
) -> pv.UnstructuredGrid:
    """Create a cylindrical arc mesh by extruding and rotating a line.

    Args:
        angle_degrees: The angle in degrees to rotate the line around the z-axis.
        radius: The radius of the cylindrical arc.
        height: The height of the cylindrical arc.
        **extrude_rotate_kwargs: Additional keyword arguments to pass to the
            *`extrude_rotate` method.

    Returns:
        A `pyvista.UnstructuredGrid` object representing the cylindrical arc mesh.

    Raises:
        TypeError: If `angle_degrees`, `radius`, or `height` are not numeric.
        ValueError: If `angle_degrees`, `radius`, or `height` are negative.

    Example:
        >>> mesh = cylindrical_arc(angle_degrees=90, radius=1, height=10)

    """
    extrude_rotate_kwargs.pop("inplace", None)
    line = pv.Line((0, 0, 0), (radius, 0, 0))
    swept = line.extrude_rotate(angle=angle_degrees, **extrude_rotate_kwargs)
    cutoff_plane = pv.Plane(
        center=(0, 0, height), direction=(0, 0, 1), i_size=radius * 2, j_size=radius * 2
    )
    extruded = swept.extrude_trim(direction=(0, 0, 1), trim_surface=cutoff_plane)
    if extrude_rotate_kwargs.get("debug", False):
        p = pv.Plotter()
        p.add_mesh(swept, color="tan", opacity=0.4, show_edges=False)
        p.add_mesh(extruded, color="tan", opacity=0.4, show_edges=False)
        p.add_mesh(cutoff_plane, color="r", opacity=0.4, show_edges=False)
        p.add_axes()
        p.show()

    return extruded


def add_mesh_clip_box(
    plotter,
    mesh,
    invert: bool = False,
    rotation_enabled: bool = True,
    widget_color=None,
    outline_translation: bool = True,
    merge_points: bool = True,
    crinkle: bool = False,
    interaction_event: pv.InteractionEventType = "end",
    **kwargs,
):
    """Clip a mesh using a box widget.

    Add a mesh to the scene with a box widget that is used to clip
    the mesh interactively.

    The clipped mesh is saved to ``plotter.widgets.box_clipped_meshes``.
    This widget uses VTK's box clipper, which can split volume cells into
    tetrahedra, unlike PyVista's plane-based grid ``clip_box`` filter.

    Parameters
    ----------
    mesh : pyvista.DataSet or vtk.vtkAlgorithm
        The input dataset to add to the scene and clip or algorithm that
        produces said mesh.

    invert : bool, optional
        Flag on whether to flip/invert the clip.

    rotation_enabled : bool, optional
        If ``False``, the box widget cannot be rotated and is strictly
        orthogonal to the cartesian axes.

    widget_color : ColorLike, optional
        Color of the widget.  Either a string, RGB sequence, or
        hex color string.  For example:

        * ``color='white'``
        * ``color='w'``
        * ``color=[1.0, 1.0, 1.0]``
        * ``color='#FFFFFF'``

    outline_translation : bool, optional
        If ``False``, the plane widget cannot be translated and is
        strictly placed at the given bounds.

    merge_points : bool, optional
        If ``True`` (default), coinciding points of independently
        defined mesh elements will be merged.

    crinkle : bool, optional
        Crinkle the clip by extracting the entire cells along the clip.

    interaction_event : InteractionEventType, optional
        The VTK interaction event to use for triggering the
        callback. Accepts either the strings ``'start'``, ``'end'``,
        ``'always'`` or a ``vtk.vtkCommand.EventIds``.

        .. versionchanged:: 0.38.0
            Changed from ``event_type`` to ``interaction_event`` and now
            accepts either strings and ``vtk.vtkCommand.EventIds``.

    **kwargs : dict, optional
        All additional keyword arguments are passed to
        :func:`pyvista.Plotter.add_mesh` to control how the mesh is
        displayed.

    Returns
    -------
    vtk.vtkActor
        VTK actor of the mesh.

    Examples
    --------
    Shows an interactive clip box.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricConicSpiral()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh_clip_box(mesh, color='white')
    >>> pl.show()

    For a full example see :ref:`box_widget_example`.

    """
    from pyvista.core.filters import _get_output  # avoids circular import

    mesh, algo = pv.algorithm_to_mesh_handler(
        pv.add_ids_algorithm(mesh, point_ids=False, cell_ids=True),
    )

    name = kwargs.get("name", mesh.memory_address)
    rng = mesh.get_data_range(kwargs.get("scalars"))
    kwargs.setdefault("clim", kwargs.pop("rng", rng))
    mesh.set_active_scalars(kwargs.get("scalars", mesh.active_scalars_name))

    plotter.add_mesh(
        pv.plotting.utilities.outline_algorithm(algo),
        name=f"{name}-outline",
        opacity=0.0,
    )  # type: ignore[attr-defined]

    port = 1 if invert else 0

    clipper = vtk.vtkBoxClipDataSet()
    if not merge_points:
        # vtkBoxClipDataSet uses vtkMergePoints by default
        clipper.SetLocator(vtk.vtkNonMergingPointLocator())
    pv.set_algorithm_input(clipper, algo)
    clipper.GenerateClippedOutputOn()

    if crinkle:
        crinkler = pv.crinkle_algorithm(clipper.GetOutputPort(port), algo)
        box_clipped_mesh = _get_output(crinkler)
    else:
        box_clipped_mesh = _get_output(clipper, oport=port)

    plotter.widgets.box_clipped_meshes.append(box_clipped_mesh)
    pv.set_new_attribute(plotter, "clipping_planes", pv.MultiBlock())
    pv.set_new_attribute(plotter, "clipped_bounds", [])

    def callback(planes):
        bounds = []
        plotter.clipping_planes = pv.MultiBlock()
        length = (clipper.GetInput()).GetLength()
        for i in range(planes.GetNumberOfPlanes()):
            plane = planes.GetPlane(i)
            normal = plane.GetNormal()
            origin = plane.GetOrigin()
            pv_plane = pv.Plane(
                center=origin,
                direction=normal,
                i_size=length,
                j_size=length,
            )
            pv_plane.field_data.set_array(normal, "normal")
            pv_plane.field_data.set_array(origin, "origin")
            plotter.clipping_planes.append(pv_plane)
            bounds.append(normal)
            bounds.append(origin)

        clipper.SetBoxClip(*bounds)
        clipper.Update()
        if crinkle:
            crinkler.Update()
            clipped = _get_output(crinkler)
        else:
            clipped = _get_output(clipper, oport=port)
        box_clipped_mesh.shallow_copy(clipped)
        box_clipped_mesh.field_data.set_array(
            bounds,
            "box_clip_bounds",
        )
        plotter.clipped_bounds = bounds

    plotter.add_box_widget(
        callback=callback,
        bounds=mesh.bounds,
        factor=1.25,
        rotation_enabled=rotation_enabled,
        use_planes=True,
        color=widget_color,
        outline_translation=outline_translation,
        interaction_event=interaction_event,
    )

    if crinkle:
        return plotter.add_mesh(crinkler, reset_camera=False, **kwargs)  # type: ignore[attr-defined]
    return plotter.add_mesh(clipper.GetOutputPort(port), reset_camera=False, **kwargs)  # type: ignore[attr-defined]
