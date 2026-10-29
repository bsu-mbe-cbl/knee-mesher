"""A module containing small utility functions not deserving a namespace."""

from __future__ import annotations

import linecache
import logging
import os
import platform
import tracemalloc
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pyacvd
import pyvista as pv
import scipy.io as spio
import trimesh
from sklearn.decomposition import PCA

# from pyvista.utilities.misc import PyvistaDeprecationWarning
from trimesh import Trimesh, unitize
from trimesh.geometry import vector_angle
from trimesh.util import row_norm

from kneemesher._typing import ArrayLike, PolyData
from kneemesher.meshing.cells import HexMesh
from kneemesher.mesh_repair import (
    RepairBackend,
    repair_mesh as _repair_mesh,
    trimesh_wrap,
)


@contextmanager
def set_log_level(logger, level):
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def prepare_for_probreg(
    points: np.ndarray,
    /,
    *more_points: np.ndarray,
    voxel_size: float | None = None,
    normals: bool = False,
    dtype: np.dtype = np.float32,
):
    """Prepare point cloud data for use with the ProbReg library.

    Processes input point arrays, optionally down-sampling them and estimating
    normals.

    Args:
        points (np.ndarray): The primary point cloud as a NumPy array.
        *more_points (np.ndarray): Additional point clouds to process.
        voxel_size (float | None, optional): The size of the voxel grid for
            down-sampling. If None, no down-sampling is performed. Defaults to None.
        normals (bool, optional): Whether to estimate normals for the point clouds.
            Defaults to False.
        dtype (np.dtype, optional): The data type for the output arrays.
            Defaults to np.float32.

    Returns:
        tuple: A tuple of processed point clouds as NumPy arrays, with the same
        order as the input arguments.

    """
    import open3d as o3d

    more_points = (points,) + more_points

    return_args = list(more_points)
    for arg_idx, point_array in enumerate(more_points):
        arg_ = o3d.geometry.PointCloud()
        arg_.points = o3d.utility.Vector3dVector(point_array)
        if normals:
            arg_.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size, max_nn=30
                )
            )
        if voxel_size is not None:
            arg_ = arg_.voxel_down_sample(voxel_size=voxel_size)
        return_args[arg_idx] = np.asarray(arg_.points, dtype=dtype)
    return tuple(return_args)


def apply_cpd_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    voxel_size: float = 5,
    use_cuda: bool = False,
    return_transformation: bool = False,
) -> tuple[np.ndarray, pv.Transform]:
    """Apply Coherent Point Drift registration to align source points to target points.

    Parameters
    ----------
        source_points (np.ndarray): The source point cloud as a NumPy array of shape
            (N, 3).
        target_points (np.ndarray): The target point cloud as a NumPy array of shape
            (M, 3).
        voxel_size (float, optional): The voxel size for downsampling the point clouds.
            Defaults to 5.
        use_cuda (bool, optional): Whether to use CUDA for GPU acceleration. Defaults
            to False.
        return_transformation (bool, optional): Whether to return the transformation
            object. Defaults to False.

    Returns
    -------
        tuple[np.ndarray, pv.Transform]:
            - A NumPy array of the transformed source points of shape (N, 3).
            - A `pv.Transform` object representing the transformation (only if
              `return_transformation` is True). The transformation is composed of
              scaling, rotation, and translation components (in that order).

    Raises
    ------
        AssertionError: If the computed transformation matrix does not match the
            expected output.

    Notes
    -----
        - If `use_cuda` is enabled, the function will attempt to use CuPy for GPU
          acceleration.
        - The transformation includes scaling, rotation, and translation components.

    """
    use_cuda = False

    try:
        if use_cuda:
            import cupy as cp

            to_cpu = cp.asnumpy
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else:
            cp = np
            to_cpu = lambda x: x  # noqa: E731
    except ImportError:
        use_cuda = False
        cp = np
        to_cpu = lambda x: x  # noqa: E731

    from probreg import cpd

    source_points = cp.asarray(source_points, dtype=cp.float32)
    target_points = cp.asarray(target_points, dtype=cp.float32)
    source_, target_ = prepare_for_probreg(
        source_points, target_points, voxel_size=voxel_size
    )
    if use_cuda:
        source_, target_ = cp.asarray(source_), cp.asarray(target_)
    tf_param, _, _ = cpd.registration_cpd(
        source=source_,
        target=target_,
        use_cuda=use_cuda,
    )

    return_points = to_cpu(tf_param.transform(cp.array(source_points)))

    trans = (
        pv.Transform(multiply_mode="post")
        .scale(tf_param.scale)
        .rotate(tf_param.rot)
        .translate(tf_param.t)
    )

    assert np.allclose(
        (
            trans.matrix
            @ np.pad(
                source_points, ((0, 0), (0, 1)), mode="constant", constant_values=1
            ).T
        )[:-1, :].T,
        return_points,
    ), "Transformation matrix does not match expected output."

    if not return_transformation:
        return return_points
    else:
        return return_points, trans


def plot_axes(
    axes: ArrayLike,
    center: ArrayLike,
    mesh: PolyData = None,
    labels: list[str] = None,
):
    """Plot the axes arrows and optionally a mesh in a 3D plot.

    Args:
        axes (ArrayLike): List of axes vectors.
        center (ArrayLike): Center point of the axes.
        mesh (PolyData, optional): Mesh to be plotted. Defaults to None.

    Returns:
        None.

    """
    if labels is None:
        labels = ["anterior", "transverse", "axial"]
    axes_arrows = [pv.Arrow(start=center, direction=a, scale=10) for a in axes]

    p = pv.Plotter()
    [
        p.add_mesh(aa, color=c, label=lbl)
        for aa, c, lbl in zip(axes_arrows, "rgb", labels)
    ]
    if mesh is not None:
        p.add_mesh(mesh, color="w", opacity=0.5)
    p.add_legend()
    p.add_axes()
    p.show()


def locate_joint_axes(
    femur_bone,
    tibia_bone,
    patella_bone,
    *args,
    use_fallback: bool = False,
    **kwargs,
):
    """Locate joint axes consistent with anatomical planes.

    Compute the principal axes of the joint and create a right-handed coordinate
    system with consistent orientation for the joint axes.

    Args:
        femur_bone: The femur bone mesh.
        femur_cartilage: The femur cartilage mesh.
        tibia_bone: The tibia bone mesh.
        patella_bone: The patella bone mesh.
        args (list): Optional list of strings specifying the names of the tissues to
            be included.
        kwargs (dict): Optional keyword arguments:
            - debug (bool): If True, plot the joint mesh and the principal axes.

    Returns:
        dict: A dictionary containing the principal axes of the joint:
            - axial (numpy.ndarray): The axial axis of the joint.
            - transverse (numpy.ndarray): The transverse axis of the joint.
            - ant-post (numpy.ndarray): The antero-posterior axis of the joint.

    """
    # Combine femur and femur cartilage
    joint_mesh = tibia_bone + femur_bone
    if len(args) > 0:
        joint_mesh = pv.merge([locals()[s] for s in args])
    center = joint_mesh.center

    # Find the principal axes of the joint
    pca = PCA(n_components=3)
    pca.fit(joint_mesh.points)
    vectors = pca.components_
    axial_axis, transverse_axis, ant_post_axis = vectors

    # Create a right-handed coordinate system with consistent orientation
    # for the principal axes
    # The axial axis should point from the tibia toward the top of the femur
    femur_com = femur_bone.center_of_mass()
    femur_cartilage_com = tibia_bone.center_of_mass()
    if np.dot(axial_axis, femur_com - femur_cartilage_com) < 0:
        axial_axis = -axial_axis
    # The ant-post axis should point from the femur toward the patella
    patella_com = patella_bone.center_of_mass()
    if np.dot(ant_post_axis, patella_com - femur_com) < 0:
        ant_post_axis = -ant_post_axis
    # The transverse axis should be positive orthogonal to the axial and ant-post axes
    transverse_axis = np.cross(axial_axis, ant_post_axis)
    transverse_axis = trimesh.unitize(transverse_axis)
    axes = np.array([ant_post_axis, transverse_axis, axial_axis])

    if kwargs.get("debug", False):
        axes_arrows = [pv.Arrow(start=center, direction=a, scale=10) for a in axes]

        p = pv.Plotter()
        [
            p.add_mesh(aa, color=c, label=l)
            for aa, c, l in zip(axes_arrows, "rgb", ["anterior", "transverse", "axial"])
        ]
        p.add_mesh(joint_mesh, color="w", opacity=0.5)
        p.add_legend()
        p.add_axes()
        p.show()
    axes_dict = {k: v for k, v in zip(["ant-post", "transverse", "axial"], axes)}
    if use_fallback:
        axes_dict = {k: v for k, v in zip(["anterior", "medial", "superior"], axes)}

    return axes_dict


def basic_logging(filename: str = None, filemode: str = None, level=None):
    logging.getLogger("shapely.geos").disabled = True
    logging.basicConfig(filename=filename, level=level, filemode=filemode)


def import_matlab_data(filename: Union[str, Path]) -> Dict:
    """Import a matlab MAT file to a python dictionary.

    Args:
        filename (Union[str, Path]): The MAT filename.

    Returns:
        Dict: A with matlab variable names as keys.

    """
    data = spio.loadmat(filename, squeeze_me=True)
    data = {k: data[k] for k in data.keys() if "__" not in k}
    return data


def export_matlab_data(filename: Union[str, Path], data_dict: Dict) -> None:
    """Export a dictionary as a MAT file for use with MATLAB.

    Args:
        filename (Union[str, Path]): The MAT file filename.
        data_dict (Dict): The dictionary with MATLAB variable names as keys.

    """
    spio.savemat(file_name=filename, mdict=data_dict)


def shortest_align_vectors(
    origin_vec: np.ndarray,
    target_vec: np.ndarray,
) -> np.ndarray:
    """Return the shortest path rotation matrix to align two vectors.

    Sourced from the wikipedia [page] for rotation matrices. The passed vectors will be
    turned into unit vectors.

    Args:
        origin_vec (np.ndarray): the vector that will be rotated.
        target_vec (np.ndarray): the vector representing the axis of alignment.

    Note:
        [page]: (wikipedia.org/wiki/Rotation_matrix#Vector_to_vector_formulation)

    Returns:
        np.ndarray: A shortest path rotation matrix of size (3, 3)

    """
    origin_vec, target_vec = [unitize(v) for v in (origin_vec, target_vec)]
    origin_vec = origin_vec[np.newaxis].T if origin_vec.ndim < 2 else origin_vec
    target_vec = target_vec[np.newaxis].T if target_vec.ndim < 2 else target_vec
    A = target_vec @ origin_vec.T - origin_vec @ target_vec.T
    s = 1 / (1 + origin_vec.T @ target_vec)
    return np.eye(origin_vec.size) + A + s * (A @ A)


def clustered_mesh(
    stl_mesh: Union[pv.UnstructuredGrid, pv.PolyData, Trimesh, HexMesh],
    target_size: float = 3.0,
    show_cluster_plot: bool = False,
    verbose_mesh_repair: bool = False,
    *,
    repair_backend: RepairBackend = "auto",
) -> pv.PolyData:
    """Use Voronoi clustering to create a uniform triangular mesh.

    Args:
        stl_mesh: A mesh object of type `pv.UnstructuredGrid`, `pv.PolyData`,
            `tm.Trimesh`, or `HexMesh`.
        target_size: The target size of each cluster in the output mesh.
        show_cluster_plot: Whether to display a plot of the clustered mesh.
        verbose_mesh_repair: Whether to print verbose output during mesh repair.
        repair_backend: The backend used at each repair stage.

    Returns:
        A `pv.PolyData` object representing the clustered mesh.

    Raises:
        TypeError: If `stl_mesh` is not of a valid mesh type.

    """
    repaired_mesh = repair_mesh(
        pv.wrap(stl_mesh),
        verbose_mesh_repair,
        backend=repair_backend,
        require_closed=True,
    )
    side_length = (target_size * 2) ** 0.5
    dense_mesh = repaired_mesh.subdivide_adaptive(max_edge_len=side_length / 4)
    repaired_dense_mesh = repair_mesh(
        dense_mesh,
        verbose_mesh_repair,
        backend=repair_backend,
        require_closed=True,
    )
    n_cluster = np.ceil(repaired_dense_mesh.area / target_size / 2).astype(int)
    cluster_surf = pyacvd.Clustering(repaired_dense_mesh)
    cluster_surf.cluster(n_cluster)
    cluster_surf.plot() if show_cluster_plot else None
    with warnings.catch_warnings():  # point_arryas -> point_data warnings
        # warnings.filterwarnings("ignore", category=PyvistaDeprecationWarning)
        cluster_mesh = cluster_surf.create_mesh()
        cluster_mesh = repair_mesh(
            cluster_mesh,
            verbose_mesh_repair,
            backend=repair_backend,
            require_closed=True,
        )
    return cluster_mesh


def repair_mesh(
    mesh: pv.DataSet | Trimesh,
    verbose: bool = False,
    *,
    backend: RepairBackend = "auto",
    require_closed: bool = False,
) -> pv.PolyData:
    """Delegate surface repair to the shared backend entry point.

    Args:
        mesh: A `pyvista.PolyData` object representing the mesh to be repaired.
        verbose: A boolean flag indicating whether to print verbose output during the
            repair process.
        backend: The repair backend, or automatic selection based on availability.
        require_closed: Whether to reject remaining closure/manifold defects.

    Returns:
        A `pyvista.PolyData` object representing the repaired mesh.

    """
    return _repair_mesh(mesh, verbose, backend=backend, require_closed=require_closed)


def log_top_memory_hogs(snapshot, key_type="lineno", limit=10):
    """Log the top memory hogs in the given snapshot.

    Args:
        snapshot: A tracemalloc.Snapshot object representing the current memory
            snapshot.
        key_type: A string representing the type of key to use for grouping the
            statistics.  Valid values are "lineno" (default) and "traceback".
        limit: An integer representing the maximum number of lines to log. Default is
            10.

    Returns:
        None

    """
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, np.__file__),
            tracemalloc.Filter(False, tracemalloc.__file__),
        )
    )
    top_stats = snapshot.statistics(key_type)

    logging.info("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        logging.info(
            "#%s: %s:%s: %.1f KiB"
            % (index, frame.filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            logging.info("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        logging.info("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    logging.info("Total allocated size: %.1f KiB" % (total / 1024))


def cartesian_to_polar(x, y, z):
    """Convert cartesian coordinates to polar coordinates.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        z (float): The z-coordinate.

    Returns:
        tuple: A tuple containing the polar coordinates (rho, phi, z).

    """
    rho = row_norm(np.vstack((x, y)).T).squeeze()
    phi = np.array(np.arctan2(y, x)).squeeze()
    z = np.array(z).squeeze()
    return rho, phi, z


def polar_to_cartesian(rho, phi):
    """Convert polar coordinates to cartesian coordinates.

    Args:
        rho (float): The radial distance from the origin.
        phi (float): The angle in radians from the positive x-axis.

    Returns:
        tuple: A tuple containing the x and y coordinates in cartesian space.

    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def fix_protruding_bones(
    bone_reconstruction: pv.PolyData,
    cart_meshes: pv.UnstructuredGrid,
    return_meshed: bool = False,
    *,
    repair_backend: RepairBackend = "auto",
    **clustered_mesh_kwargs,
) -> pv.PolyData:
    """Remove pieces of bone protruding through the cartilage mesh.

    Args:
        bone_reconstruction: the unoptimized bone reconstruction surface mesh.
        cart_meshes: the hexahedral meshes of the paired articular cartilage tissues.
        return_meshed: whether to return the modified reconstruction or finite element
            ready mesh. Defaults to False.
        repair_backend: The backend used for repair and optional clustering.

    Returns:
        Either the bone reconstruction with protrusions removed, or the finite element
        ready surface mesh of the modified reconstruction.

    """
    tri_carts = [None] * len(cart_meshes)
    bs_surfs = [None] * len(cart_meshes)
    for ct_idx, cart_mesh in enumerate(cart_meshes):
        tri_cart = cart_mesh.extract_surface(algorithm=None).triangulate()
        tcart = trimesh_wrap(tri_cart)
        tcart.fix_normals()
        tri_cart.compute_normals(inplace=True)
        if not np.allclose(tri_cart.cell_data["Normals"], tcart.face_normals):
            tri_cart.flip_faces(inplace=True)
        tri_cart.compute_normals(inplace=True)
        tri_cart_mesh_pids = tri_cart.point_data["mesh_pids"]
        joint_side_mesh_pids = cart_mesh.field_data["joint_side_pids"]
        needs_moving = np.isin(tri_cart_mesh_pids, joint_side_mesh_pids)
        tri_cart.point_data["disp_normals"] = tri_cart.point_data["Normals"].copy()
        tri_cart.point_data["disp_normals"][~needs_moving] = np.zeros(3)
        tri_cart.points += 3 * tri_cart.point_data["disp_normals"]
        tri_carts[ct_idx] = tri_cart

        bs_surf = tri_cart.extract_points(
            np.isin(
                # tri_cart.point_data["mesh_pids"], ct.field_data["joint_side_pids"]
                tri_cart.point_data["mesh_pids"],
                cart_mesh.field_data["bone_side_pids"],
            ),
            adjacent_cells=False,
        ).extract_surface(algorithm=None)
        bs_surf.flip_faces(inplace=True)
        bs_surfs[ct_idx] = bs_surf

    difference = bone_reconstruction.boolean_difference(pv.merge(tri_carts))
    fixed_mesh = repair_mesh(
        pv.merge((difference, *bs_surfs)),
        backend=repair_backend,
        require_closed=True,
    )
    if return_meshed:
        fixed_mesh = clustered_mesh(
            fixed_mesh, repair_backend=repair_backend, **clustered_mesh_kwargs
        )
    return fixed_mesh


def fix_protruding_bones_with_raytracing(
    bone_reconstruction: pv.PolyData,
    cart_meshes: pv.UnstructuredGrid,
    max_displacement: float = 5,
    return_meshed: bool = False,
    **clustered_mesh_kwargs,
):
    """Fix protruding bones in a knee joint by raytracing towards the joint-side.

    Args:
        bone_reconstruction (pv.PolyData): The reconstructed bone mesh.
        cart_meshes (pv.UnstructuredGrid): The cartilage meshes.
        max_displacement (float, optional): The maximum displacement allowed. Defaults
            to 5.
        return_meshed (bool, optional): Whether to return a clustered mesh. Defaults to
            False.
        **clustered_mesh_kwargs: Additional keyword arguments to pass to the
            *`clustered_mesh` function.

    Returns:
        pv.PolyData: The fixed bone mesh.

    """
    tri_carts = [None] * len(cart_meshes)
    bs_surfs = [None] * len(cart_meshes)
    # Loop through the cartilage meshes
    for ct_idx, cart_mesh in enumerate(cart_meshes):
        tri_cart = cart_mesh.extract_surface(algorithm=None).triangulate()
        tcart = trimesh_wrap(tri_cart)
        tcart.fix_normals()
        tri_cart.compute_normals(inplace=True)
        if not np.allclose(tri_cart.cell_data["Normals"], tcart.face_normals):
            tri_cart.flip_faces(inplace=True)
        tri_cart.compute_normals(inplace=True)
        tri_cart_mesh_pids = tri_cart.point_data["mesh_pids"]

        # Select the bone side surface
        bone_side_mesh_pids = cart_mesh.field_data["bone_side_pids"]
        needs_tracing = np.isin(tri_cart_mesh_pids, bone_side_mesh_pids)
        bs_surf = tri_cart.extract_points(needs_tracing, adjacent_cells=False)
        bs_surf.point_data["Normals"]
        bs_surfs[ct_idx] = bs_surf.extract_surface(algorithm=None)
        tri_carts[ct_idx] = tri_cart


    trimesh_bone = trimesh_wrap(bone_reconstruction)
    origins = trimesh_bone.vertices
    directions = unitize(-1 * trimesh_bone.vertex_normals)

    combined_bs_surfs = trimesh_wrap(pv.merge(bs_surfs))
    tri_idx, ray_idx, locations = combined_bs_surfs.ray.intersects_id(
        ray_origins=origins,
        ray_directions=directions,
        return_locations=True,
        multiple_hits=False,
    )

    # Only include backface intersections
    bs_face_norms = combined_bs_surfs.face_normals[tri_idx]
    ray_norms = directions[ray_idx]
    angle_between = vector_angle(np.stack((ray_norms, bs_face_norms), axis=1))
    is_backface = np.cos(angle_between) > 0.0
    ray_idx = ray_idx[is_backface]
    locations = locations[is_backface]

    # make the moves
    deltas = np.zeros_like(origins)
    deltas[ray_idx] = locations - origins[ray_idx]
    deltas[row_norm(deltas) > max_displacement] = np.zeros(3)

    bone_reconstruction.points += deltas
    fixed_mesh = bone_reconstruction

    if return_meshed:
        fixed_mesh = clustered_mesh(fixed_mesh, **clustered_mesh_kwargs)
    return fixed_mesh


def with_debug_kwarg(func):
    """Decorate function with the debug kwarg.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, debug=True, **kwargs)

    return wrapper


def with_screenshot_kwargs(func):
    """Decorate function with the screenshot_kwargs kwarg.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, screenshot_kwargs=True, **kwargs)

    return wrapper


LEFT_KNEE_SUBJECTS = [
    "9040390",
    "9047800",
    "9054866",
    "9087863",
    "9146462",
    "9172459",
    "9192885",
    "9211869",
    "9215390",
    "9264046",
    "9309170",
    "9311328",
    "9331465",
    "9332085",
    "9368622",
    "9382271",
    "9415074",
    "9444401",
    "9482482",
    "9493245",
    "9500390",
    "9539084",
    "9597990",
    "9599539",
    "9602703",
    "9607698",
    "9625955",
    "9626069",
    "9660697",
    "9663706",
    "9674570",
    "9723911",
    "9745458",
    "9750920",
    "9755935",
    "9761431",
    "9802877",
    "9892765",
    "9897397",
    "9902757",
    "9912946",
    "9947240",
    "9954040",
    "9958234",
    "9967358",
    "9992358",
]
FLIPPED_SUBJECTS = [
    "9047800_M00",
    "9047800_M12",
    # "9054866_M00",
]
FAILED_TIBIA_SUBJECTS = [
    "9085290_M00",
    "9102858_M12",
    "9146462_M00",
    "9146462_M12",
    "9368622_M00",
    "9368622_M12",
    "9482482_M12",
    "9493245_M00",
    "9493245_M12",
    "9657464_M00",
    "9657464_M12",
    "9674570_M00",
    "9693364_M00",
    "9745458_M12",
    "9897397_M00",
    "9904241_M00",
    "9954040_M00",
    "9968924_M00",
    "9968924_M12",
    "9085290_M00",
    "9102858_M12",
    "9146462_M00",
    "9146462_M12",
    "9368622_M00",
    "9368622_M12",
    "9482482_M12",
    "9493245_M00",
    "9493245_M12",
    "9657464_M00",
    "9657464_M12",
    "9674570_M00",
    "9693364_M00",
    "9745458_M12",
    "9897397_M00",
    "9904241_M00",
    "9954040_M00",
    "9968924_M00",
    "9968924_M12",
]
FAILED_PATELLA_SUBJECTS = [
    "9087863_M00",
    "9331465_M12",
]


class StrictDataclass:
    def __setattr__(self, name, value):
        if name not in self.__dataclass_fields__:
            raise AttributeError(f"Cannot assign unknown attribute '{name}'")
        super().__setattr__(name, value)


def find_zip_archive_on_mounted_disks(
    archive_pattern: str,
    search_dirs: list = None,
) -> Path:
    """Locate a zip archive on a mounted disk across OSes.

    Args:
        archive_name: Name of the zip archive to find (e.g., 'inp.zip').
        cohort_dir: Optional subdirectory to look for (e.g., '20250724Cohort').
        search_dirs: Optional list of base directories to search (for Linux/Mac).

    Returns:
        Path to the found archive.

    Raises:
        FileNotFoundError: If the archive is not found.

    Example:
        >>> find_zip_archive_on_mounted_disks('inp.zip', '20250724Cohort')
        PosixPath('/mnt/some_drive/20250724Cohort/inp.zip')

    """
    system = platform.system().lower()
    if system == "windows":
        # On Windows, check all drives
        if search_dirs:
            drives = [f"{d.upper().strip(':\\/')}:/" for d in search_dirs]
        elif hasattr(os, "listdrives"):
            drives = os.listdrives()
        else:
            # Fallback: try common drive letters
            drives = [f"{chr(d)}:/" for d in range(69, 82)]  # C: to Z:\
        for drive in drives:
            matches = Path(drive).rglob(archive_pattern)
            for candidate in matches:
                if candidate.exists():
                    return candidate
    elif system == "linux" or system == "darwin":
        # On Linux/Mac, check common mount points and home
        if search_dirs is None:
            search_dirs = [
                Path("/run/media"),
                Path("/mnt"),
                Path("/media"),
                Path.home(),
            ]
        for base_dir in search_dirs:
            if base_dir.exists():
                matches = base_dir.rglob(archive_pattern)
                for candidate in matches:
                    if candidate.exists():
                        return candidate
    else:
        raise RuntimeError(f"Unsupported OS for locating {archive_pattern}.")
    raise FileNotFoundError(f"Could not find {archive_pattern} archive.")
