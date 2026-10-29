"""Plotting utilities for visualizing meshes, vectors, and pointclouds."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pyvista as pv
import trimesh

from kneemesher.meshing.cells import HexElement, HexMesh


def scene(
    point_clouds: Tuple[np.ndarray] = None,
    reconstructions: Tuple[Union[str, Path, trimesh.Trimesh]] = None,
    meshes: Tuple[Union[HexMesh, Tuple[np.ndarray, np.ndarray]]] = None,
    rays: Tuple[np.ndarray, np.ndarray] = None,
    rays_as_lines: bool = False,
    opacities: Union[list, tuple] = [1, 1, 1],
) -> pv.Plotter:
    """Create a pyvista scene with multiple objects.

    Args:
        point_clouds (Tuple[np.ndarray], optional): a tuple of (n, 3) point locations.
            Defaults to None.
        reconstructions (Tuple[Union[str, Path, trimesh.Trimesh]], optional): filename
            or trimesh.Trimesh objects. Defaults to None.
        meshes (Tuple[Union[HexMesh, Tuple[np.ndarray, np.ndarray]]], optional): HexMesh
            objects, or tuples of (elements, node_location) definitions. Defaults to
            None.
        rays (Tuple[np.ndarray, np.ndarray], optional): a tuple of (n, 3) vector origin
            and termination locations. Defaults to None.
        rays_as_lines (bool, optional): wheter to render the rays as lines instead of
            arrows. Defaluts to False.
        opacities (Union[list, tuple], optional): Opacities for the point clouds,
            reconstructions, and meshes. Defaults to [1, 1, 1].

    Returns:
        pv.Plotter: The plotter object, which can be called using the `show()` method.
    """
    plotter = pv.Plotter()
    plotter.add_axes()

    if point_clouds:
        point_clouds = ensure_point_cloud(*point_clouds, force_list_output=True)
        point_clouds = ensure_pyvista(*point_clouds)
        [
            plotter.add_points(
                x, color=np.random.randint(3 * [256]), opacity=opacities[0]
            )
            for x in point_clouds
        ]

    if reconstructions:
        reconstructions = ensure_trimesh(*reconstructions, force_list_output=True)
        reconstructions = ensure_pyvista(*reconstructions, force_list_output=True)
        [
            plotter.add_mesh(x, color=np.random.rand(3), opacity=opacities[1])
            for x in reconstructions
        ]

    if meshes:
        meshes = ensure_hex_mesh(*meshes, force_list_output=True)
        meshes = ensure_pyvista(*meshes, force_list_output=True)
        [
            plotter.add_mesh(
                x,
                color=np.random.rand(3),
                opacity=opacities[2],
                show_edges=True,
                label=str(idx),
            )
            for idx, x in enumerate(meshes)
        ]

    if rays:
        rays = ensure_arrow_glyphs(*rays, force_list_output=True, as_line=rays_as_lines)
        [
            plotter.add_mesh(arrows, show_scalar_bar=False, color=np.random.rand(3))
            for arrows in rays
        ]

    return plotter


def ray_glyphs(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    ray_length: Union[int, float] = None,
    as_line: bool = False,
) -> pv.PolyData:
    """Create pyvista vector glyphs from ray vector origins and directions.

    Args:
        ray_origins (np.ndarray): (n, 3) origin of the vectors/rays.
        ray_directions (np.ndarray): (n, 3) termination locations of the vectors/rays.
        ray_length (Union[int, float], optional): A uniform length to scale the glyphs
            to. Defaults to None.

    Returns:
        pv.PolyData: the pyvista compatible glyphs for use in plotting routines.
    """
    rays = ensure_point_cloud(ray_origins)
    rays["vectors"] = (
        ray_length * trimesh.unitize(ray_directions) if ray_length else ray_directions
    )
    rays.set_active_vectors("vectors")
    geom = pv.Line((0, 0, 0), (1, 0, 0)) if as_line else pv.Arrow()
    return rays.glyph(geom=geom)


def show_rays(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    ray_length: Union[int, float] = None,
    use_trimesh: bool = False,
) -> None:
    """View rays using either pyvista or trimesh plotting utilities.

    Args:
        ray_origins (np.ndarray): An (n, 3) array of vector origin locations.
        ray_directions (np.ndarray): An (n, 3) array of vector termination locations.
        ray_length (Union[int, float], optional): Cast the vectors to uniform lengths.
            Defaults to None.
        use_trimesh (bool, optional): Whether to use trimesh plotting. Defaults to
            False.
    """
    if not use_trimesh:
        ray_glyphs(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_length=ray_length,
            as_line=True,
        ).plot(color="r")
    else:
        rays_to_paths(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_length=ray_length,
        ).show()


def rays_to_paths(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    ray_length: Union[int, float] = None,
) -> trimesh.path.Path2D:
    """Convert vector origins and directions to trimesh paths for visualization.

    Args:
        ray_origins (np.ndarray): An (n, 3) array of cartesian origin points.
        ray_directions (np.ndarray): A corresponding (n, 3) array of cartesian vector
            termination points.
        ray_length (Union[int, float], optional): A scalar multiplier applied to the
            length of the resulting trimesh path objects. Defaults to 5.

    Returns:
        trimesh.path.Path2D: trimesh paths with scaling applied for use in trimesh.Scene
        visualizations.
    """
    ray_length = 5 if ray_length is None else ray_length
    return trimesh.load_path(
        np.hstack((ray_origins, ray_origins + ray_directions * ray_length)).reshape(
            -1, 2, 3
        )
    )


def show_mesh(
    elements: np.ndarray,
    nodes: np.ndarray,
    reconstruction: Union[str, Path, trimesh.Trimesh] = None,
    show_legend: bool = False,
) -> None:
    """Show a hexahedral mesh in an interactive window.

    Args:
        elements (np.ndarray): The element definition array containing node IDs.
        nodes (np.ndarray): The cartesian node locations.
        reconstruction (Union[str, Path, trimesh.Trimesh], optional): A triangular
            reconstruction mesh, or the filename with a format that can be imported by
            the trimesh package into a mesh. Defaults to None.
        show_legend (bool): if true, show a legend within the plot window.
    """
    hex_mesh = ensure_hex_mesh((elements, nodes))

    plotter = hex_mesh.plotter
    if reconstruction:
        reconstruction = ensure_trimesh(reconstruction)
        plotter.add_mesh(
            reconstruction,
            color="r",
            opacity=0.5,
            label="reconstruction",
            culling="backface",
        )
    plotter.add_legend() if show_legend else None
    plotter.show()


def ensure_trimesh(
    *potential_meshes: Union[str, Path, trimesh.Trimesh],
    force_list_output: bool = False,
) -> trimesh.Trimesh:
    """Ensure passed parameters are Trimesh objects, converting filenames as necessary.

    Args:
        potential_meshes (Union[str, Path, trimesh.Trimesh]): Trimesh objects, or
            filenames that must be loaded into Trimesh meshes.
        force_list_output (bool): If true, even single object outputs are wrapped in a
            list. Defaults to False.

    Returns:
        trimesh.Trimesh: The trimesh objects.
    """
    potential_meshes = [
        trimesh.load_mesh(Path(pm)) if not isinstance(pm, trimesh.Trimesh) else pm
        for pm in potential_meshes
    ]
    return (
        potential_meshes
        if len(potential_meshes) > 1 or force_list_output
        else potential_meshes[0]
    )


def ensure_hex_mesh(
    *potential_meshes: Union[HexMesh, Tuple[np.ndarray, np.ndarray]],
    force_list_output: bool = False,
) -> HexMesh:
    """Ensure types are HexMesh, converting (element, node) arrays as necessary.

    Args:
        potential_meshes (Union[str, Path, trimesh.Trimesh]): HexMesh objects, or
            tuples of element (cell) and node location definitions that must be
            converted into HexMesh meshes.
        force_list_output (bool): If true, even single object outputs are wrapped in a
            list. Defaults to False.

    Returns:
        HexMesh: The HexMesh objects.
    """
    potential_meshes = list(potential_meshes)
    for idx, pm in enumerate(potential_meshes):
        if not isinstance(pm, HexMesh):
            elements, nodes = pm
            elements = elements if elements.shape[1] == 8 else elements.T
            potential_meshes[idx] = HexMesh(
                cells=elements, nodes=nodes, name="hex mesh"
            )
    return (
        potential_meshes
        if len(potential_meshes) > 1 or force_list_output
        else potential_meshes[0]
    )


def ensure_arrow_glyphs(
    *potential_glyphs: Union[pv.PolyData, Tuple[np.ndarray, np.ndarray]],
    force_list_output: bool = False,
    **ray_glyph_kwargs,
) -> pv.PolyData:
    """Ensure types are PolyData, converting (origins, directions) arrays as necessary.

    Args:
        force_list_output (bool, optional): If true, even single object outputs are
            wrapped in a list. Defaults to False.

    Returns:
        pv.PolyData: The pyvista.PolyData arrow glyphs for use within the pyvista
        ecosystem.
    """
    potential_glyphs = list(potential_glyphs)
    for idx, pg in enumerate(potential_glyphs):
        if not isinstance(pg, pv.PolyData):
            potential_glyphs[idx] = ray_glyphs(*pg, **ray_glyph_kwargs)
    return (
        potential_glyphs
        if len(potential_glyphs) > 1 or force_list_output
        else potential_glyphs[0]
    )


def ensure_point_cloud(
    *potential_clouds: Union[pv.PolyData, trimesh.PointCloud, np.ndarray],
    force_list_output: bool = False,
) -> pv.PolyData:
    """Ensure types are pyvista.PolyData point clouds, converting as necessary.

    Will wrap trimesh.PointCloud objects using pyvista, or convert (n, 3) numpy arrays
    to pyvista PolyData objects.

    Args:
        potential_clouds (PointLike): (n, 3) arrays, trimesh PointCluds, or pyvista
            PolyData objects.
        force_list_output (bool): If true, even single object outputs are wrapped in a
            list. Defaults to False.

    Returns:
        pv.PolyData: The pyvista.PolyData pointclouds for use within the pyvista
        ecosystem.
    """
    potential_clouds = list(potential_clouds)
    for idx, pc in enumerate(potential_clouds):
        if isinstance(pc, trimesh.PointCloud):
            potential_clouds[idx] = pv.wrap(pc)
        elif isinstance(pc, np.ndarray):
            potential_clouds[idx] = pv.PolyData(pc)
    return (
        potential_clouds
        if len(potential_clouds) > 1 or force_list_output
        else potential_clouds[0]
    )


def ensure_pyvista(
    *potential_mesh: Union[
        pv.PolyData,
        pv.UnstructuredGrid,
        HexMesh,
        HexElement,
        trimesh.Trimesh,
        trimesh.PointCloud,
    ],
    force_list_output: bool = False,
) -> Union[pv.PolyData, pv.UnstructuredGrid]:
    """Ensure types are pyvista objects for consistent plotting API.

    Args:
        potential_mesh (UnstructuredGridLike): any object capable of being converted to
            either PolyData or UnstructuredGrid using pyvista.
        force_list_output (bool): If true, even single object outputs are wrapped in a
            list. Defaults to False.

    Returns:
        Union[pv.PolyData, pv.UnstructuredGrid]: The pyvista objects for use in a
        pyvista.Plotter object.
    """
    potential_mesh = list(potential_mesh)
    for idx, pt in enumerate(potential_mesh):
        if isinstance(pt, HexElement) or isinstance(pt, HexMesh):
            potential_mesh[idx] = pt.mesh
        elif isinstance(pt, trimesh.Trimesh) or isinstance(pt, trimesh.PointCloud):
            potential_mesh[idx] = pv.wrap(pt)
    return (
        potential_mesh
        if len(potential_mesh) > 1 or force_list_output
        else potential_mesh[0]
    )
