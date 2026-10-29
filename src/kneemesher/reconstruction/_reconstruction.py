"""Module for reconstructing knee images into 3D meshes.

This module provides functions and classes to process image data,
apply morphological operations, and generate 3D knee geometries.
"""

from typing import Optional, Tuple

import numpy as np
import pyvista as pv
import vtk
from numpy.typing import ArrayLike

from kneemesher.utilities import with_debug_kwarg
from kneemesher.mesh_repair import RepairBackend, repair_mesh


def image_import(
    data_string: ArrayLike,
    n_channels: int = 1,
    extent: Optional[Tuple[float, float, float, float, float, float]] = None,
    height_spacing: Optional[float] = None,
    width_spacing: Optional[float] = None,
    depth_spacing: Optional[float] = None,
) -> vtk.vtkImageImport:
    """Import image string into VTK object.

    Import a string that represents an image stack into the VTK object system,
    so that other VTK filters may be used on it.

    Args:
        data_string (string): A string representing an image stack for
            importing into VTK.
        n_channels (int, optional): The number of channels this image is
            represented by. Grayscale would be 1 channel, RGB would be 3, and RGBA
            would be represented by 4. Defaults to 1.
        extent (array:float, optional): An array designating the first and last
            pixel in the width, height, and depth directions. Defaults to None.
        height_spacing (float, optional): The spacing between each pixel in the
            height dimension. Defaults to None.
        width_spacing (float, optional): The spacing between each pixel in the
            width dimension. Defaults to None.
        depth_spacing (float, optional): The spacing between each pixel in the
            depth dimension. Defaults to None.

    Returns:
        VtkObject: The VtkObject to be passed on to further VTK filters.
    """
    data_str = np.array(data_string).tobytes()
    data_importer = vtk.vtkImageImport()
    data_importer.CopyImportVoidPointer(data_str, len(data_str))
    data_importer.SetDataScalarTypeToUnsignedChar()
    data_importer.SetNumberOfScalarComponents(n_channels)
    data_importer.SetDataExtent(*extent)
    data_importer.SetWholeExtent(*extent)
    data_importer.SetDataSpacing(height_spacing, width_spacing, depth_spacing)
    # data_importer.SetDataOrigin(np.array((-170.02570717, -111.61828577, 64.01785202)))
    # data_importer.SetDataOrigin(np.array((-111.61828577, -170.02570717, 64.01785202)))
    return data_importer


# Dilate-Erode
def dilate(
    output_port: vtk.vtkAlgorithmOutput, kernel_size: Tuple[int, int, int]
) -> vtk.vtkImageContinuousErode3D:
    """Apply a dilation operation to the input image using a specified kernel size.

    Args:
        output_port (vtk.vtkAlgorithmOutput): The output port of the VTK pipeline
            to which the dilation filter will be applied.
        kernel_size (Tuple[int, int, int]): The size of the kernel to be used
            for the dilation operation. Should be a tuple of three integers.

    Returns:
        vtk.vtkImageContinuousErode3D: The VTK dilation filter configured with
            the specified kernel size.
    """
    kernel_size = list(kernel_size)
    dilater = vtk.vtkImageContinuousErode3D()
    dilater.SetInputConnection(output_port)
    dilater.SetKernelSize(*kernel_size)
    return dilater


def erode(
    output_port: vtk.vtkAlgorithmOutput, kernel_size: Tuple[int, int, int]
) -> vtk.vtkImageContinuousDilate3D:
    """Apply an erosion operation to the input image using a specified kernel size.

    Args:
        output_port (vtk.vtkAlgorithmOutput): The output port of the VTK pipeline
            to which the erosion filter will be applied.
        kernel_size (Tuple[int, int, int]): The size of the kernel to be used
            for the erosion operation. Should be a tuple of three integers.

    Returns:
        vtk.vtkImageContinuousDilate3D: The VTK erosion filter configured with
            the specified kernel size.
    """
    kernel_size = list(kernel_size)
    eroder = vtk.vtkImageContinuousDilate3D()
    eroder.SetInputConnection(output_port)
    eroder.SetKernelSize(*kernel_size)
    return eroder


def dilate_erode(
    output_port: vtk.vtkAlgorithmOutput,
    kernel_size: Tuple[int, int, int],
    dilate_value: int = 1,
    erode_value: int = 0,
) -> vtk.vtkImageDilateErode3D:
    """Apply dilation and erosion to an image using a specified kernel size.

    Args:
        output_port (vtk.vtkAlgorithmOutput): The output port of the VTK pipeline
            to which the filter will be connected.
        kernel_size (Tuple[int, int, int]): The size of the kernel to be used
            for dilation and erosion. Should be a tuple of three integers.
        dilate_value (int, optional): The value to be used for dilation.
            Default is 1.
        erode_value (int, optional): The value to be used for erosion.
            Default is 0.

    Returns:
        vtk.vtkImageDilateErode3D: The VTK image filter configured for
            dilation and erosion.
    """
    kernel_size = list(kernel_size)
    morph_closer = vtk.vtkImageDilateErode3D()
    morph_closer.SetInputConnection(output_port)
    morph_closer.SetDilateValue(dilate_value)
    morph_closer.SetErodeValue(erode_value)
    morph_closer.SetKernelSize(*kernel_size)
    return morph_closer


def preprocess_grid(
    mask_data: ArrayLike, voxel_spacing: Tuple[float, float, float]
) -> vtk.vtkImageImport:
    """Preprocess the grid by importing the image data.

    Args:
        mask_data (ArrayLike): The mask data to be imported.
        voxel_spacing (ArrayLike): The spacing between voxels.

    Returns:
        vtk.vtkImageImport: The imported VTK image.
    """
    shape = mask_data.shape
    return image_import(
        data_string=mask_data,
        n_channels=1,
        extent=(0, shape[1] - 1, 0, shape[2] - 1, 0, shape[0] - 1),
        height_spacing=voxel_spacing[0],
        width_spacing=voxel_spacing[1],
        depth_spacing=voxel_spacing[2],
    )


def apply_morphological_operations(
    grid: vtk.vtkImageImport, tissue_name: str
) -> pv.PolyData:
    """Apply morphological operations (dilate and erode) to the grid.

    Args:
        grid (vtk.vtkImageImport): The VTK image grid.
        tissue_name (str): The name of the tissue.

    Returns:
        pyvista.PolyData: The processed grid.
    """
    dilate_eroded = dilate_erode(grid.GetOutputPort(), kernel_size=(4, 4, 4))
    dilated = dilate(dilate_eroded.GetOutputPort(), kernel_size=(3, 3, 3))
    dilated.Update()
    return pv.wrap(dilated.GetOutput())


def debug_plot(
    grid: vtk.vtkImageImport, g: pv.PolyData, smoothed: Optional[pv.PolyData] = None
) -> None:
    """Plot the grid and optionally the smoothed mesh for debugging.

    Args:
        grid (vtk.vtkImageImport): The VTK image grid.
        g (pyvista.PolyData): The processed grid.
        smoothed (pyvista.PolyData, optional): The smoothed mesh. Defaults to None.
    """
    grid.Update()
    pv_grid = pv.wrap(grid.GetOutput()).contour(isosurfaces=2, scalars="scalars")
    p = pv.Plotter()
    p.add_mesh(pv_grid, opacity=0.5, color="white", label="original")
    p.add_mesh(g, color="blue", opacity=0.5, label="dilated")
    if smoothed is not None:
        p.add_mesh(smoothed, color="red", opacity=0.5, label="smoothed")
    p.add_axes()
    p.add_legend()
    p.show()


def smooth_mesh(
    decimated: pv.PolyData, bone: bool, n_bone_iters: int = 300, n_iters: int = 50
) -> pv.PolyData:
    """Smooth the mesh using Taubin or Laplacian smoothing.

    Args:
        decimated (pyvista.PolyData): The decimated mesh.
        bone (bool): Whether the tissue is bone.

    Returns:
        pyvista.PolyData: The smoothed mesh.
    """
    try:
        if bone:
            return decimated.smooth_taubin(
                n_iter=n_bone_iters, normalize_coordinates=True
            )
        else:
            return decimated.smooth_taubin(n_iter=n_iters, normalize_coordinates=True)
    except AttributeError:
        return decimated.smooth(n_iter=50 if bone else 8)


def postprocess_mesh(
    g: pv.PolyData,
    tissue_name: str,
    bone: bool,
    debug: bool,
    *,
    repair_backend: RepairBackend = "auto",
    **smooth_kwargs,
) -> pv.PolyData:
    """Postprocess the mesh by extracting the largest body and smoothing it.

    Args:
        g (pyvista.PolyData): The processed grid.
        tissue_name (str): The name of the tissue.
        bone (bool): Whether the tissue is bone.
        debug (bool): Whether to enable debugging plots.
        repair_backend: The backend used for surface repair.

    Returns:
        pyvista.PolyData: The postprocessed mesh.
    """
    bodies = g.split_bodies()
    areas = np.array([body.area for body in bodies]).astype(int)
    idx = int(np.argmax(areas))
    largest = bodies[idx].extract_surface(algorithm=None)
    decimated = largest.decimate_pro(0.5) if not bone else largest
    decimated = repair_mesh(decimated, backend=repair_backend, require_closed=True)
    smoothed = smooth_mesh(decimated, bone, **smooth_kwargs)
    if debug:
        debug_plot(largest, g, smoothed)
    if bone:
        smoothed = repair_mesh(smoothed, backend=repair_backend, require_closed=True)

    return smoothed


def transform_mesh(
    smoothed: pv.PolyData,
    tissue_name: str,
    bone: bool,
    *,
    repair_backend: RepairBackend = "auto",
) -> pv.PolyData:
    """Transform the mesh by rotating, translating, and optionally clipping it.

    Args:
        smoothed (pyvista.PolyData): The smoothed mesh.
        tissue_name (str): The name of the tissue.
        bone (bool): Whether the tissue is bone.
        repair_backend: The backend used for surface repair.

    Returns:
        pyvista.PolyData: The transformed mesh.
    """
    transformed = smoothed.rotate_z(-90, inplace=False)
    transformed.rotate_y(90, inplace=True)
    transformed.translate((10, 100, 90), inplace=True)
    if bone:
        z_bounds = transformed.bounds[-2:]
        clip_distance = (z_bounds[1] - z_bounds[0]) * 0.02
        if "femur" in tissue_name.lower():
            clip_plane_point = z_bounds[1] - clip_distance
            normal = (0, 0, -1)
        elif "tibia" in tissue_name.lower() or "fibula" in tissue_name.lower():
            clip_plane_point = z_bounds[0] + clip_distance
            normal = (0, 0, 1)
        if "femur" in tissue_name.lower() or "tibia" in tissue_name.lower():
            transformed = transformed.clip_closed_surface(
                normal=normal,
                origin=(0, 0, clip_plane_point),
            )
        transformed = repair_mesh(
            transformed, backend=repair_backend, require_closed=True
        )
    return transformed


def reconstruct(
    mask_data: ArrayLike,
    tissue_name: str,
    voxel_spacing: Tuple[float, float, float] = (0.36458333, 0.36458333, 0.7),
    bone: bool = False,
    raw_reconstruction: bool = False,
    smooth_kwargs: Optional[dict] = None,
    *,
    repair_backend: RepairBackend = "auto",
    **kwargs,
) -> pv.PolyData:
    """Reconstruct the mesh from the mask data.

    Args:
        mask_data (ArrayLike): The mask data.
        tissue_name (str): The name of the tissue.
        voxel_spacing (ArrayLike, optional): The spacing between voxels. Defaults to (0.36458333, 0.36458333, 0.7).
        bone (bool, optional): Whether the tissue is bone. Defaults to False.
        raw_reconstruction (bool, optional): Whether to perform raw reconstruction. Defaults to False.
        **kwargs: Additional keyword arguments.
        repair_backend: The backend used throughout reconstruction.

    Returns:
        pyvista.PolyData: The reconstructed mesh.
    """
    grid = preprocess_grid(mask_data, voxel_spacing)
    if raw_reconstruction:
        grid.Update()
        g = pv.wrap(grid.GetOutput()).contour(isosurfaces=2, scalars="scalars")
    else:
        g = apply_morphological_operations(grid, tissue_name)
        g = g.contour(isosurfaces=2, scalars="scalars")
    if kwargs.get("debug", False):
        debug_plot(grid, g)
    if g.n_cells == 0:
        return pv.PolyData()
    if raw_reconstruction:
        smoothed = g
    else:
        smoothed = postprocess_mesh(
            g,
            tissue_name,
            bone,
            kwargs.get("debug", False),
            repair_backend=repair_backend,
            **(smooth_kwargs or {}),
        )
    if kwargs.get("debug", False):
        debug_plot(grid, g, smoothed)
    return transform_mesh(smoothed, tissue_name, bone, repair_backend=repair_backend)
