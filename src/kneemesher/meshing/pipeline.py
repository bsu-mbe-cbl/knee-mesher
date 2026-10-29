# %%
import datetime
import logging
import tomllib  # Update to use tomllib from the standard library
import traceback
import warnings  # noqa
from collections import namedtuple
from collections.abc import Callable
from functools import wraps
from logging.handlers import MemoryHandler, QueueHandler
from multiprocessing import Queue, get_logger

# warnings.simplefilter("error")  # noqa
from pathlib import Path
from typing import Optional, ParamSpec, Tuple, TypeVar

import pyvista as pv  # noqa
import vtk

from kneemesher import __file__ as module_path_str
from kneemesher.errors import (
    DegenerateError,
    ElementQualityError,
    LimitReachedError,
    OptimizationError,
    RaySectorError,
)
from kneemesher.meshing.meshing import (
    mesh_femur_cartilage,
    mesh_meniscus,
    mesh_patella_cartilage,
    mesh_tibia_cartilage,
    mesh_tibia_cartilage_grid,
)
from kneemesher.meshing.progress import WorkProgress, begin_attempt
from kneemesher.utilities import basic_logging, log_top_memory_hogs  # noqa

# %load_ext autoreload
# %autoreload 2

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
logging.getLogger("vtk").disabled = True
P = ParamSpec("P")
T = TypeVar("T")


def _restore_log_handlers(function: Callable[P, T]) -> Callable[P, T]:
    """Release task-local logging handlers even when a worker is interrupted."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        root = logging.getLogger()
        logger = get_logger()
        root_handlers = root.handlers[:]
        handlers = logger.handlers[:]
        root_level, logger_level = root.level, logger.level
        propagate = logger.propagate
        logger.propagate = False
        try:
            return function(*args, **kwargs)
        finally:
            for handler in logger.handlers[:]:
                if handler not in handlers:
                    handler.flush()
                    logger.removeHandler(handler)
                    handler.close()
            for handler in root.handlers[:]:
                if handler not in root_handlers:
                    root.removeHandler(handler)
                    handler.close()
            root.setLevel(root_level)
            logger.setLevel(logger_level)
            logger.propagate = propagate

    return wrapped


# %%
@_restore_log_handlers
def flexible_mesh_femur_cartilage(
    points: Tuple[float, float],
    reconstruction_path=None,
    reconstruction_smoothing_iters: int = 1,
    bone_reconstruction_path=None,
    resolution: Tuple[int, int] = (5, 2),
    do_blending: bool = True,
    early_subdivision: bool = False,
    do_fuse_cartilage_to_bone: bool = False,
    do_blend_fusion: bool = True,
    blending_grade: float = 1,
    minimum_grade_scaling: float = 0.3,
    n_layers_or_slice_locs: int = 4,
    bone_to_joint_displacement_weights: Tuple[float, float] = (0.7, 0.3),
    queue: Optional[Queue] = None,
):
    i_start = datetime.datetime.now()
    qh = QueueHandler(queue) if queue is not None else logging.NullHandler()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(qh)

    mem_handler = MemoryHandler(100, flushLevel=logging.CRITICAL, target=root)
    mem_handler.setLevel(logging.INFO)
    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.addHandler(mem_handler)
    logger.info("Meshing femur cartilage...")

    do_it = True
    blending = do_blending
    blend_fusion = do_blend_fusion
    fuse_cartilage_to_bone = do_fuse_cartilage_to_bone
    have_femur = False
    attempt = 0
    retry_states = set()
    with WorkProgress() as dbar:
        while do_it:
            state = (
                reconstruction_smoothing_iters, fuse_cartilage_to_bone,
                blend_fusion, blending, early_subdivision, minimum_grade_scaling,
            )
            if state in retry_states:
                logger.error("No remaining femur retry settings.")
                break
            retry_states.add(state)
            attempt += 1
            begin_attempt(attempt)
            try:
                femur_cartilage_mesh = mesh_femur_cartilage(
                    reconstruction_path=reconstruction_path,
                    bone_reconstruction_path=bone_reconstruction_path,
                    points=points,
                    radial_longitudinal_resolutions=resolution,
                    blending_grade=blending_grade,
                    reconstruction_smoothing_iters=reconstruction_smoothing_iters,
                    blending=blending,
                    bone_to_joint_displacement_weights=bone_to_joint_displacement_weights,
                    early_subdivision=early_subdivision,
                    fuse_cartilage_to_bone=fuse_cartilage_to_bone,
                    allow_blended_edge_bone_fusion=blend_fusion,
                    minimum_grade_scaling=minimum_grade_scaling,
                    n_layers_or_slice_locs=n_layers_or_slice_locs,
                )
                have_femur = True
                do_it = False
                logger.info("Femur meshing successful.")
            except (OptimizationError, DegenerateError) as e:
                if reconstruction_smoothing_iters == 0:
                    reconstruction_smoothing_iters = 1
                elif fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                elif blend_fusion:
                    blend_fusion = False
                elif not early_subdivision:
                    logger.error(f"{repr(e)}")
                    logger.info("Trying again with early subdivision...")
                    dbar.set_description("Trying again with early subdivision...")
                    early_subdivision = True
                    fuse_cartilage_to_bone = do_fuse_cartilage_to_bone
                    blend_fusion = do_blend_fusion
                else:
                    logger.error(f"{repr(e)}")
                    break
            except (LimitReachedError, ValueError) as e:
                if fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                elif blend_fusion:
                    blend_fusion = False
                elif not early_subdivision:
                    logger.error(f"{repr(e)}")
                    logger.info("Trying again with early subdivision...")
                    dbar.set_description("Trying again with early subdivision...")
                    early_subdivision = True
                    fuse_cartilage_to_bone = do_fuse_cartilage_to_bone
                    blend_fusion = do_blend_fusion
                elif minimum_grade_scaling != 0:
                    logger.error(f"{repr(e)}")
                    logger.info("Trying again without limiting blending retries...")
                    dbar.set_description(
                        "Trying again without limiting blending retries..."
                    )
                    minimum_grade_scaling = 0
                else:
                    logger.exception(f"{traceback.format_exc()}")
                    break
            except ElementQualityError:
                if fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                if blend_fusion:
                    blend_fusion = False
                else:
                    blending = False
            except Exception:
                logger.exception(f"{traceback.format_exc()}")
                break
        if not have_femur:
            logger.warning("Failed to mesh femur_cartilage.")
        i_end = datetime.datetime.now()
        i_tdelta = i_end - i_start
        i_tdelta = i_tdelta - datetime.timedelta(microseconds=i_tdelta.microseconds)
        logger.info(
            f"fuse_cartilage_to_bone:{fuse_cartilage_to_bone}, blend_fusion:{blend_fusion}, blending:{blending}, early_subdivision:{early_subdivision}"
        )
        logger.info(f"femur cartilage meshed in: {i_tdelta}.")
        mem_handler.flush()
        logger.removeHandler(mem_handler)
        mem_handler.close()
        return femur_cartilage_mesh if have_femur else None


@_restore_log_handlers
def flexible_mesh_patella_cartilage(
    points: Tuple[float, float],
    reconstruction_path=None,
    reconstruction_smoothing_iters: int = 0,
    bone_reconstruction_path=None,
    resolution: Tuple[int, int] = (5, 2),
    do_blending: bool = True,
    early_subdivision: bool = False,
    do_fuse_cartilage_to_bone: bool = True,
    do_blend_fusion: bool = True,
    blending_grade: float = 1,
    minimum_grade_scaling: float = 0.3,
    bone_to_joint_displacement_weights: Tuple[float, float] = (0.7, 0.3),
    n_layers_or_slice_locs: int = 4,
    queue: Optional[Queue] = None,
    reverse_blend: Optional[bool] = False,
):
    i_start = datetime.datetime.now()
    qh = QueueHandler(queue) if queue is not None else logging.NullHandler()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(qh)

    mem_handler = MemoryHandler(101, flushLevel=logging.CRITICAL, target=root)
    mem_handler.setLevel(logging.INFO)
    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.addHandler(mem_handler)
    logger.info("Meshing patella cartilage...")

    blending = do_blending
    blend_fusion = do_blend_fusion
    fuse_cartilage_to_bone = do_fuse_cartilage_to_bone
    have_patella = False
    do_it = True
    attempt = 0
    retry_states = set()
    with WorkProgress() as dbar:
        while do_it:
            state = (
                reconstruction_smoothing_iters, fuse_cartilage_to_bone,
                blend_fusion, blending, early_subdivision, minimum_grade_scaling,
            )
            if state in retry_states:
                logger.error("No remaining patella retry settings.")
                break
            retry_states.add(state)
            attempt += 1
            begin_attempt(attempt)
            try:
                patella_cartilage_mesh = mesh_patella_cartilage(
                    reconstruction_path=reconstruction_path,
                    bone_reconstruction_path=bone_reconstruction_path,
                    reconstruction_smoothing_iters=reconstruction_smoothing_iters,
                    points=points,
                    radial_longitudinal_resolutions=resolution,
                    blending_grade=blending_grade,
                    blending=blending,
                    bone_to_joint_displacement_weights=bone_to_joint_displacement_weights,
                    early_subdivision=early_subdivision,
                    fuse_cartilage_to_bone=fuse_cartilage_to_bone,
                    allow_blended_edge_bone_fusion=blend_fusion,
                    minimum_grade_scaling=minimum_grade_scaling,
                    n_layers_or_slice_locs=n_layers_or_slice_locs,
                    reverse_blend=reverse_blend,
                )
                have_patella = True
                do_it = False
                logger.info("Patella meshing successful.")
            except (OptimizationError, DegenerateError) as e:
                if reconstruction_smoothing_iters == 0:
                    reconstruction_smoothing_iters = 1
                elif fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                elif blend_fusion:
                    blend_fusion = False
                elif not early_subdivision:
                    logger.error(f"{repr(e)}")
                    logger.info("Trying again with early subdivision...")
                    dbar.set_description("Trying again with early subdivision...")
                    early_subdivision = True
                    fuse_cartilage_to_bone = do_fuse_cartilage_to_bone
                    blend_fusion = do_blend_fusion
                else:
                    logger.error(repr(e))
                    break
            except LimitReachedError as e:
                if fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                elif blend_fusion:
                    blend_fusion = False
                elif not early_subdivision:
                    logger.error(f"{repr(e)}")
                    logger.info("Trying again with early subdivision...")
                    dbar.set_description("Trying again with early subdivision...")
                    early_subdivision = True
                    fuse_cartilage_to_bone = do_fuse_cartilage_to_bone
                    blend_fusion = do_blend_fusion
                elif minimum_grade_scaling != 0:
                    logger.error(f"{repr(e)}")
                    logger.info("Trying again without limiting blending retries...")
                    dbar.set_description(
                        "Trying again without limiting blending retries..."
                    )
                    minimum_grade_scaling = 0
                else:
                    logger.exception(f"{traceback.format_exc()}")
                    break
            except ElementQualityError:
                if fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                if blend_fusion:
                    blend_fusion = False
                else:
                    blending = False
            except Exception:
                logger.exception(f"{traceback.format_exc()}")
                break
        if not have_patella:
            logger.warning("Failed to mesh patella_cartilage.")
        i_end = datetime.datetime.now()
        i_tdelta = i_end - i_start
        i_tdelta = i_tdelta - datetime.timedelta(microseconds=i_tdelta.microseconds)
        logger.info(
            f"fuse_cartilage_to_bone:{fuse_cartilage_to_bone}, blend_fusion:{blend_fusion}, blending:{blending}, early_subdivision:{early_subdivision}"
        )
        logger.info(f"patella cartilage meshed in: {i_tdelta}.")
        mem_handler.flush()
        logger.removeHandler(mem_handler)
        mem_handler.close()
        return patella_cartilage_mesh if have_patella else None


@_restore_log_handlers
def flexible_mesh_tibia_cartilage(
    elem_counts: Optional[Tuple[int, int, int]] = (13, 10, 9),
    resolution: Tuple[int, int] = (2, 2),
    reconstruction_path=None,
    reconstruction_smoothing_iters: int = 1,
    bone_reconstruction_path=None,
    do_blending: bool = True,
    early_subdivision: bool = False,
    do_fuse_cartilage_to_bone: bool = True,
    do_blend_fusion: bool = True,
    blending_grade: float = 1,
    minimum_grade_scaling: float = 0.3,
    bone_to_joint_displacement_weights: Tuple[float, float] = (0.7, 0.3),
    n_layers_or_slice_locs: int = 4,
    queue: Optional[Queue] = None,
    medial_or_lateral: Optional[str] = None,
):
    i_start = datetime.datetime.now()
    qh = QueueHandler(queue) if queue is not None else logging.NullHandler()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(qh)

    mem_handler = MemoryHandler(102, flushLevel=logging.CRITICAL, target=root)
    mem_handler.setLevel(logging.INFO)
    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.addHandler(mem_handler)
    logger.info(f"Meshing {medial_or_lateral} tibial cartilage...")

    do_it = True
    blending = do_blending
    blend_fusion = do_blend_fusion
    fuse_cartilage_to_bone = do_fuse_cartilage_to_bone
    have_tibia = False
    tib_func = mesh_tibia_cartilage
    attempt = 0
    retry_states = set()
    with WorkProgress() as dbar:
        while do_it:
            state = (
                tib_func, reconstruction_smoothing_iters, fuse_cartilage_to_bone,
                blend_fusion, blending, early_subdivision, minimum_grade_scaling,
            )
            if state in retry_states:
                logger.error("No remaining tibia retry settings.")
                break
            retry_states.add(state)
            attempt += 1
            begin_attempt(attempt)
            try:
                if tib_func is mesh_tibia_cartilage:
                    tibia_cartilage_mesh = tib_func(
                        reconstruction_path=reconstruction_path,
                        bone_reconstruction_path=bone_reconstruction_path,
                        elem_counts=elem_counts,
                        n_triangles=8000,
                        reconstruction_smoothing_iters=reconstruction_smoothing_iters,
                        blending=blending,
                        blending_grade=blending_grade,
                        bone_to_joint_displacement_weights=bone_to_joint_displacement_weights,
                        fuse_cartilage_to_bone=fuse_cartilage_to_bone,
                        allow_blended_edge_bone_fusion=blend_fusion,
                        minimum_grade_scaling=minimum_grade_scaling,
                        n_layers_or_slice_locs=n_layers_or_slice_locs,
                    )
                    logger.info(
                        f"{medial_or_lateral} tibia successfully meshed with SectorCartilageMesh."
                    )
                elif tib_func is mesh_tibia_cartilage_grid:
                    tibia_cartilage_mesh = tib_func(
                        reconstruction_path=reconstruction_path,
                        bone_reconstruction_path=bone_reconstruction_path,
                        resolution=resolution,
                        blending=blending,
                        blending_grade=blending_grade,
                        early_subdivision=early_subdivision,
                        bone_to_joint_displacement_weights=bone_to_joint_displacement_weights,
                        fuse_cartilage_to_bone=fuse_cartilage_to_bone,
                        allow_blended_edge_bone_fusion=blend_fusion,
                        minimum_grade_scaling=minimum_grade_scaling,
                    )
                    logger.info(
                        f"{medial_or_lateral} tibia successfully meshed with GridCartilageMesh."
                    )
                have_tibia = True
                do_it = False
            except (RaySectorError, DegenerateError) as e:
                if fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                elif blend_fusion:
                    blend_fusion = False
                elif tib_func is mesh_tibia_cartilage:
                    logger.error(f"{repr(e)}")
                    dbar.set_description("Trying again with grid-based raytracing...")
                    tib_func = mesh_tibia_cartilage_grid
                    early_subdivision = False
                    fuse_cartilage_to_bone = do_fuse_cartilage_to_bone
                    blend_fusion = do_blend_fusion
                    blending = do_blending
                    minimum_grade_scaling = 0.3
                elif tib_func is mesh_tibia_cartilage_grid and not early_subdivision:
                    early_subdivision = True
                else:
                    logger.error(repr(e))
                    break
            except OptimizationError as e:
                if tib_func is mesh_tibia_cartilage_grid and not early_subdivision:
                    logger.error(f"{repr(e)}")
                    logger.info("Trying again with early subdivision...")
                    dbar.set_description("Trying again with early subdivision...")
                    early_subdivision = True
                elif fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                elif blend_fusion:
                    blend_fusion = False
                elif tib_func is mesh_tibia_cartilage:
                    logger.info("Trying again with grid-based raytracing...")
                    dbar.set_description("Trying again with grid-based raytracing...")
                    tib_func = mesh_tibia_cartilage_grid
                    fuse_cartilage_to_bone = do_fuse_cartilage_to_bone
                    blend_fusion = do_blend_fusion
                    early_subdivision = False
                    blending = do_blending
                    minimum_grade_scaling = 0.3
                else:
                    logger.error(f"{repr(e)}")
                    early_subdivision = False
                    break
            except LimitReachedError as e:
                if not early_subdivision and tib_func is mesh_tibia_cartilage_grid:
                    logger.error(f"{repr(e)}")
                    logger.info("Trying again with early subdivision...")
                    dbar.set_description("Trying again with early subdivision...")
                    early_subdivision = True
                elif fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                elif blend_fusion:
                    blend_fusion = False
                elif minimum_grade_scaling != 0:
                    logger.error(f"{repr(e)}")
                    logger.info("Trying again without limiting blending retries...")
                    dbar.set_description(
                        "Trying again without limiting blending retries..."
                    )
                    minimum_grade_scaling = 0
                else:
                    logger.exception(f"{traceback.format_exc()}")
                    break
            except ElementQualityError:
                if fuse_cartilage_to_bone:
                    fuse_cartilage_to_bone = False
                if blend_fusion:
                    blend_fusion = False
                else:
                    blending = False
            except Exception:
                logger.error(f"{traceback.format_exc()}")
                break

        if not have_tibia:
            logger.warning(f"Failed to mesh {medial_or_lateral} tibia cartilage.")
        i_end = datetime.datetime.now()
        i_tdelta = i_end - i_start
        i_tdelta = i_tdelta - datetime.timedelta(microseconds=i_tdelta.microseconds)
        logger.info(
            f"fuse_cartilage_to_bone:{fuse_cartilage_to_bone}, blend_fusion:{blend_fusion}, blending:{blending}, early_subdivision:{early_subdivision}"
        )
        logger.info(f"{medial_or_lateral} tibial cartilage meshed in: {i_tdelta}.")
        mem_handler.flush()
        logger.removeHandler(mem_handler)
        mem_handler.close()
        return tibia_cartilage_mesh if have_tibia else None


@_restore_log_handlers
def flexible_mesh_meniscus(
    is_medial: bool,
    origin: Tuple[float, float, float] = None,
    reconstruction_path=None,
    reconstruction_smoothing_iters: int = 0,
    bone_reconstruction_path=None,
    tibia_reconstruction_path=None,
    resolution: Tuple[int, int] = (5, 2),
    early_subdivision: bool = True,
    queue: Optional[Queue] = None,
):
    ml_str = "medial" if is_medial else "lateral"
    i_start = datetime.datetime.now()
    qh = QueueHandler(queue) if queue is not None else logging.NullHandler()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(qh)

    mem_handler = MemoryHandler(101, flushLevel=logging.CRITICAL, target=root)
    mem_handler.setLevel(logging.INFO)
    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.addHandler(mem_handler)
    logger.info("Meshing mensicus...")
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    logging.getLogger("vtk").disabled = True

    have_meniscus = False
    do_it = True
    attempt = 0
    retry_states = set()
    with WorkProgress() as dbar:
        while do_it:
            state = (reconstruction_smoothing_iters, early_subdivision)
            if state in retry_states:
                logger.error("No remaining meniscus retry settings.")
                break
            retry_states.add(state)
            attempt += 1
            begin_attempt(attempt)
            try:
                meniscus_mesh = mesh_meniscus(
                    is_medial=is_medial,
                    origin=origin,
                    reconstruction_path=reconstruction_path,
                    resolution=resolution,
                    # early_subdivision=early_subdivision,
                )
                have_meniscus = True
                do_it = False
                logger.info("Patella meshing successful.")
            except (OptimizationError, DegenerateError) as e:
                if reconstruction_smoothing_iters == 1:
                    reconstruction_smoothing_iters = 0
                elif not early_subdivision:
                    logger.error(f"{repr(e)}")
                    break
                    logger.info("Trying again with early subdivision...")
                    dbar.set_description("Trying again with early subdivision...")
                    early_subdivision = True
                else:
                    logger.error(repr(e))
                    break
            except LimitReachedError as e:
                if not early_subdivision:
                    logger.error(f"{repr(e)}")
                    break
                    logger.info("Trying again with early subdivision...")
                    dbar.set_description("Trying again with early subdivision...")
                    early_subdivision = True
            except Exception:
                logger.exception(f"{traceback.format_exc()}")
                break
        if not have_meniscus:
            logger.warning("Failed to mesh meniscus.")
        i_end = datetime.datetime.now()
        i_tdelta = i_end - i_start
        i_tdelta = i_tdelta - datetime.timedelta(microseconds=i_tdelta.microseconds)
        logger.info(f"early_subdivision:{early_subdivision}")
        logger.info(f"meniscus meshed in: {i_tdelta}.")
        mem_handler.flush()
        logger.removeHandler(mem_handler)
        mem_handler.close()
        return meniscus_mesh if have_meniscus else None


# Replace toml.load with tomllib.load in the parse_meshing_config_file function
def parse_meshing_config_file(config_path, **kwargs):
    """Parse a TOML configuration file and return a named tuple of its values.

    Args:
        config_path (str): The path to the TOML file.
        **kwargs: Additional keyword arguments.

    Returns:
        A named tuple of the configuration values.

    """
    boolean_keys = [
        "do_",
        "is_",
        "has_",
        "force_",
        "flip_",
        "append_",
        "prepend_",
        "early_",
        "use_",
        "allow_",
    ]
    float_keys = [
        "blending_grade",
        "minimum_grade_scaling",
        "target_size",
        "_rotation_degree",
        "blending_grade",
    ]
    tuple_keys = [
        "blending_weights",
        "resolution",
        "elem_counts",
        "bone_to_joint_displacement_weights",
        "n_layers_or_slice_locs",
        "search_ignore_patterns",
    ]
    int_keys = [
        "_iters",
        "n_layers_or_slice_locs",
        "_count",
    ]
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    Config = namedtuple("Config", " ".join(data.keys()))
    section_list = []
    for section, section_data in data.items():
        varnames = [varname for varname in section_data.keys()]
        section_dict = {}
        for varname, value in section_data.items():
            varval = value

            # Special handling for directories
            if varname == "home_dir":
                varval = (
                    Path(module_path_str).parents[2]
                    if value.upper() == "REPO_DIR"
                    else Path(value)
                )

            if varname == "import_dir":
                varval = (
                    section_dict["home_dir"].joinpath(value)
                    if not Path(value).is_absolute()
                    else Path(value)
                )

            if varname in ("root_export_dir", "export_root_dir"):
                varval = (
                    section_dict["home_dir"].joinpath(value)
                    if not Path(value).is_absolute()
                    else Path(value)
                )

            if varname == "logging_filename":
                varval = (
                    section_dict["home_dir"].joinpath(value)
                    if not Path(value).is_absolute()
                    else Path(value)
                )

            if varname == "append_datetime_to_log" and value:
                # get the datetime
                time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                # append it to the logging_filename
                logging_filename = Path(section_dict["logging_filename"])
                section_dict["logging_filename"] = (
                    logging_filename.parent
                    / f"{logging_filename.stem}-{time}{logging_filename.suffix}"
                )

            if varname == "prepend_import_dirname_to_log" and value:
                # get the import_dirname
                import_dirname = Path(section_dict["import_dir"]).name

                # prepend it to the logging_filename
                logging_filename = Path(section_dict["logging_filename"])
                section_dict["logging_filename"] = (
                    logging_filename.parent
                    / f"{import_dirname}-{logging_filename.stem}{logging_filename.suffix}"
                )

            if varname == "logging_level":
                varval = getattr(logging, value.upper())

            if any(tuple_key in varname for tuple_key in tuple_keys):
                try:
                    varval = tuple(
                        float(x.strip().strip("()"))
                        for x in section_data[varname].split(",")
                    )
                    varval = tuple([int(x) if x.is_integer() else x for x in varval])
                except ValueError:
                    varval = tuple(
                        x.strip().strip("() ") for x in section_data[varname].split(",")
                    )
                except AttributeError:
                    varval = section_data[varname]

                try:
                    if varname == "n_layers_or_slice_locs" and len(varval) == 1:
                        varval = varval[0]
                    elif varname == "search_ignore_patterns" and len(varval) == 1:
                        varval = varval[0]
                except TypeError:
                    pass

            elif any(float_key in varname for float_key in float_keys):
                varval = float(value)

            elif any(int_keys in varname for int_keys in int_keys):
                varval = int(value)

            elif any(boolean_key in varname for boolean_key in boolean_keys):
                varval = bool(value)

            section_dict[varname] = varval

            if kwargs.get("debug", False):
                print(f"{section}.{varname} = {varval} ({type(varval)})")
        Section = namedtuple(section, " ".join(varnames))
        section_list.append(Section(**section_dict))
    return Config(*section_list)
