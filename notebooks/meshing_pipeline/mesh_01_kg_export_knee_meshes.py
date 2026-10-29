"""Main file for exporting Knee meshes.

Make sure to edit configs/01-kg-export-knee-meshes.toml first.

Expected folder structure for finding STL files:
    import_dir/subject_dir/left_side/<stl_files>
    import_dir/subject_dir/right_side/<stl_files>

    or

    import_dir/subject_dir/<stl_files>

    > Note: left_or_right side folder is expected to have the words "left" or "right".
    If not, the script will assume the side is "right". There is a configuration option
    to run a batch of left knees.
"""

# %%
import datetime
import logging
import multiprocessing
import shutil
import threading
import tomllib
import warnings  # noqa
from contextlib import contextmanager
from logging.handlers import QueueHandler
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue
from multiprocessing.synchronize import RLock
from pathlib import Path
from typing import Any, Iterator, Tuple
from uuid import uuid4

import numpy as np
import psutil  # noqa
import pyvista as pv  # noqa
import vtk
from tqdm import tqdm

import kneemesher.meshing.pipeline as pipe
import kneemesher.utilities as utils
from kneemesher import __file__ as module_path_str
from kneemesher.errors._errors import MeshPreparationError
from kneemesher.io import write_inp_file
from kneemesher.meshing.meshing import mesh_rigid_body
from kneemesher.meshing.progress import (
    BatchProgress,
    report,
    run_job,
    wait_for_results,
)
from kneemesher.preprocessing import prepare
from kneemesher.utilities import (
    FLIPPED_SUBJECTS,
    LEFT_KNEE_SUBJECTS,
)

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)


config_filename = "01-kg-export-knee-meshes.toml"


def _subject_is_included(
    subject_id: str,
    subject_month_id: str,
    subject_month_side_id: str,
    include: list[str],
) -> bool:
    """Allow every subject when the include list is empty.

    """
    return not include or any(
        query in include
        for query in (
            subject_month_side_id,
            subject_month_id,
            subject_id,
            subject_month_id + "_LR",
        )
    )


def _logger_thread(q):
    """Process log records from a queue until None is encountered.

    Args:
        q (queue.Queue): A thread-safe queue of log records.

    """
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


def _initialize_worker(lock: RLock, queue: Queue) -> None:
    """Capture worker warnings without duplicating tissue logging handlers."""
    tqdm.set_lock(lock)
    warning_logger = logging.getLogger("py.warnings")
    warning_logger.handlers = [QueueHandler(queue)]
    warning_logger.setLevel(logging.WARNING)
    warning_logger.propagate = False
    logging.captureWarnings(True)


def rotate_pv_object(
    pv_object: pv.PolyData,
    rotation_order: Tuple[str, str, str] = ("z", "x", "y"),
    x_rotation: float = 0.0,
    y_rotation: float = 0.0,
    z_rotation: float = 0.0,
) -> pv.PolyData:
    """Rotate a PyVista object around the x, y, and z axes in the specified order.

    Args:
        pv_object: The PyVista object to be rotated.
        rotation_order: The order in which to apply the rotations. Default is ('z', 'x',
            'y').
        x_rotation: The angle of rotation around the x-axis in degrees. Default is 0.0.
        y_rotation: The angle of rotation around the y-axis in degrees. Default is 0.0.
        z_rotation: The angle of rotation around the z-axis in degrees. Default is 0.0.

    Returns:
        The rotated PyVista object.

    """
    m = pv_object.copy()
    rotation_angle_dict = {
        "x": x_rotation,
        "y": y_rotation,
        "z": z_rotation,
    }
    rotation_order_dict = {
        "x": m.rotate_x,
        "y": m.rotate_y,
        "z": m.rotate_z,
    }
    for rotation_key in rotation_order:
        rotation_angle = rotation_angle_dict[rotation_key]
        if rotation_angle != 0:
            rotation_method = rotation_order_dict[rotation_key]
            rotation_method(rotation_angle_dict[rotation_key], inplace=True)
    return m


def save_rigid_body(
    bone_stl,
    savepath,
    reflect_normal=None,
    reflect_point=None,
    target_size=3.0,
    triangle_type="R3D3",
    node_start_num=1,
    elem_start_num=1,
    **rotate_pv_object_kwargs,
):
    """Save a rigid body mesh to an INP file.

    Args:
        bone_stl: The input STL file path.
        savepath: The output INP file path.
        reflect_normal: A tuple of three floats representing the normal vector of the
            plane of reflection.
        reflect_point: A tuple of three floats representing a point on the plane of
            reflection.
        target_size: The target size of the mesh.
        triangle_type: The type of triangle to use.
        **rotate_pv_object_kwargs: Additional keyword arguments to pass to the
            *`rotate_pv_object` function.

    Returns:
        The mesh object.

    """
    mesh = mesh_rigid_body(bone_stl, target_size=target_size)
    report("Rotating rigid mesh")
    mesh = rotate_pv_object(mesh, **rotate_pv_object_kwargs)
    if "fem" in bone_stl.stem.lower():
        node_start = 1100100
        elem_start = 1100100
    elif "pat" in bone_stl.stem.lower():
        node_start = 3100100
        elem_start = 3100100
    elif "tib" in bone_stl.stem.lower():
        node_start = 2100100
        elem_start = 2100100
    else:
        node_start = 5100100
        elem_start = 5100100

    report("Writing rigid INP")
    write_inp_file(
        mesh,
        savepath,
        triangle_type=triangle_type,
        reflect_normal=reflect_normal,
        reflect_point=reflect_point,
        node_num_begins=node_start_num,
        elem_num_begins=elem_start_num,
    )
    return mesh


def parse_config():
    """Load and parse configuration data for knee mesh export."""
    config_dir = Path(module_path_str).parents[2].joinpath("configs")
    assert config_dir.is_dir()
    config_path = config_dir.joinpath(config_filename)
    return pipe.parse_meshing_config_file(config_path)
    with open(config_path, "rb") as f:
        return pipe.parse_meshing_config_file(tomllib.load(f))


class KneeMeshExporter:
    """Export knee meshes from STL files."""

    def __init__(self, config):
        """Initialize the KneeMeshExporter object.

        Args:
            config: The configuration object.

        """
        self.config = config
        self.dirs_and_logs = config.directories_and_logging
        self.bone_kwargs = config.bones._asdict()
        self.STL_IMPORT_FILE_STEMS = config.stl_import_stems._asdict()
        self.search_pattern = self.STL_IMPORT_FILE_STEMS.pop("search_pattern")
        self.search_ignore_patterns = self.STL_IMPORT_FILE_STEMS.pop(
            "search_ignore_patterns"
        )
        self.INP_EXPORT_FILE_STEMS = config.inp_export_stems._asdict()
        self.fem_cart_kwargs = config.femur_cartilage._asdict()
        self.pat_cart_kwargs = config.patella_cartilage._asdict()
        self.tibia_cart_med_kwargs = config.tibia_cartilage_medial._asdict()
        self.tibia_cart_lat_kwargs = config.tibia_cartilage_lateral._asdict()

    def prepare_stl_files(self):
        pass

    def process_bones(
        self, bone_stls, pool, q, tbar, left_right_flip_normal, left_right_flip_center
    ):
        """Process bone meshes and save them to INP files.

        Args:
            bone_stls: List of bone STL file paths.
            pool: Multiprocessing pool for parallel processing.
            q (queue.Queue): Queue for logging.
            tbar (tqdm): Progress bar object.
            left_right_flip_normal: Normal vector for optional reflection plane.
            left_right_flip_center: Center point for optional reflection plane.

        """
        processes = [None] * len(bone_stls)
        keys = [None] * len(bone_stls)
        for proc_idx, bone_stl in enumerate(bone_stls):
            save_dir = bone_stl.parent / self.dirs_and_logs.export_dirname
            save_dir.mkdir(exist_ok=True)
            inp_key = [
                k for k, v in self.STL_IMPORT_FILE_STEMS.items() if v in bone_stl.stem
            ][0]
            savename = f"{self.INP_EXPORT_FILE_STEMS[inp_key]}.inp"

            keys[proc_idx] = inp_key
            savepath = save_dir / savename
            tbar.set_description(f"saving {savename}...")
            processes[proc_idx] = pool.apply_async(
                save_rigid_body,
                (bone_stl, savepath),
                kwds=dict(
                    reflect_normal=left_right_flip_normal,
                    reflect_point=left_right_flip_center,
                    z_rotation=-self.config.rotations.z_rotation_degree,
                    **self.bone_kwargs,
                ),
            )

        results = [p.get() for p in processes]
        for key, result in zip(keys, results):
            self.meshes[key] = result

    def save_bone_meshes(self):
        pass

    def load_existing_bone_meshes(self):
        pass

    def process_femur_cartilage(self, femur_cylinder_points, pool, q, tbar):
        """Process femur cartilage and save it to INP files.

        Args:
            femur_cylinder_points: Points for femur cylinder.
            pool: Multiprocessing pool for parallel processing.
            q (queue.Queue): Queue for logging.
            tbar (tqdm): Progress bar object.

        """
        if self.config.tissue_selections.do_femur_cart:
            tbar.set_description("Meshing femoral cartilage...")
            process = pool.apply_async(
                pipe.flexible_mesh_femur_cartilage,
                kwds=dict(
                    points=femur_cylinder_points,
                    reconstruction_path=self.tissues["femur_cartilage"],
                    bone_reconstruction_path=self.meshes.get("femur_bone", None),
                    queue=q,
                    **self.fem_cart_kwargs,
                ),
            )
            self.meshes["femur_cartilage"] = process.get()

    def process_patella_cartilage(self, patella_cylinder_points, pool, q, tbar):
        """Process patella cartilage and save it to INP files.

        Args:
            patella_cylinder_points: Points for patella cylinder.
            pool: Multiprocessing pool for parallel processing.
            q (queue.Queue): Queue for logging.
            tbar (tqdm): Progress bar object.

        """
        if self.config.tissue_selections.do_patella_cart:
            tbar.set_description("Meshing patellar cartilage...")
            process = pool.apply_async(
                pipe.flexible_mesh_patella_cartilage,
                kwds=dict(
                    points=patella_cylinder_points,
                    reconstruction_path=self.tissues["patella_cartilage"],
                    bone_reconstruction_path=self.meshes.get("patella_bone", None),
                    queue=q,
                    **self.pat_cart_kwargs,
                ),
            )
            self.meshes["patella_cartilage"] = process.get()

    def process_tibia_cartilage(self, pool, q, tbar):
        """Process tibia cartilage and save it to INP files.

        Args:
            pool: Multiprocessing pool for parallel processing.
            q (queue.Queue): Queue for logging.
            tbar (tqdm): Progress bar object.

        """
        if self.config.tissue_selections.do_tibia_cart_lateral:
            tbar.set_description("Meshing lateral tibial cartilage...")
            process = pool.apply_async(
                pipe.flexible_mesh_tibia_cartilage,
                kwds=dict(
                    reconstruction_path=self.tissues["tibia_cartilage_lateral"],
                    bone_reconstruction_path=self.meshes.get("tibia_bone", None),
                    queue=q,
                    **self.tibia_cart_lat_kwargs,
                ),
            )
            self.meshes["tibia_cartilage_lateral"] = process.get()

        if self.config.tissue_selections.do_tibia_cart_medial:
            tbar.set_description("Meshing medial tibial cartilage...")
            process = pool.apply_async(
                pipe.flexible_mesh_tibia_cartilage,
                kwds=dict(
                    reconstruction_path=self.tissues["tibia_cartilage_medial"],
                    bone_reconstruction_path=self.meshes.get("tibia_bone", None),
                    queue=q,
                    **self.tibia_cart_med_kwargs,
                ),
            )
            self.meshes["tibia_cartilage_medial"] = process.get()

    def process_meniscus_cartilage(self, mass_centers, pool, q, tbar, swap_meniscus_ml):
        """Process meniscus cartilage and save it to INP files.

        Args:
            mass_centers: Mass centers for meniscus.
            pool: Multiprocessing pool for parallel processing.
            q (queue.Queue): Queue for logging.
            tbar (tqdm): Progress bar object.
            swap_meniscus_ml: If True, meniscus medial/lateral swap is needed.

        """
        if self.config.tissue_selections.do_meniscus_lateral:
            tbar.set_description("Meshing lateral meniscus...")
            process = pool.apply_async(
                pipe.flexible_mesh_meniscus,
                kwds=dict(
                    is_medial=False if not swap_meniscus_ml else True,
                    origin=mass_centers["meniscus_lateral"],
                    reconstruction_path=self.tissues["meniscus_lateral"],
                    bone_reconstruction_path=self.meshes.get("tibia_bone", None),
                    queue=q,
                    **self.config.meniscus_lateral._asdict(),
                ),
            )
            self.meshes["meniscus_lateral"] = process.get()

        if self.config.tissue_selections.do_meniscus_medial:
            tbar.set_description("Meshing medial meniscus...")
            process = pool.apply_async(
                pipe.flexible_mesh_meniscus,
                kwds=dict(
                    is_medial=True if not swap_meniscus_ml else False,
                    origin=mass_centers["meniscus_medial"],
                    reconstruction_path=self.tissues["meniscus_medial"],
                    bone_reconstruction_path=self.meshes.get("tibia_bone", None),
                    queue=q,
                    **self.config.meniscus_medial._asdict(),
                ),
            )
            self.meshes["meniscus_medial"] = process.get()

    def save_cartilage_meshes(self, tbar):
        pass

    def run(self):
        pass


def _requested_tissues(config: Any, stems: dict[str, str]) -> list[str]:
    """List selected outputs, including tissues whose input files are missing."""
    selections = config.tissue_selections
    requested = [key for key in stems if "bone" in key] if selections.do_bones else []
    for option, key in (
        ("do_femur_cart", "femur_cartilage"),
        ("do_patella_cart", "patella_cartilage"),
        ("do_tibia_cart_lateral", "tibia_cartilage_lateral"),
        ("do_tibia_cart_medial", "tibia_cartilage_medial"),
        ("do_meniscus_lateral", "meniscus_lateral"),
        ("do_meniscus_medial", "meniscus_medial"),
    ):
        if getattr(selections, option):
            requested.append(key)
    return requested


@contextmanager
def _batch_resources(context: Any, total: int) -> Iterator[tuple[Any, ...]]:
    """Keep workers, warning listener, event queue, and scratch data scoped."""
    work_dir = Path.cwd() / f".mesh-export-{uuid4().hex}"
    work_dir.mkdir()
    was_capturing = logging._warnings_showwarning is not None
    logging.captureWarnings(True)
    try:
        # A relative socket path stays within the Unix socket path-length limit.
        address = str(work_dir.relative_to(Path.cwd()) / "manager.sock")
        with SyncManager(address=address, ctx=context) as manager:
            logs = manager.Queue()
            events = manager.Queue()
            listener = threading.Thread(target=_logger_thread, args=(logs,))
            listener.start()
            progress = None
            try:
                progress = BatchProgress(total, events)
                yield progress, logs, events, work_dir
            finally:
                try:
                    if progress is not None:
                        progress.close()
                finally:
                    logs.put(None)
                    listener.join()
    finally:
        try:
            shutil.rmtree(work_dir)
        finally:
            if not was_capturing:
                logging.captureWarnings(False)


@contextmanager
def _knee_scope(
    context: Any,
    progress: BatchProgress,
    logs: Any,
    config: Any,
    stl_path: Path,
    stems: dict[str, str],
    successful: list[str],
    unsuccessful: list[str],
    unsuccessful_tissues: list[list[str]],
) -> Iterator[Any]:
    """Count each sliced knee once and always stop its worker processes."""
    rel_path = stl_path.relative_to(config.directories_and_logging.import_dir)
    subject = rel_path.parts[0]
    month = "_".join(rel_path.parts[:2])
    knee = month + ("_L" if "left" in str(stl_path).lower() else "_R")
    selected = knee not in config.subjects.exclude and _subject_is_included(
        subject, month, knee, config.subjects.include
    )
    requested = _requested_tissues(config, stems) if selected else []
    progress.start_knee(knee, requested)
    pool = None
    interrupted = False
    try:
        if selected and requested:
            pool = context.Pool(
                min(6, psutil.cpu_count(logical=False) or 1),
                initializer=_initialize_worker,
                initargs=(tqdm.get_lock(), logs),
            )
            pool._mesh_worker_pids = {worker.pid for worker in pool._pool}
        yield pool
    except Exception as error:
        logging.exception("Failed to export %s", knee)
        for tissue in requested:
            progress.tissue(tissue, f"Failed: {type(error).__name__}", status="failed")
    except BaseException:
        interrupted = True
        for tissue in requested:
            progress.tissue(tissue, "Cancelled", status="cancelled")
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        progress.drain()
        if interrupted:
            pass
        elif not selected or not requested:
            progress.finish_knee("skipped")
        elif all(event.status == "exported" for event in progress.events.values()):
            successful.append(knee)
            progress.finish_knee("exported")
        else:
            unsuccessful.append(knee)
            unsuccessful_tissues.append(
                [
                    tissue
                    for tissue, event in progress.events.items()
                    if event.status != "exported"
                ]
            )
            progress.finish_knee("failed")


def _submit(
    pool: Any,
    progress: BatchProgress,
    tissue: str,
    function: Any,
    args: tuple[Any, ...] = (),
    kwds: dict[str, Any] | None = None,
) -> Any:
    """Submit a picklable job with its own progress identity."""
    return pool.apply_async(
        run_job, (progress.queue, progress.knee, tissue, function, args, kwds or {})
    )


def _save_meshes(
    meshes: dict[str, Any],
    tissues: dict[str, Path],
    config: Any,
    stems: dict[str, str],
    save_dir: Path,
    temp_dir: Path,
    flip_normal: Any,
    flip_center: Any,
    progress: BatchProgress,
) -> None:
    """Write cartilage files and confirm requested outputs in their destination."""
    save_dir.mkdir(exist_ok=True)
    for key in meshes:
        if "bone" in key:
            continue
        progress.tissue(key, "Writing INP")
        node_sets = None
        element_sets = None
        if "meniscus" not in key:
            node_sets = {
                f"{key.upper()}_BONE_SIDE_NODES": meshes[key].field_data[
                    "bone_side_pids"
                ],
                f"{key.upper()}_JOINT_SIDE_NODES": meshes[key].field_data[
                    "joint_side_pids"
                ],
            }
            element_sets = {
                f"{key.upper()}_{name.upper()}_ELEMS": value
                for name, value in meshes[key].field_data.items()
                if "layer_" in name
            }
            meshes[key]._mark_bone_joint_interfaces()
        else:
            is_lateral = "lateral" in key
            is_left = "left" in str(save_dir)
            if (is_lateral and is_left) or (not is_lateral and not is_left):
                this_meniscus = meshes[key].copy()
                meshes[key].field_data["posterior_horn_pids"] = (
                    this_meniscus.field_data["anterior_horn_pids"]
                )
                meshes[key].field_data["anterior_horn_pids"] = this_meniscus.field_data[
                    "posterior_horn_pids"
                ]
            node_sets = {
                f"{key.upper()}_POSTERIOR_HORN_NODES": meshes[key].field_data[
                    "posterior_horn_pids"
                ],
                f"{key.upper()}_ANTERIOR_HORN_NODES": meshes[key].field_data[
                    "anterior_horn_pids"
                ],
            }
        meshes[key] = rotate_pv_object(
            meshes[key],
            rotation_order=reversed(config.rotations.rotation_order),
            x_rotation=-config.rotations.x_rotation_degree,
            y_rotation=-config.rotations.y_rotation_degree,
            z_rotation=-config.rotations.z_rotation_degree,
        )
        write_inp_file(
            meshes[key],
            save_dir / f"{stems[key]}.inp",
            hexahedron_type=config.cartilage.hexahedron_type,
            reflect_normal=flip_normal,
            reflect_point=flip_center,
            node_sets=node_sets,
            elem_sets=element_sets,
            node_num_begins=config.node_number_starts._asdict().get(key, 1),
            elem_num_begins=config.element_number_starts._asdict().get(key, 1),
        )

    destination = config.directories_and_logging.root_export_dir / save_dir.relative_to(
        temp_dir
    )
    destination.mkdir(parents=True, exist_ok=True)
    for source in save_dir.iterdir():
        target = destination / source.name
        key = next(
            (key for key in progress.rows if f"{stems[key]}.inp" == source.name), None
        )
        if key is not None:
            progress.tissue(key, "Moving to final destination")
        if target.exists():
            target.unlink()
        shutil.move(source, target)
        if key is not None and key in meshes and target.is_file():
            progress.tissue(key, "Exported", status="exported")


def run_main():
    """Main execution flow for exporting knee meshes."""
    kwargs = {}
    multiprocessing.freeze_support()
    # Native geometry libraries start threads during forkserver preloading.
    context = multiprocessing.get_context("spawn")
    logging.getLogger("shapely.geos").disabled = True
    logging.getLogger("vtk").disabled = True

    pv.set_plot_theme("document")

    # %%

    config = parse_config()
    dirs_and_logs = config.directories_and_logging
    bone_kwargs = config.bones._asdict()
    STL_IMPORT_FILE_STEMS = config.stl_import_stems._asdict()
    search_pattern = STL_IMPORT_FILE_STEMS.pop("search_pattern")
    search_ignore_patterns = STL_IMPORT_FILE_STEMS.pop("search_ignore_patterns")
    INP_EXPORT_FILE_STEMS = config.inp_export_stems._asdict()

    fem_cart_kwargs = config.femur_cartilage._asdict()
    pat_cart_kwargs = config.patella_cartilage._asdict()
    tibia_cart_med_kwargs = config.tibia_cartilage_medial._asdict()
    tibia_cart_lat_kwargs = config.tibia_cartilage_lateral._asdict()
    men_med_kwargs = config.meniscus_medial._asdict()
    men_lat_kwargs = config.meniscus_lateral._asdict()
    for option, value in config.cartilage_blending._asdict().items():
        for x in (
            fem_cart_kwargs,
            pat_cart_kwargs,
            tibia_cart_med_kwargs,
            tibia_cart_lat_kwargs,
        ):
            x.setdefault(option, value)

    assert dirs_and_logs.import_dir.is_dir(), (
        f"import_dir {dirs_and_logs.import_dir} is not a directory."
    )

    logging.basicConfig(
        filename=dirs_and_logs.logging_filename,
        level=dirs_and_logs.logging_level,
        filemode="w",
    )
    stl_list = sorted(dirs_and_logs.import_dir.rglob(f"*{search_pattern}*.stl"))
    for stl in stl_list:
        for ignore_pattern in search_ignore_patterns:
            if ignore_pattern in stl.stem:
                stl_list.remove(stl)
                break

    knee_index_slice_config = config.knee_index_slice._asdict()
    start = knee_index_slice_config.get("start", None)
    step = knee_index_slice_config.get("step", 1)
    stop = knee_index_slice_config.get("stop", None)

    knee_index_slice = slice(start, stop, step)

    # %%
    successful = []
    unsuccessful = []
    unsuccessful_tissues = []
    start_time = datetime.datetime.now()
    stl_list = stl_list[knee_index_slice]

    print(f"\nImport directory is {dirs_and_logs.import_dir}.")
    logging.info(f"\nImport directory is {dirs_and_logs.import_dir}.")
    tqdm.set_lock(context.RLock())
    with _batch_resources(context, len(stl_list)) as (progress, q, events, td):
        for stl_idx, stl_path in enumerate(stl_list):
            with _knee_scope(
                context, progress, q, config, stl_path, STL_IMPORT_FILE_STEMS,
                successful, unsuccessful, unsuccessful_tissues,
            ) as pool:
                if pool is None:
                    continue
                temp_dir = Path(td)

                rel_path = stl_path.relative_to(dirs_and_logs.import_dir)

                temp_stl_path = temp_dir / rel_path

                subject_id = rel_path.parts[0]
                subject_month_id = "_".join(rel_path.parts[:2])
                subject_month_side_id = "_".join(
                    [
                        subject_month_id,
                        "L" if "left" in str(temp_stl_path).lower() else "R",
                    ]
                )
                if (
                    config.subjects.exclude
                    and subject_month_side_id in config.subjects.exclude
                ):
                    logging.info(
                        f"Skipping {subject_month_side_id} due to exclusion list."
                    )
                    continue
                if not _subject_is_included(
                    subject_id,
                    subject_month_id,
                    subject_month_side_id,
                    config.subjects.include,
                ):
                    logging.info(
                        f"Skipping {subject_month_side_id} due to inclusion list."
                    )
                    continue
                # Copy the stem directory structure to the temp_dir
                source_dir = dirs_and_logs.import_dir / rel_path.parts[0]
                dest_dir = temp_dir / rel_path.parts[0]
                if not dest_dir.is_dir():
                    shutil.copytree(source_dir, dest_dir)

                i_start = datetime.datetime.now()
                if (
                    (  # original needs to be flipped
                        subject_id in LEFT_KNEE_SUBJECTS
                        and subject_month_id not in FLIPPED_SUBJECTS
                    )
                    or (  # corrected needs flipped
                        subject_id not in LEFT_KNEE_SUBJECTS
                        and subject_month_id in FLIPPED_SUBJECTS
                    )
                    or "left" in str(temp_stl_path).lower()
                    or config.batch_processing.is_left_knee_batch
                ):
                    if config.batch_processing.flip_left_knees_to_right:
                        left_right_flip_normal = (1, 0, 0)
                        left_right_flip_center = (0, 0, 0)
                    else:
                        left_right_flip_normal, left_right_flip_center = None, None
                    swap_meniscus_ml = True
                else:
                    left_right_flip_normal, left_right_flip_center = None, None
                    swap_meniscus_ml = False

                logging.info(f"Processing {'_'.join(temp_stl_path.parts[-3:-1])}...")

                knee_tissue_stls = sorted(temp_stl_path.parent.glob("*.stl"))
                knee_tissue_stls = [
                    x for x in knee_tissue_stls if "raw_marching_cubes" not in str(x)
                ]
                hexa_stls = []
                bone_stls = []
                for kts in knee_tissue_stls:
                    if any(
                        v in kts.stem
                        for k, v in STL_IMPORT_FILE_STEMS.items()
                        if "bone" in k
                    ):
                        bone_stls.append(kts)
                    else:
                        hexa_stls.append(kts)
                tissues = {}
                for stl_key, stl_stem in STL_IMPORT_FILE_STEMS.items():
                    has_match = [stl_stem in x.stem for x in knee_tissue_stls]
                    if any(has_match):
                        tissues[stl_key] = np.array(knee_tissue_stls)[has_match][0]
                for tissue_key in progress.rows:
                    if tissue_key not in tissues:
                        progress.tissue(
                            tissue_key, "Missing STL input", status="failed"
                        )

                # Handle the rotations
                for tissue_key, geom_path in tissues.items():
                    if tissue_key in progress.rows:
                        progress.tissue(tissue_key, "Preparing reconstruction")
                    m = pv.read(geom_path)
                    needs_save = False
                    if any([x != 0 for x in config.rotations[1:]]):
                        if tissue_key in progress.rows:
                            progress.tissue(tissue_key, "Rotating reconstruction")
                        m = rotate_pv_object(
                            pv.read(geom_path),
                            rotation_order=config.rotations.rotation_order,
                            x_rotation=config.rotations.x_rotation_degree,
                            y_rotation=config.rotations.y_rotation_degree,
                            z_rotation=config.rotations.z_rotation_degree,
                        )
                        needs_save = True
                    if not m.is_manifold:
                        if tissue_key in progress.rows:
                            progress.tissue(tissue_key, "Repairing reconstruction")
                        m = utils.repair_mesh(m)
                        needs_save = True if m.is_manifold else False
                        if not needs_save:
                            logging.error("Mesh is not manifold.")
                    if needs_save:
                        if tissue_key in progress.rows:
                            progress.tissue(tissue_key, "Saving prepared reconstruction")
                        m.save(geom_path)

                meshes = {}
                requested_tissue_keys = []
                if config.tissue_selections.do_bones:
                    processes = [None] * len(bone_stls)
                    keys = [None] * len(bone_stls)
                    for proc_idx, bone_stl in enumerate(bone_stls):
                        save_dir = bone_stl.parent / dirs_and_logs.export_dirname
                        save_dir.mkdir(exist_ok=True)
                        inp_key = [
                            k
                            for k, v in STL_IMPORT_FILE_STEMS.items()
                            if v in bone_stl.stem
                        ][0]
                        savename = f"{INP_EXPORT_FILE_STEMS[inp_key]}.inp"

                        keys[proc_idx] = inp_key
                        requested_tissue_keys.append(inp_key)
                        savepath = save_dir / savename
                        processes[proc_idx] = _submit(
                            pool, progress, inp_key,
                            save_rigid_body,
                            (bone_stl, savepath),
                            kwds=dict(
                                reflect_normal=left_right_flip_normal,
                                reflect_point=left_right_flip_center,
                                z_rotation=-config.rotations.z_rotation_degree,
                                node_start_num=config.node_number_starts._asdict().get(
                                    inp_key, 1
                                ),
                                elem_start_num=config.element_number_starts._asdict().get(
                                    inp_key, 1
                                ),
                                **bone_kwargs,
                            ),
                        )

                    if kwargs.get("debug", False):
                        save_rigid_body(
                            bone_stl,
                            savepath,
                            reflect_normal=left_right_flip_normal,
                            reflect_point=left_right_flip_center,
                            z_rotation=-config.rotations.z_rotation,
                        )

                    results = wait_for_results(
                        list(zip(keys, processes)), progress, pool
                    )
                    for key, result in zip(keys, results):
                        if result is not None:
                            meshes[key] = result

                else:
                    for proc_idx, bone_stl in enumerate(bone_stls):
                        save_dir = bone_stl.parent / dirs_and_logs.export_dirname
                        mesh_export_dir = (
                            dirs_and_logs.root_export_dir / save_dir.relative_to(temp_dir)
                        )
                        save_dir = temp_stl_path.parent / dirs_and_logs.export_dirname
                        inp_key = [
                            k
                            for k, v in STL_IMPORT_FILE_STEMS.items()
                            if v in bone_stl.stem
                        ][0]
                        savename = f"{INP_EXPORT_FILE_STEMS[inp_key]}.inp"
                        save_dir.mkdir(exist_ok=True)
                        try:
                            meshes[inp_key] = pv.read(
                                mesh_export_dir / savename, file_format="abaqus"
                            )
                        except FileNotFoundError:
                            pass
                try:
                    logging.info("\tPreparing mesh...")
                    for tissue in progress.rows:
                        if "bone" not in tissue:
                            progress.tissue(tissue, "Preparing landmarks")
                    (
                        femur_cylinder_points,
                        patella_cylinder_points,
                        mass_centers,
                    ) = prepare(**tissues)
                except MeshPreparationError as e:
                    logging.error(f" {repr(e)}")
                    continue
                except KeyError as e:
                    found_files = ", ".join([x.name for x in hexa_stls])
                    logging.error(
                        f" {repr(e)} Only located {len(hexa_stls)} files: \n\t"
                    )
                    logging.error("\n\t".join([x.name for x in hexa_stls]))
                    continue
                except ValueError as e:
                    logging.error("Suspect STL files are mislabeled.")
                    logging.error(f" {repr(e)}")
                    continue
                have_femur, have_patella, have_tibia_lateral, have_tibia_medial = [
                    False
                ] * 4

                # Femur
                processes = []
                meshed_cartilage_keys = []
                if config.tissue_selections.do_femur_cart:
                    requested_tissue_keys.append("femur_cartilage")
                    processes.append(
                        _submit(
                            pool, progress, "femur_cartilage",
                            pipe.flexible_mesh_femur_cartilage,
                            kwds=dict(
                                points=femur_cylinder_points,
                                reconstruction_path=tissues["femur_cartilage"],
                                bone_reconstruction_path=meshes.get("femur_bone", None),
                                queue=q,
                                **fem_cart_kwargs,
                            ),
                        )
                    )
                    meshed_cartilage_keys.append("femur_cartilage")

                # Patella
                if config.tissue_selections.do_patella_cart:
                    requested_tissue_keys.append("patella_cartilage")
                    processes.append(
                        _submit(
                            pool, progress, "patella_cartilage",
                            pipe.flexible_mesh_patella_cartilage,
                            kwds=dict(
                                points=patella_cylinder_points,
                                reconstruction_path=tissues["patella_cartilage"],
                                bone_reconstruction_path=meshes.get(
                                    "patella_bone", None
                                ),
                                queue=q,
                                **pat_cart_kwargs,
                            ),
                        )
                    )
                    meshed_cartilage_keys.append("patella_cartilage")

                # Tibia
                if config.tissue_selections.do_tibia_cart_lateral:
                    requested_tissue_keys.append("tibia_cartilage_lateral")
                    processes.append(
                        _submit(
                            pool, progress, "tibia_cartilage_lateral",
                            pipe.flexible_mesh_tibia_cartilage,
                            kwds=dict(
                                reconstruction_path=tissues["tibia_cartilage_lateral"],
                                bone_reconstruction_path=meshes.get("tibia_bone", None),
                                queue=q,
                                **tibia_cart_lat_kwargs,
                            ),
                        )
                    )
                    meshed_cartilage_keys.append("tibia_cartilage_lateral")

                if config.tissue_selections.do_tibia_cart_medial:
                    requested_tissue_keys.append("tibia_cartilage_medial")
                    processes.append(
                        _submit(
                            pool, progress, "tibia_cartilage_medial",
                            pipe.flexible_mesh_tibia_cartilage,
                            kwds=dict(
                                reconstruction_path=tissues["tibia_cartilage_medial"],
                                bone_reconstruction_path=meshes.get("tibia_bone", None),
                                queue=q,
                                **tibia_cart_med_kwargs,
                            ),
                        )
                    )
                    meshed_cartilage_keys.append("tibia_cartilage_medial")

                # Meniscus NOTE: Meniscus meshing is experimental
                if config.tissue_selections.do_meniscus_lateral:
                    requested_tissue_keys.append("meniscus_lateral")
                    processes.append(
                        _submit(
                            pool, progress, "meniscus_lateral",
                            pipe.flexible_mesh_meniscus,
                            kwds=dict(
                                is_medial=False if not swap_meniscus_ml else True,
                                origin=mass_centers["meniscus_lateral"],
                                reconstruction_path=tissues["meniscus_lateral"],
                                bone_reconstruction_path=meshes.get("tibia_bone", None),
                                queue=q,
                                **men_lat_kwargs,
                            ),
                        )
                    )
                    meshed_cartilage_keys.append("meniscus_lateral")

                if config.tissue_selections.do_meniscus_medial:
                    requested_tissue_keys.append("meniscus_medial")
                    processes.append(
                        _submit(
                            pool, progress, "meniscus_medial",
                            pipe.flexible_mesh_meniscus,
                            kwds=dict(
                                is_medial=True if not swap_meniscus_ml else False,
                                origin=mass_centers["meniscus_medial"],
                                reconstruction_path=tissues["meniscus_medial"],
                                bone_reconstruction_path=meshes.get("tibia_bone", None),
                                queue=q,
                                **men_med_kwargs,
                            ),
                        )
                    )
                    meshed_cartilage_keys.append("meniscus_medial")

                results = wait_for_results(
                    list(zip(meshed_cartilage_keys, processes)), progress, pool
                )
                for key, result in zip(meshed_cartilage_keys, results):
                    if result is not None:
                        meshes[key] = result
                _save_meshes(
                    meshes, tissues, config, INP_EXPORT_FILE_STEMS,
                    temp_stl_path.parent / dirs_and_logs.export_dirname, temp_dir,
                    left_right_flip_normal, left_right_flip_center, progress,
                )
                logging.info(
                    "Elapsed meshing time for this knee: %s.",
                    datetime.datetime.now() - i_start,
                )

    # Final logging for the total batch
    if len(successful) > 0:
        tqdm.write("Succesful:\n    ", end="")
        tqdm.write("\n    ".join(successful))
        logging.info("Successful:\n    ")
        logging.info("\n    ".join(["Successful:"] + successful))

    if len(unsuccessful) > 0:
        for idx, (knee, tissues) in enumerate(zip(unsuccessful, unsuccessful_tissues)):
            tqdm.write(f"{knee} failed to mesh:\n    ", end="")
            tqdm.write("\n    ".join(tissues))
            logging.info(f"{knee} failed to mesh:\n    ")
            logging.info("\n    ".join(tissues))
            if idx < len(unsuccessful) - 1:
                tqdm.write("\n")

        tqdm.write("At least one tissue failed:\n    ", end="")
        tqdm.write("\n    ".join(unsuccessful))
        logging.info("\n    ".join(["Not successful:"] + unsuccessful))

    if len(successful) == 0:
        tqdm.write("None succeeded.")

    if len(unsuccessful) == 0:
        tqdm.write("No failures.")

    end = datetime.datetime.now()
    tdelta = end - start_time
    tdelta = tdelta - datetime.timedelta(microseconds=tdelta.microseconds)
    logging.info(f"Total runtime: {str(tdelta)}")
    if successful:
        logging.info(f"Average time per knee: {str(tdelta / len(successful))}")


if __name__ == "__main__":
    run_main()
