"""Reconstruct knee geometries from HDF5 segmentation data.

This script processes HDF5 files containing segmented knee images, applies
morphological operations, and generates 3D meshes of various tissues.

Usage:
    uv run python notebooks/meshing_pipeline/mesh_00_kg_reconstruct_knee_geometries.py

Configuration:
    Loads 'configs/00-kg-reconstruct-knee-geometries.toml'.
    max_knees limits the sorted inputs after applying the subject filter.

"""

# %%
import logging
import multiprocessing
import tomllib
import warnings
from collections.abc import Iterator
from contextlib import closing, contextmanager
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import pyvista as pv
from tqdm.auto import tqdm

from kneemesher import __file__ as module_str
from kneemesher.core_api import TISSUE_COLORS
from kneemesher.reconstruction import reconstruct

logger = logging.getLogger(__name__)

# %%


def _configure_worker_logging(queue: Queue) -> None:
    """Send worker logs and Python warnings to the parent logging queue."""
    root = logging.getLogger()
    root.handlers = [QueueHandler(queue)]
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)


@contextmanager
def _file_logging(filename: Path) -> Iterator[Queue]:
    """Keep terminal output free of logs while preserving parent and worker records."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    with closing(logging.FileHandler(filename, mode="a")) as handler:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        queue = multiprocessing.Queue()
        listener = QueueListener(queue, handler)
        listener.start()
        with warnings.catch_warnings():
            try:
                _configure_worker_logging(queue)
                yield queue
            finally:
                logging.captureWarnings(False)
                root.handlers = handlers
                root.setLevel(level)
                listener.stop()
                queue.close()
                queue.join_thread()


def read_imorphics(
    subject_filename: str,
    axes_shifts: Optional[Tuple[int, ...]] = None,
    flip_left_right: bool = False,
    masks_map: Optional[dict] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """Read the imorphics data from an HDF5 file.

    Args:
        subject_filename (str): Path to the HDF5 file.
        axes_shifts (Optional[Tuple[int, ...]]): Axes shifts for the data.
        flip_left_right (bool): Whether to flip the data left to right.
        masks_map (Optional[dict]): Mapping of internal mask names to final mask names.

    Returns:
        Tuple[Optional[np.ndarray], dict]: A tuple containing images and a dictionary of masks.

    """
    with h5py.File(subject_filename, "r") as f:
        try:
            images = f["images"][:]
        except (KeyError, OSError):
            images = None
        mask_group = f["masks"]
        tissues = list(mask_group.keys())
        masks = np.asarray([mask_group[t][:] for t in tissues])
        if flip_left_right:
            masks = masks[..., ::-1]
            if images is not None:
                images = images[..., ::-1]
        if axes_shifts is not None:
            axes_shifts = np.array(axes_shifts)
            if masks.ndim == 4:
                axes_shifts += 1
                masks = np.moveaxis(masks, *axes_shifts)
        mask_dict = {t: m for t, m in zip(tissues, masks)}
        if masks_map:
            try:
                masks_map = {v: k for k, v in masks_map.items()}
                mask_dict = {masks_map.get(t, t): m for t, m in zip(tissues, masks)}
            except KeyError:
                pass
    return images, mask_dict


class KneeReconstructor:
    """A class to reconstruct knee geometries from HDF5 files.

    Attributes:
        config (dict): Configuration dictionary.
        axes_shifts (list): Axes shifts for the reconstruction.
        rotations_xyz (list): Rotations to apply on the x, y, and z axes.
        flip_x (bool): Whether to flip the reconstruction along the x-axis.
        flip_y (bool): Whether to flip the reconstruction along the y-axis.
        flip_z (bool): Whether to flip the reconstruction along the z-axis.
        flip_left_right_orig (bool): Whether to flip the left-right orientation.
        segmentation_dir_path (str): Path to the segmentation directory.
        export_dir_path (str): Path to the export directory.
        overwrite_existing (bool): Whether to overwrite existing files.
        segmentation_dir (Path): Path object for the segmentation directory.
        export_dir (Path): Path object for the export directory.
        num_processors (int): Number of processors to use for multiprocessing.
        voxel_spacing (tuple): Voxel spacing for the reconstruction.

    """

    def __init__(self, config: dict) -> None:
        """Initialize the KneeReconstructor with configuration.

        Args:
            config (dict): Configuration dictionary.

        """
        self.config = config
        self.axes_shifts = config.get("axes_shifts")
        self.rotations_xyz = config.get("rotations_xyz")
        self.flip_x = config.get("flip_x", False)
        self.flip_y = config.get("flip_y", False)
        self.flip_z = config.get("flip_z", False)
        self.flip_left_right_orig = config.get("flip_left_right_orig", False)
        self.segmentation_dir_path = config.get("segmentation_dir")
        self.export_dir_path = config.get("export_dir")
        self.overwrite_existing = config.get("overwrite_existing", False)
        self.max_knees = config.get("max_knees")
        if self.max_knees is not None and (
            type(self.max_knees) is not int or self.max_knees < 1
        ):
            raise ValueError("max_knees must be a positive integer.")

        self.num_processors = config.get(
            "num_processors", max(1, multiprocessing.cpu_count() - 1)
        )

        self.voxel_spacing = config.get("voxel_spacing", (0.36458333, 0.36458333, 0.7))

        self.masks_map = config.get("masks_map", {})

        self.smoothing_params = config.get(
            "smoothing", {}
        )

        module_dir = Path(module_str).parents[2]
        if not Path(self.segmentation_dir_path).is_absolute():
            self.segmentation_dir_path = module_dir / self.segmentation_dir_path

        self.segmentation_dir = Path(self.segmentation_dir_path)
        if not self.segmentation_dir.exists():
            logger.error(f"Directory {self.segmentation_dir} does not exist.")
            raise FileNotFoundError(self.segmentation_dir)

        if self.export_dir_path:
            self.export_dir = Path(self.export_dir_path)
            if not self.export_dir.is_absolute():
                self.export_dir = module_dir / self.export_dir
        else:
            self.export_dir = (
                module_dir
                / "data"
                / "reconstructions"
                / self.segmentation_dir.name
            )
        self.include_subjects = config.get("subjects", {}).get("include", None)
        if self.include_subjects:
            self.include_subjects = self.expand_subject_keys(self.include_subjects)
        else:
            self.include_subjects = set()

    @staticmethod
    def expand_subject_keys(patterns: list[str]) -> set[str]:
        """Keep subject IDs and expand timepoint keys into both sides.

        """
        expanded = set()
        for pattern in patterns:
            if "_" not in pattern:
                expanded.add(pattern)
            elif pattern.endswith("_LR"):
                base = pattern[:-3]
                expanded.add(f"{base}_L")
                expanded.add(f"{base}_R")
            elif pattern.endswith("_L") or pattern.endswith("_R"):
                expanded.add(pattern)
            else:
                expanded.add(f"{pattern}_L")
                expanded.add(f"{pattern}_R")
        return expanded

    def process_file(self, h5_fname: Path) -> Tuple[dict, list, Path, str, str]:
        """Process a single HDF5 file to extract and prepare data for reconstruction.

        Args:
            h5_fname (Path): Path to the HDF5 file.

        Returns:
            Tuple[dict, list, Path, str, str]: A tuple containing label dictionary, tissue names, save directory, subject, and month.

        """
        flip_left_right = (
            self.flip_left_right_orig
            if any(x in h5_fname.stem.lower() for x in ["left", "right"])
            else self.flip_left_right_orig
        )
        try:
            _, label_dict = read_imorphics(
                h5_fname,
                axes_shifts=self.axes_shifts,
                flip_left_right=flip_left_right,
                masks_map=self.masks_map,
            )
        except (OSError, KeyError) as e:
            logger.error(f"Error reading {h5_fname}: {e}")
            return {}, [], Path(), "", ""
        tissue_names = list(label_dict.keys())
        fname_parts = h5_fname.stem.split("_")
        subject, month = fname_parts[:2]
        side = "L" if "left" in h5_fname.stem.lower() else "R"
        save_dir = self.export_dir.joinpath(
            subject, month, "_".join(fname_parts[2:]) if len(fname_parts) > 2 else ""
        )
        return label_dict, tissue_names, save_dir, subject, month, side

    def reconstruct_tissues(
        self,
        pool: multiprocessing.Pool,
        label_dict: dict,
        tissue_names: list,
        subject: str,
        month: str,
        save_dir: Path,
    ) -> Tuple[list, list]:
        """Reconstruct tissues using multiprocessing.

        Args:
            pool (multiprocessing.Pool): Multiprocessing pool.
            label_dict (dict): Dictionary of labels.
            tissue_names (list): List of tissue names.
            subject (str): Subject identifier.
            month (str): Month identifier.
            save_dir (Path): Directory to save the results.

        Returns:
            Tuple[list, list]: A tuple containing processes and save names.

        """
        procs = []
        savenames = []
        for tname in tissue_names:
            is_bone = "bone" in tname
            try:
                mask_arr = label_dict[tname]
            except KeyError as e:
                logger.error(f"{subject}_{month} KeyError {repr(e)}")
                continue
            savename = "_".join((subject, month, tname))
            save_path = save_dir.joinpath(savename).with_suffix(".stl")
            if save_path.exists() and not self.overwrite_existing:
                logger.info(
                    f"File {save_path} already exists. Skipping reconstruction."
                )
                continue
            smooth_kwargs = {
                "n_bone_iters": self.smoothing_params.get("n_bone_iters", None),
                "n_iters": self.smoothing_params.get("n_iters", None),
            }
            procs.append(
                pool.apply_async(
                    reconstruct,
                    (mask_arr, tname),
                    kwds={
                        "bone": is_bone,
                        "raw_reconstruction": False,
                        "voxel_spacing": self.voxel_spacing,
                        "smooth_kwargs": smooth_kwargs,
                    },
                )
            )
            savenames.append(savename)
        return procs, savenames

    def apply_transformations(self, reconstructions: list) -> list:
        """Apply transformations to the reconstructed results.

        Args:
            reconstructions (list): List of reconstructed results.

        Returns:
            list: List of transformed results.

        """
        for idx, reconstruction in enumerate(reconstructions):
            if reconstruction.n_cells == 0:
                continue
            if self.rotations_xyz:
                reconstruction = reconstruction.rotate_x(self.rotations_xyz[0])
                reconstruction = reconstruction.rotate_y(self.rotations_xyz[1])
                reconstruction = reconstruction.rotate_z(self.rotations_xyz[2])
            if self.flip_x:
                reconstruction = reconstruction.flip_x()
            if self.flip_y:
                reconstruction = reconstruction.flip_y()
            if self.flip_z:
                reconstruction = reconstruction.flip_z()
            reconstructions[idx] = reconstruction
        return reconstructions

    def save_results(
        self, results: list, tissue_names: list, save_dir: Path, savenames: list
    ) -> None:
        """Save the reconstructed results to disk and generate screenshots.

        Args:
            results (list): List of reconstructed results.
            tissue_names (list): List of tissue names.
            save_dir (Path): Directory to save the results.
            savenames (list): List of save names.

        """
        is_empty = {t: r.n_cells == 0 for t, r in zip(tissue_names, results)}
        if any(is_empty.values()):
            logger.warning(
                f"Empty reconstructions: {', '.join([t for t, e in is_empty.items() if e])}"
            )
        results_dict = dict(zip(tissue_names, results))
        p = pv.Plotter(off_screen=True)
        p.add_axes()
        p.add_title("_".join(save_dir.parts[-3:]))
        plot_tissues = [
            t
            for t in tissue_names
            if any(x in t for x in ["meniscus", "tibia", "fibula"])
        ]
        try:
            [
                p.add_mesh(results_dict[pt], color=TISSUE_COLORS[pt])
                for pt in plot_tissues
                if results_dict[pt].n_points > 0
            ]
            p.screenshot(save_dir / "_".join(save_dir.parts[-3:]))
        except KeyError:
            pass
        finally:
            p.close()
        [
            r.save(save_dir.joinpath(sn).with_suffix(".stl"))
            for r, sn in zip(results, savenames)
            if sn is not None and r.n_points > 0
        ]

    def run(self) -> None:
        """Process HDF5 files and reconstruct knee geometries."""
        tqdm.set_lock(multiprocessing.RLock())
        # read the first file and count the tissues to set the number of processes
        h5_files = sorted(self.segmentation_dir.rglob("*.h5"))
        if self.include_subjects:
            h5_files = [
                path
                for path in h5_files
                if path.stem.split("_")[0] in self.include_subjects
                or (
                    f"{'_'.join(path.stem.split('_')[:2])}_"
                    f"{'L' if 'left' in path.stem.lower() else 'R'}"
                ) in self.include_subjects
            ]
        if self.max_knees is not None:
            h5_files = h5_files[: self.max_knees]
        if not h5_files:
            raise FileNotFoundError(
                f"No matching HDF5 files in {self.segmentation_dir}."
            )
        _, tissue_names, *_ = self.process_file(h5_files[0])
        num_cores = min(len(tissue_names), self.num_processors)
        log_path = (
            Path(__file__).resolve().parents[2] / "logging/reconstruction.log"
        )
        with _file_logging(log_path) as queue:
            logger.info("Starting reconstruction batch: %s knees", len(h5_files))
            with multiprocessing.Pool(
                num_cores,
                initializer=_configure_worker_logging,
                initargs=(queue,),
            ) as pool:
                with tqdm(h5_files, dynamic_ncols=True, leave=True) as pbar:
                    for knee_index, h5_fname in enumerate(pbar, start=1):
                        logger.info(
                            "Starting knee %s/%s: %s",
                            knee_index,
                            len(h5_files),
                            h5_fname.stem,
                        )
                        label_dict, tissue_names, save_dir, subject, month, _ = (
                            self.process_file(h5_fname)
                        )
                        save_dir.mkdir(exist_ok=True, parents=True)
                        if label_dict:
                            procs, savenames = self.reconstruct_tissues(
                                pool,
                                label_dict,
                                tissue_names,
                                subject,
                                month,
                                save_dir,
                            )
                            if procs:
                                pbar.set_description(f"Processing {h5_fname.stem}")
                                results = [p.get() for p in procs]
                                results = self.apply_transformations(results)
                                self.save_results(
                                    results, tissue_names, save_dir, savenames
                                )
                                logger.info(
                                    "Completed knee %s/%s: %s (%s tissues)",
                                    knee_index,
                                    len(h5_files),
                                    h5_fname.stem,
                                    len(results),
                                )
                            else:
                                pbar.set_description(f"Skipping {h5_fname.stem}")
                                logger.info(
                                    "Skipped knee %s/%s: %s (outputs already exist)",
                                    knee_index,
                                    len(h5_files),
                                    h5_fname.stem,
                                )
                        else:
                            logger.warning(
                                "Skipped knee %s/%s: %s (no readable tissue masks)",
                                knee_index,
                                len(h5_files),
                                h5_fname.stem,
                            )
            logger.info("Finished reconstruction batch: %s knees visited", len(h5_files))


def load_config() -> dict:
    """Load configuration from TOML file.

    Returns:
        dict: Configuration dictionary.

    """
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs/00-kg-reconstruct-knee-geometries.toml"
    )
    with open(config_path, "rb") as f:
        return tomllib.load(f)


if __name__ == "__main__":
    config = load_config()
    reconstructor = KneeReconstructor(config)
    reconstructor.run()
