"""Verify configured reconstruction batches and bone clipping."""

import importlib.util
import logging
import warnings
from multiprocessing.pool import ThreadPool
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest
import pyvista as pv

from kneemesher.reconstruction._reconstruction import transform_mesh


@pytest.fixture
def pipeline() -> ModuleType:
    """Load the notebook script without running its main entry point."""
    path = (
        Path(__file__).parents[1]
        / "notebooks/meshing_pipeline/mesh_00_kg_reconstruct_knee_geometries.py"
    )
    spec = importlib.util.spec_from_file_location("reconstruction_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reconstruction_logs_stay_out_of_terminal(
    pipeline: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write warning bursts to disk without scrolling the progress display."""
    filename = tmp_path / "logging/reconstruction.log"
    original_handlers = logging.getLogger().handlers[:]
    with pipeline._file_logging(filename):
        with pipeline.tqdm(total=1, disable=False, leave=True) as bar:
            for index in range(100):
                pipeline.logger.warning("parent message %s", index)
            with ThreadPool(1) as pool:
                pool.apply(pipeline.logger.warning, ("worker message",))
            warnings.warn("library warning", UserWarning, stacklevel=1)
            bar.update()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "100%" in captured.err
    for message in ("parent message", "worker message", "library warning"):
        assert message not in captured.err
    text = filename.read_text()
    assert text.count("parent message") == 100
    assert text.count("worker message") == 1
    assert "UserWarning: library warning" in text
    assert logging.getLogger().handlers == original_handlers


@pytest.mark.parametrize("include", [[], ["9002_M00", "9004_M00", "9006_M00"]])
def test_batch_limit_follows_subject_filter(
    pipeline: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include: list[str],
) -> None:
    """Process only the first matching files rather than truncating before filtering."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for subject in reversed(range(7)):
        with h5py.File(inputs / f"900{subject}_M00.h5", "w") as file:
            file.create_dataset("masks/femur_bone", data=np.ones((3, 3, 3)))
    processed = []
    log_path = tmp_path / "reconstruction.log"
    original_logging = pipeline._file_logging
    monkeypatch.setattr(
        pipeline, "_file_logging", lambda filename: original_logging(log_path)
    )

    def reconstruct(mask: np.ndarray, name: str, **kwargs) -> pv.PolyData:
        """Check configured arguments without running the expensive tissue pipeline."""
        assert mask.shape == (3, 3, 3)
        assert name == "femur_bone"
        assert kwargs["bone"] is True
        assert kwargs["voxel_spacing"] == [1, 2, 3]
        assert kwargs["smooth_kwargs"] == {"n_bone_iters": 4, "n_iters": 2}
        return pv.Cube().triangulate()

    monkeypatch.setattr(pipeline, "reconstruct", reconstruct)
    monkeypatch.setattr(pipeline.multiprocessing, "Pool", ThreadPool)
    runner = pipeline.KneeReconstructor(
        {
            "segmentation_dir": str(inputs),
            "export_dir": str(tmp_path / "outputs"),
            "max_knees": 5,
            "num_processors": 1,
            "subjects": {"include": include},
            "voxel_spacing": [1, 2, 3],
            "smoothing": {"n_bone_iters": 4, "n_iters": 2},
        }
    )
    monkeypatch.setattr(
        runner,
        "save_results",
        lambda results, tissues, directory, names: processed.extend(names),
    )
    runner.run()
    expected = include if include else [f"900{i}_M00" for i in range(5)]
    assert processed == [f"{subject}_femur_bone" for subject in expected]
    log_text = log_path.read_text()
    assert f"Starting reconstruction batch: {len(expected)} knees" in log_text
    for index, subject in enumerate(expected, start=1):
        assert f"Starting knee {index}/{len(expected)}: {subject}" in log_text
        assert f"Completed knee {index}/{len(expected)}: {subject}" in log_text
    assert f"Finished reconstruction batch: {len(expected)} knees visited" in log_text


def test_subject_ids_keep_all_timepoints_and_sides(
    pipeline: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select five subjects without truncating their sides or timepoints.

    """
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    subjects = [f"900{index}" for index in range(5)]
    expected = []
    for subject in subjects + ["9999"]:
        for month in ("M00", "M12"):
            for side in ("left", "right"):
                name = f"{subject}_{month}_{side}"
                with h5py.File(inputs / f"{name}.h5", "w") as file:
                    file.create_dataset("masks/femur_bone", data=np.ones((3, 3, 3)))
                if subject in subjects:
                    expected.append(name)
    processed = []
    monkeypatch.setattr(pipeline.multiprocessing, "Pool", ThreadPool)
    monkeypatch.setattr(
        pipeline, "reconstruct", lambda *args, **kwargs: pv.Cube().triangulate()
    )
    runner = pipeline.KneeReconstructor(
        {
            "segmentation_dir": str(inputs),
            "export_dir": str(tmp_path / "outputs"),
            "num_processors": 1,
            "subjects": {"include": subjects},
        }
    )
    monkeypatch.setattr(
        runner,
        "save_results",
        lambda results, tissues, directory, names: processed.append(
            "_".join(directory.parts[-3:])
        ),
    )
    runner.run()
    assert processed == sorted(expected)


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "5"])
def test_invalid_batch_limit(
    pipeline: ModuleType,
    tmp_path: Path,
    limit: object,
) -> None:
    """Reject invalid limits rather than processing an unexpected set of knees."""
    with pytest.raises(ValueError, match="max_knees"):
        pipeline.KneeReconstructor(
            {
                "segmentation_dir": str(tmp_path),
                "max_knees": limit,
            }
        )


def test_empty_input_directory(pipeline: ModuleType, tmp_path: Path) -> None:
    """Report absent input files before indexing or starting worker processes."""
    runner = pipeline.KneeReconstructor({"segmentation_dir": str(tmp_path)})
    with pytest.raises(FileNotFoundError, match="No matching HDF5"):
        runner.run()


def test_default_export_directory_uses_reconstruction_root(
    pipeline: ModuleType, tmp_path: Path
) -> None:
    """Place default outputs under data/reconstructions by input group name."""
    runner = pipeline.KneeReconstructor({"segmentation_dir": str(tmp_path)})
    expected = Path(pipeline.module_str).parents[2] / "data" / "reconstructions"
    assert runner.export_dir == expected / tmp_path.name


def test_missing_input_directory(pipeline: ModuleType, tmp_path: Path) -> None:
    """Fail initialization explicitly when the configured input directory is absent."""
    with pytest.raises(FileNotFoundError):
        pipeline.KneeReconstructor({"segmentation_dir": str(tmp_path / "absent")})


def test_configuration_is_relative_to_script(
    pipeline: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the associated configuration independently of the working directory."""
    expected = pipeline.load_config()
    monkeypatch.chdir(tmp_path)
    assert pipeline.load_config() == expected


@pytest.mark.parametrize("tissue", ["femur_bone", "tibia_bone"])
def test_bone_clipping_uses_scalar_origin(tissue: str) -> None:
    """Clip the intended two percent using a valid scalar plane coordinate."""
    mesh = pv.Sphere(theta_resolution=16, phi_resolution=16)
    mesh.clear_data()
    original = mesh.points.copy()
    positioned = mesh.rotate_z(-90, inplace=False).rotate_y(90, inplace=False)
    positioned.translate((10, 100, 90), inplace=True)
    low, high = positioned.bounds[-2:]
    distance = (high - low) * 0.02
    expected = (
        [low, high - distance] if tissue == "femur_bone" else [low + distance, high]
    )
    result = transform_mesh(mesh, tissue, True, repair_backend="trimesh_open3d")
    assert result.n_cells > 0
    assert result.n_open_edges == 0
    np.testing.assert_allclose(result.bounds[-2:], expected, atol=1e-5)
    np.testing.assert_array_equal(mesh.points, original)
