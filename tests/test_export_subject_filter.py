"""Verify subject inclusion for the STL export pipeline."""

import importlib.util
import logging
import tomllib
import warnings
from pathlib import Path
from queue import Queue
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def pipeline() -> ModuleType:
    """Load the export script without starting its batch processing."""
    path = (
        Path(__file__).parents[1]
        / "notebooks/meshing_pipeline/mesh_01_kg_export_knee_meshes.py"
    )
    spec = importlib.util.spec_from_file_location("export_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_warnings_reach_export_logging(
    pipeline: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Forward each worker warning once through the existing log listener."""
    queue = Queue()
    warning_logger = logging.getLogger("py.warnings")
    for attribute in ("handlers", "level", "propagate"):
        monkeypatch.setattr(
            warning_logger, attribute, getattr(warning_logger, attribute)
        )
    monkeypatch.setattr(pipeline.tqdm, "set_lock", lambda lock: None)
    with warnings.catch_warnings():
        try:
            pipeline._initialize_worker(None, queue)
            warnings.simplefilter("always")
            warnings.warn("export library warning", UserWarning, stacklevel=1)
            record = queue.get_nowait()
            assert record.name == "py.warnings"
            assert record.levelno == logging.WARNING
            assert "UserWarning: export library warning" in record.getMessage()
            assert queue.empty()
        finally:
            logging.captureWarnings(False)
    warning_logger.handlers = []
    warning_logger.propagate = True
    queue.put(record)
    queue.put(None)
    pipeline._logger_thread(queue)
    assert sum("export library warning" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize(
    ("include", "expected"),
    [
        ([], True),
        (["9003406"], True),
        (["9003406_M00"], True),
        (["9003406_M00_R"], True),
        (["9003406_M00_LR"], True),
        (["9003406_M00_L"], False),
        (["9003406_M12"], False),
        (["9007827"], False),
    ],
)
def test_subject_inclusion(
    pipeline: ModuleType, include: list[str], expected: bool
) -> None:
    """Allow empty lists and retain supported identifier matching."""
    assert (
        pipeline._subject_is_included(
            "9003406", "9003406_M00", "9003406_M00_R", include
        )
        is expected
    )


def test_sample_subjects_match_reconstruction_config(
    pipeline: ModuleType,
) -> None:
    """Keep the same five subjects and all sides/timepoints in both stages.

    """
    root = Path(__file__).parents[1]
    with (root / "configs/01-kg-export-knee-meshes.toml").open("rb") as file:
        config = tomllib.load(file)
    with (root / "configs/00-kg-reconstruct-knee-geometries.toml").open(
        "rb"
    ) as file:
        reconstruction = tomllib.load(file)
    subjects = ["9056363", "9068453", "9087863", "9094865", "9102858"]
    assert config["subjects"]["include"] == subjects
    assert reconstruction["subjects"]["include"] == subjects
    assert reconstruction.get("max_knees") is None
    for subject in subjects:
        for month in ("M00", "M12"):
            for side in ("L", "R"):
                assert pipeline._subject_is_included(
                    subject,
                    f"{subject}_{month}",
                    f"{subject}_{month}_{side}",
                    config["subjects"]["include"],
                )
    assert not pipeline._subject_is_included(
        "9999999", "9999999_M00", "9999999_M00_R", config["subjects"]["include"]
    )
    parsed = pipeline.parse_config()
    directories = parsed.directories_and_logging
    assert directories.import_dir == root / reconstruction["export_dir"]
    assert directories.root_export_dir == root / "exports/sample"
