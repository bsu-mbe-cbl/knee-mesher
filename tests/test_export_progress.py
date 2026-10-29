"""Verify export outcomes, final destinations, and resource cleanup."""

import importlib.util
import logging
import multiprocessing
import shutil
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import pyvista as pv

from kneemesher.meshing.progress import BatchProgress, ProgressEvent, run_job


@pytest.fixture(scope="module")
def exporter() -> ModuleType:
    """Load the export entry point without executing it."""
    path = (
        Path(__file__).parents[1]
        / "notebooks/meshing_pipeline/mesh_01_kg_export_knee_meshes.py"
    )
    spec = importlib.util.spec_from_file_location("progress_exporter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workspace() -> Any:
    """Keep test files inside the repository rather than a system scratch path."""
    path = Path.cwd() / f".test-export-{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


class InlinePool:
    """Exercise asynchronous result handling without running native meshing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Track cleanup and keep a stable fake worker identity."""
        self._pool = []
        self.terminated = False
        self.joined = False

    def apply_async(
        self,
        function: Any,
        args: tuple[Any, ...] = (),
        kwds: dict[str, Any] | None = None,
    ) -> Any:
        """Return the same ready/get contract as multiprocessing."""
        result, error = None, None
        try:
            result = function(*args, **(kwds or {}))
        except Exception as exc:
            error = exc

        def get() -> Any:
            """Raise stored failures only when the parent collects the result."""
            if error is not None:
                raise error
            return result

        return SimpleNamespace(ready=lambda: True, get=get)

    def terminate(self) -> None:
        """Record process termination."""
        self.terminated = True

    def join(self) -> None:
        """Record that all workers stop before resource release."""
        self.joined = True


def _config(exporter: ModuleType, workspace: Path) -> Any:
    """Retain real config parsing while selecting only bone outputs."""
    config = exporter.parse_config()
    selections = config.tissue_selections._replace(
        **{name: name == "do_bones" for name in config.tissue_selections._fields}
    )
    return config._replace(
        subjects=config.subjects._replace(include=[], exclude=[]),
        tissue_selections=selections,
        directories_and_logging=config.directories_and_logging._replace(
            import_dir=workspace / "inputs",
            root_export_dir=workspace / "exports",
        ),
        rotations=config.rotations._replace(
            x_rotation_degree=0, y_rotation_degree=0, z_rotation_degree=0
        ),
    )


@pytest.mark.parametrize(
    ("index_slice", "include", "counts"),
    [
        ({}, ["100", "300"], dict(exported=1, skipped=1, failed=1)),
        ({"step": 2}, ["100", "300"], dict(exported=1, skipped=0, failed=1)),
        (
            {"start": 2, "step": -1},
            ["100", "300"],
            dict(exported=1, skipped=1, failed=1),
        ),
        (
            {"start": 1, "stop": 3, "step": 1},
            ["100", "300"],
            dict(exported=0, skipped=1, failed=1),
        ),
        ({"stop": 0}, [], dict(exported=0, skipped=0, failed=0)),
        ({}, ["absent"], dict(exported=0, skipped=3, failed=0)),
    ],
)
def test_batch_export_counts_only_final_outputs(
    exporter: ModuleType,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_slice: dict[str, int],
    include: list[str],
    counts: dict[str, int],
) -> None:
    """Run the batch with an export, filtered knee, and failed bone task."""
    config = _config(exporter, workspace)
    slice_parameters = config.knee_index_slice._asdict() | index_slice
    config = config._replace(
        subjects=config.subjects._replace(include=include),
        knee_index_slice=SimpleNamespace(_asdict=lambda: slice_parameters),
    )
    stems = config.stl_import_stems._asdict()
    for subject in ("100", "200", "300"):
        directory = config.directories_and_logging.import_dir / subject / "M00"
        directory.mkdir(parents=True)
        for key, stem in stems.items():
            if "bone" in key or key == "femur_cartilage":
                (directory / f"{subject}_{stem}.stl").write_text("surface")
    stale = (
        config.directories_and_logging.root_export_dir
        / "300/M00"
        / config.directories_and_logging.export_dirname
        / f"{config.inp_export_stems.femur_bone}.inp"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text("mesh from a previous run")

    pools = []
    displays = []

    def make_pool(*args: Any, **kwargs: Any) -> InlinePool:
        """Record every selected knee's worker cleanup."""
        pool = InlinePool()
        pools.append(pool)
        return pool

    def make_display(total: int, queue: Any) -> BatchProgress:
        """Keep the completed batch counters accessible for assertions."""
        display = BatchProgress(total, queue, auto_refresh=False)
        displays.append(display)
        return display

    def save_bone(source: Path, destination: Path, **kwargs: Any) -> str:
        """Write real destination candidates and fail one requested tissue."""
        if "300" in source.name and "femur" in source.name:
            raise RuntimeError("bone failed")
        destination.write_text("exported mesh")
        return "bone mesh"

    context = SimpleNamespace(Pool=make_pool, RLock=threading.RLock)
    monkeypatch.setattr(exporter, "parse_config", lambda: config)
    monkeypatch.setattr(exporter.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(exporter, "BatchProgress", make_display)
    monkeypatch.setattr(
        exporter,
        "SyncManager",
        lambda **kwargs: nullcontext(SimpleNamespace(Queue=Queue)),
    )
    monkeypatch.setattr(exporter, "save_rigid_body", save_bone)
    monkeypatch.setattr(
        exporter.pv, "read", lambda _: SimpleNamespace(is_manifold=True)
    )
    monkeypatch.setattr(exporter, "prepare", lambda **kwargs: (None, None, {}))
    exporter.run_main()
    display = displays[0]
    assert display.counts == counts
    assert display.batch.n == display.batch.total == sum(counts.values())
    assert len(pools) == counts["exported"] + counts["failed"]
    assert all(p.terminated and p.joined for p in pools)
    destination = config.directories_and_logging.root_export_dir / "100/M00"
    assert len(list(destination.rglob("*.inp"))) == 4 * counts["exported"]
    assert not list(Path.cwd().glob(".mesh-export-*"))


@pytest.mark.parametrize("interrupt", [False, True])
def test_knee_failure_and_interruption_cleanup(
    exporter: ModuleType,
    workspace: Path,
    interrupt: bool,
) -> None:
    """Stop workers on both failed preparation and keyboard interruption."""
    config = _config(exporter, workspace)
    path = config.directories_and_logging.import_dir / "100/M00/femur.stl"
    pool = InlinePool()
    context = SimpleNamespace(Pool=lambda *args, **kwargs: pool)
    display = BatchProgress(1, Queue(), auto_refresh=False)
    successful, failed, tissues = [], [], []
    error = KeyboardInterrupt if interrupt else RuntimeError
    try:
        with pytest.raises(KeyboardInterrupt) if interrupt else nullcontext():
            with exporter._knee_scope(
                context,
                display,
                Queue(),
                config,
                path,
                {"femur_bone": "femur_bone"},
                successful,
                failed,
                tissues,
            ):
                raise error("stop")
        assert pool.terminated and pool.joined
        assert not successful
        if interrupt:
            assert display.batch.n == 0
            assert display.events["femur_bone"].status == "cancelled"
        else:
            assert display.counts["failed"] == 1
            assert failed == ["100_M00_R"]
    finally:
        display.close()


def test_batch_resources_cleanup_on_interrupt(
    exporter: ModuleType,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop warning and refresh threads and remove scratch files on interruption."""
    monkeypatch.chdir(workspace)
    context = SimpleNamespace()
    monkeypatch.setattr(
        exporter,
        "SyncManager",
        lambda **kwargs: nullcontext(SimpleNamespace(Queue=Queue)),
    )
    with pytest.raises(KeyboardInterrupt):
        with exporter._batch_resources(context, 1) as (display, logs, events, path):
            assert path.is_dir()
            assert logs is not events
            assert display.thread.is_alive()
            raise KeyboardInterrupt
    assert not display.thread.is_alive()
    assert not path.exists()


def test_missing_requested_input_keeps_fixed_row(
    exporter: ModuleType,
    workspace: Path,
) -> None:
    """Fail a selected knee even if a configured bone input is absent."""
    config = _config(exporter, workspace)
    pool = InlinePool()
    display = BatchProgress(1, Queue(), auto_refresh=False)
    try:
        with exporter._knee_scope(
            SimpleNamespace(Pool=lambda *args, **kwargs: pool),
            display,
            Queue(),
            config,
            config.directories_and_logging.import_dir / "100/M00/femur.stl",
            {"femur_bone": "femur_bone", "tibia_bone": "tibia_bone"},
            [],
            [],
            [],
        ):
            display.tissue("femur_bone", "Exported", status="exported")
        assert list(display.rows) == ["femur_bone", "tibia_bone"]
        assert display.counts == dict(exported=0, skipped=0, failed=1)
    finally:
        display.close()


def test_real_manager_and_refresh_cleanup(
    exporter: ModuleType,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Close the real manager and refresh thread after an interrupted batch."""
    monkeypatch.chdir(workspace)
    context = multiprocessing.get_context("spawn")
    with pytest.raises(KeyboardInterrupt):
        with exporter._batch_resources(context, 1) as (display, logs, events, path):
            logs.put(
                logging.makeLogRecord(
                    {
                        "name": "export-progress-test",
                        "levelno": logging.WARNING,
                        "levelname": "WARNING",
                        "msg": "forwarded warning",
                    }
                )
            )
            display.start_knee("knee", ["bone"])
            events.put(ProgressEvent("knee", "bone", "Stage", 1, 2))
            display.drain()
            assert display.events["bone"].completed == 1
            assert path.is_dir()
            raise KeyboardInterrupt
    assert not display.thread.is_alive()
    assert not path.exists()
    assert sum(record.message == "forwarded warning" for record in caplog.records) == 1


@pytest.mark.parametrize("move_fails", [False, True])
def test_final_move_controls_exported_status(
    exporter: ModuleType,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    move_fails: bool,
) -> None:
    """Do not count a mesh as exported before its final move succeeds."""
    config = _config(exporter, workspace)
    work = workspace / "work"
    save_dir = work / "100/M00/meshes"
    save_dir.mkdir(parents=True)
    (save_dir / "FEM.inp").write_text("mesh")
    display = BatchProgress(1, Queue(), auto_refresh=False)
    pool = InlinePool()
    move = exporter.shutil.move

    def checked_move(source: Any, destination: Any) -> Any:
        """Check status at the exact final-placement boundary."""
        assert display.events["femur_bone"].status != "exported"
        if move_fails:
            raise OSError("destination unavailable")
        return move(source, destination)

    monkeypatch.setattr(exporter.shutil, "move", checked_move)
    try:
        with exporter._knee_scope(
            SimpleNamespace(Pool=lambda *args, **kwargs: pool),
            display,
            Queue(),
            config,
            config.directories_and_logging.import_dir / "100/M00/femur.stl",
            {"femur_bone": "femur_bone"},
            [],
            [],
            [],
        ):
            exporter._save_meshes(
                {"femur_bone": "mesh"},
                {},
                config,
                {"femur_bone": "FEM"},
                save_dir,
                work,
                None,
                None,
                display,
            )
        assert display.counts["failed" if move_fails else "exported"] == 1
        destination = config.directories_and_logging.root_export_dir / "100/M00/meshes"
        assert (destination / "FEM.inp").is_file() is not move_fails
    finally:
        display.close()


def test_native_rigid_meshing_events(
    exporter: ModuleType,
    workspace: Path,
) -> None:
    """Run native rigid meshing and INP writing with structured stage events."""
    source = workspace / "femur_bone.stl"
    destination = workspace / "FEM.inp"
    pv.Sphere(theta_resolution=12, phi_resolution=12).save(source)
    queue = Queue()
    result = run_job(
        queue,
        "knee",
        "femur_bone",
        exporter.save_rigid_body,
        (source, destination),
        {"target_size": 0.1},
    )
    stages = []
    while not queue.empty():
        stages.append(queue.get_nowait().stage)
    assert result.n_cells > 0
    assert destination.is_file() and destination.stat().st_size > 0
    assert stages == [
        "Starting",
        "Preprocessing rigid surface",
        "Clustering rigid surface",
        "Rotating rigid mesh",
        "Writing rigid INP",
        "Mesh ready",
    ]


def test_spawned_native_export_and_warning_listener(
    exporter: ModuleType,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Run native export through the real worker initializer and both queues."""
    script_dir = Path(__file__).parents[1] / "notebooks/meshing_pipeline"
    monkeypatch.syspath_prepend(str(script_dir))
    production = importlib.import_module("mesh_01_kg_export_knee_meshes")
    monkeypatch.setattr(production.psutil, "cpu_count", lambda **kwargs: 1)
    config = _config(exporter, workspace)
    source = config.directories_and_logging.import_dir / "100/M00/femur_bone.stl"
    source.parent.mkdir(parents=True)
    pv.Sphere(theta_resolution=12, phi_resolution=12).save(source)
    context = multiprocessing.get_context("spawn")
    production.tqdm.get_lock()
    monkeypatch.setattr(production.tqdm, "_lock", context.RLock())
    with production._batch_resources(context, 1) as (display, logs, events, work):
        save_dir = work / "100/M00/meshes"
        save_dir.mkdir(parents=True)
        with production._knee_scope(
            context,
            display,
            logs,
            config,
            source,
            {"femur_bone": "femur_bone"},
            [],
            [],
            [],
        ) as pool:
            pool.apply_async(warnings.warn, ("spawned export warning",)).get(timeout=30)
            job = production._submit(
                pool,
                display,
                "femur_bone",
                production.save_rigid_body,
                (source, save_dir / "FEM.inp"),
                {"target_size": 0.1},
            )
            result = production.wait_for_results([("femur_bone", job)], display, pool)
            assert result[0].n_cells > 0
            production._save_meshes(
                {"femur_bone": result[0]},
                {},
                config,
                {"femur_bone": "FEM"},
                save_dir,
                work,
                None,
                None,
                display,
            )
        assert display.counts == dict(exported=1, skipped=0, failed=0)
    assert (
        config.directories_and_logging.root_export_dir / "100/M00/meshes/FEM.inp"
    ).is_file()
    assert sum("spawned export warning" in r.message for r in caplog.records) == 1
