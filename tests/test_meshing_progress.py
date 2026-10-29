"""Verify structured worker progress and parent-only terminal rendering."""

import logging
import multiprocessing
import os
from multiprocessing.managers import SyncManager
from queue import Queue
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

import kneemesher.meshing.pipeline as pipeline
import kneemesher.meshing.progress as progress_module
from kneemesher.errors import ElementQualityError, OptimizationError
from kneemesher.meshing.progress import (
    BatchProgress,
    ProgressEvent,
    WorkProgress,
    begin_attempt,
    report,
    run_job,
    wait_for_results,
)


class RecordingBar:
    """Record rendering calls without writing terminal control sequences."""

    def __init__(self, **kwargs: Any) -> None:
        """Store bar positions, counters, and the rendering process."""
        self.options = kwargs
        self.total = kwargs.get("total")
        self.n = 0
        self.closed = False
        self.resets = 0
        self.pid = os.getpid()

    def reset(self, total: int | None = None) -> None:
        """Mirror tqdm's total retention for indeterminate stage regression tests."""
        self.n = 0
        if total is not None:
            self.total = total
        self.resets += 1

    def update(self, n: int = 1) -> None:
        """Advance the batch count."""
        self.n += n

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        """Accept a stage label."""

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        """Accept running outcome counts."""

    def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
        """Accept work units."""

    def refresh(self) -> None:
        """Record a parent-owned refresh."""
        assert self.pid == os.getpid()

    def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True


@pytest.fixture
def display(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Create a deterministic parent display and close it after each test."""
    monkeypatch.setattr(progress_module, "tqdm", RecordingBar)
    instance = BatchProgress(3, Queue(), auto_refresh=False)
    instance.start_knee("knee", ["bone", "cartilage"])
    yield instance
    instance.close()


def _events(queue: Any) -> list[ProgressEvent]:
    """Collect synchronous queue events after a task returns."""
    result = []
    while not queue.empty():
        result.append(queue.get_nowait())
    return result


def test_fixed_rows_retry_reset_and_late_events(display: BatchProgress) -> None:
    """Keep row positions stable and reject old knee and terminal updates."""
    row = display.rows["bone"]
    display.apply(ProgressEvent("knee", "bone", "Raytracing", 3, 10))
    assert row.n == 3 and row.total == 10
    display.apply(ProgressEvent("knee", "bone", "Optimizing", attempt=2))
    assert row.n == 0 and row.total is None
    display.apply(ProgressEvent("knee", "bone", "Old attempt", 9, 10))
    assert display.events["bone"].stage == "Optimizing"
    display.tissue("bone", "Exported", status="exported")
    display.apply(ProgressEvent("knee", "bone", "Late update", attempt=2))
    display.apply(ProgressEvent("previous knee", "cartilage", "Late update"))
    assert display.events["bone"].status == "exported"
    assert display.events["cartilage"].status == "pending"
    assert display.rows["bone"] is row
    assert [bar.options["position"] for bar in display.rows.values()] == [1, 2]


def test_batch_outcomes_and_row_reuse(display: BatchProgress) -> None:
    """Advance exported, skipped, and failed knees exactly once per completion."""
    for tissue in display.rows:
        display.tissue(tissue, "Exported", status="exported")
    old_rows = list(display.rows.values())
    display.finish_knee("exported")
    display.start_knee("excluded", [])
    assert all(row.closed for row in old_rows)
    display.finish_knee("skipped")
    display.start_knee("missing", ["bone"])
    display.finish_knee("failed")
    assert display.counts == dict(exported=1, skipped=1, failed=1)
    assert display.batch.n == 3
    assert display.events["bone"].status == "failed"


def test_actual_work_counts_and_no_worker_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report completed items and unknown convergence totals without any bars."""
    monkeypatch.setattr(
        progress_module,
        "tqdm",
        lambda **kwargs: pytest.fail("Workers must not create terminal bars"),
    )
    queue = Queue()

    def operation() -> bool:
        """Exercise finite NumPy iterables and an indeterminate loop."""
        for _ in WorkProgress(np.arange(4), desc="Slices", unit="slices"):
            pass
        with WorkProgress(desc="Convergence") as work:
            work.update(3)
        begin_attempt(2)
        report("Native operation")
        return True

    assert run_job(queue, "knee", "cartilage", operation, (), {})
    events = _events(queue)
    assert any(e.stage == "Slices" and e.completed == e.total == 4 for e in events)
    assert all(e.total is None for e in events if e.stage == "Convergence")
    assert events[-1].attempt == 2 and events[-1].status == "ready"
    report("Outside worker")
    assert queue.empty()


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_worker_failure_restores_context(error: type[BaseException]) -> None:
    """Send a failed event and restore reporting after errors and interrupts."""
    queue = Queue()

    def fail() -> None:
        """Raise the requested failure."""
        raise error()

    with pytest.raises(error):
        run_job(queue, "knee", "bone", fail, (), {})
    assert _events(queue)[-1].status == "failed"
    report("Outside worker")
    assert queue.empty()


def test_poll_all_results_while_draining(
    display: BatchProgress,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consume later completed jobs before the first and preserve result order."""
    order = []
    first = SimpleNamespace(ready=lambda: False, get=lambda: order.append("first") or 1)
    second = SimpleNamespace(
        ready=lambda: True, get=lambda: order.append("second") or 2
    )

    def advance(_: float) -> None:
        """Make the first result ready after observing the second."""
        assert order == ["second"]
        display.queue.put(ProgressEvent("knee", "bone", "Slices", 2, 4))
        first.ready = lambda: True

    monkeypatch.setattr(progress_module, "sleep", advance)
    pool = SimpleNamespace(_pool=[SimpleNamespace(pid=1, exitcode=None)])
    result = wait_for_results([("bone", first), ("cartilage", second)], display, pool)
    assert result == [1, 2] and order == ["second", "first"]
    assert display.events["bone"].completed == 2


def test_result_error_does_not_discard_other_results(display: BatchProgress) -> None:
    """Mark a failed tissue while collecting other successful meshes."""

    def fail() -> None:
        """Raise an asynchronous task error."""
        raise RuntimeError("meshing failed")

    pool = SimpleNamespace(_pool=[])
    jobs = [
        ("bone", SimpleNamespace(ready=lambda: True, get=fail)),
        ("cartilage", SimpleNamespace(ready=lambda: True, get=lambda: "mesh")),
    ]
    assert wait_for_results(jobs, display, pool) == [None, "mesh"]
    assert display.events["bone"].status == "failed"


@pytest.mark.parametrize("hard_exit", [False, True])
def test_spawn_queue_and_worker_death(
    display: BatchProgress,
    hard_exit: bool,
) -> None:
    """Use a real spawned worker and detect lost jobs without waiting forever."""
    context = multiprocessing.get_context("spawn")
    with SyncManager(
        address=f".test-manager-{uuid4().hex}.sock", ctx=context
    ) as manager:
        display.queue = manager.Queue()
        with context.Pool(1) as pool:
            pool._mesh_worker_pids = {worker.pid for worker in pool._pool}
            function, args = (os._exit, (7,)) if hard_exit else (sum, ([1, 2],))
            job = pool.apply_async(
                run_job, (display.queue, "knee", "bone", function, args, {})
            )
            if hard_exit:
                with pytest.raises(RuntimeError, match="worker exited"):
                    wait_for_results([("bone", job)], display, pool)
            else:
                assert wait_for_results([("bone", job)], display, pool) == [3]
                assert display.events["bone"].status == "ready"
                assert display.rows["bone"].pid == os.getpid()


@pytest.mark.parametrize(
    ("wrapper", "algorithm", "kwargs"),
    [
        ("flexible_mesh_femur_cartilage", "mesh_femur_cartilage", {"points": (0, 1)}),
        (
            "flexible_mesh_patella_cartilage",
            "mesh_patella_cartilage",
            {"points": (0, 1)},
        ),
        ("flexible_mesh_tibia_cartilage", "mesh_tibia_cartilage", {}),
    ],
)
def test_retry_events_and_logging_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: str,
    algorithm: str,
    kwargs: dict[str, Any],
) -> None:
    """Retain retry settings, reset attempt work, and release task log handlers."""
    calls = []

    def mesh(**options: Any) -> str:
        """Fail once before returning a mesh on the existing retry path."""
        calls.append(options)
        if len(calls) == 1:
            raise OptimizationError("retry")
        report("Successful stage", total=2)
        return "mesh"

    monkeypatch.setattr(pipeline, algorithm, mesh)
    queue, logs = Queue(), Queue()
    root = logging.getLogger()
    handlers = root.handlers[:]
    logger = multiprocessing.get_logger()
    worker_handlers = logger.handlers[:]
    try:
        result = run_job(
            queue,
            "knee",
            "cartilage",
            getattr(pipeline, wrapper),
            (),
            dict(queue=logs, **kwargs),
        )
        assert result == "mesh" and len(calls) == 2
        assert root.handlers == handlers and logger.handlers == worker_handlers
        events = _events(queue)
        assert {event.attempt for event in events} == {1, 2}
        assert any(e.stage == "Retrying meshing" and e.completed == 0 for e in events)
    finally:
        root.setLevel(logging.WARNING)


def test_exhausted_retry_settings_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop when an element-quality retry no longer changes any meshing setting."""
    calls = []

    def fail(**kwargs: Any) -> None:
        """Reject every attempted mesh."""
        calls.append(kwargs)
        raise ElementQualityError("bad quality")

    monkeypatch.setattr(pipeline, "mesh_femur_cartilage", fail)
    assert pipeline.flexible_mesh_femur_cartilage(points=(0, 1)) is None
    assert len(calls) == 3
