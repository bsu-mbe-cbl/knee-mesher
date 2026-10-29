"""Report meshing work without rendering from worker processes."""

import logging
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from queue import Empty
from threading import Event, RLock, Thread
from time import monotonic, sleep
from typing import Any, Literal, ParamSpec, TypeVar

from tqdm import tqdm

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True)
class ProgressEvent:
    """Identify one tissue's current work within a knee and retry attempt."""

    knee: str
    tissue: str
    stage: str
    completed: int = 0
    total: int | None = None
    attempt: int = 1
    status: str = "running"
    unit: str = "work"


_reporter: ContextVar[Callable[..., None] | None] = ContextVar(
    "mesh_progress", default=None
)
_attempt: ContextVar[int] = ContextVar("mesh_attempt", default=1)


def report(
    stage: str,
    completed: int = 0,
    total: int | None = None,
    *,
    status: str = "running",
    unit: str = "work",
) -> None:
    """Send actual work to the active job's event queue, if present."""
    callback = _reporter.get()
    if callback is not None:
        callback(stage, completed, total, _attempt.get(), status, unit)


def begin_attempt(number: int) -> None:
    """Reset stage work when a meshing wrapper starts another attempt."""
    _attempt.set(number)
    report("Meshing" if number == 1 else "Retrying meshing")


def stage(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Mark a native or vectorized operation whose work count is unknown."""

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            report(name)
            return function(*args, **kwargs)

        return wrapped

    return decorate


class WorkProgress:
    """Count real loop work without owning any terminal output.

    A convergence loop has no known total; its iteration limit is not a prediction.

    """

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        *,
        total: int | None = None,
        desc: str = "Meshing",
        unit: str = "iterations",
        **kwargs: Any,
    ) -> None:
        """Track an iterable's work or an open-ended convergence loop."""
        self.iterable = iterable
        self.total = (
            len(iterable)
            if iterable is not None and hasattr(iterable, "__len__")
            else total
        )
        self.desc = desc
        self.unit = unit
        self.n = 0

    def __enter__(self) -> "WorkProgress":
        """Publish the start of this loop."""
        self.refresh()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Leave completed work unchanged on early convergence or failure."""

    def __iter__(self) -> Iterator[T]:
        """Publish counts only after each yielded item finishes."""
        self.refresh()
        iterable = self.iterable if self.iterable is not None else ()
        for item in iterable:
            yield item
            self.update()

    def update(self, n: int = 1) -> None:
        """Record completed loop iterations."""
        self.n += n
        self.refresh()

    def set_description(self, desc: str, **kwargs: Any) -> None:
        """Publish the loop's current description while retaining its count."""
        self.desc = desc
        self.refresh()

    def refresh(self) -> None:
        """Publish a snapshot rather than a terminal bar."""
        report(self.desc, self.n, self.total, unit=self.unit)


def run_job(
    queue: Any,
    knee: str,
    tissue: str,
    function: Callable[..., T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> T:
    """Bind structured reporting to one worker task and restore worker state."""
    previous = None
    last_sent = 0.0

    def send(
        name: str,
        completed: int,
        total: int | None,
        attempt: int,
        status: str,
        unit: str,
    ) -> None:
        nonlocal previous, last_sent
        key = (name, total, attempt, status, unit)
        now = monotonic()
        if (
            key == previous
            and completed > 0
            and (total is None or completed < total)
            and now - last_sent < 0.1
        ):
            return
        queue.put(
            ProgressEvent(knee, tissue, name, completed, total, attempt, status, unit)
        )
        previous, last_sent = key, now

    token = _reporter.set(send)
    attempt_token = _attempt.set(1)
    try:
        report("Starting")
        result = function(*args, **kwargs)
        report(
            "Mesh ready" if result is not None else "Meshing failed",
            status="ready" if result is not None else "failed",
        )
        return result
    except BaseException:
        report("Worker failed", status="failed")
        raise
    finally:
        _reporter.reset(token)
        _attempt.reset(attempt_token)


def _synchronized(function: Callable[P, T]) -> Callable[P, T]:
    """Serialize parent-thread row updates with periodic elapsed-time refreshes."""

    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            return function(self, *args, **kwargs)

    return wrapped


class BatchProgress:
    """Render one batch bar and fixed tissue rows exclusively in the parent."""

    def __init__(self, total: int, queue: Any, *, auto_refresh: bool = True) -> None:
        """Create parent-owned bars and a periodic elapsed-time refresh thread."""
        self.lock = RLock()
        self.stop = Event()
        self.thread = None
        self.queue = queue
        self.batch = tqdm(total=total, desc="Knees", position=0, unit="knee")
        self.rows: dict[str, Any] = {}
        self.events: dict[str, ProgressEvent] = {}
        self.knee = ""
        self.counts = dict(exported=0, skipped=0, failed=0)
        self._counts()
        if auto_refresh:
            self.thread = Thread(target=self._refresh, name="mesh-progress")
            self.thread.start()

    def _refresh(self) -> None:
        """Keep elapsed time visible during blocking parent-side native work."""
        while not self.stop.wait(0.2):
            self.drain()

    def _counts(self) -> None:
        """Keep all terminal batch outcomes visible."""
        self.batch.set_postfix(self.counts)

    @_synchronized
    def start_knee(self, knee: str, tissues: list[str]) -> None:
        """Reuse fixed row positions for the next selected knee."""
        self.drain()
        for row in reversed(list(self.rows.values())):
            row.close()
        self.rows.clear()
        self.events.clear()
        self.knee = knee
        self.batch.set_description(f"Knees | {knee}")
        for position, tissue in enumerate(tissues, 1):
            self.rows[tissue] = tqdm(
                total=None,
                desc=f"{tissue}: pending",
                position=position,
                leave=False,
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
            )
            self.events[tissue] = ProgressEvent(
                knee, tissue, "Pending", status="pending"
            )

    @_synchronized
    def apply(self, event: ProgressEvent) -> None:
        """Ignore old knees and late updates after a terminal tissue outcome."""
        if event.knee != self.knee or event.tissue not in self.rows:
            return
        old = self.events[event.tissue]
        if old.status in {"failed", "exported", "cancelled"}:
            return
        if event.attempt < old.attempt:
            return
        row = self.rows[event.tissue]
        if (event.stage, event.attempt) != (
            old.stage,
            old.attempt,
        ) or event.completed < old.completed:
            row.reset(total=event.total)
            # tqdm.reset(None) retains its previous total.
            row.total = event.total
        row.n = event.completed
        row.unit = event.unit
        row.set_description(
            f"{event.tissue}: {event.stage} (attempt {event.attempt})",
            refresh=False,
        )
        row.set_postfix_str(event.unit, refresh=False)
        self.events[event.tissue] = event
        row.refresh()

    @_synchronized
    def tissue(self, tissue: str, name: str, **kwargs: Any) -> None:
        """Display a parent-owned preparation, writing, or finalization stage."""
        attempt = self.events[tissue].attempt
        self.apply(ProgressEvent(self.knee, tissue, name, attempt=attempt, **kwargs))

    @_synchronized
    def drain(self) -> None:
        """Drain available worker events and refresh elapsed time while waiting."""
        while True:
            try:
                self.apply(self.queue.get_nowait())
            except Empty:
                break
        for row in self.rows.values():
            row.refresh()

    @_synchronized
    def finish_knee(self, outcome: Literal["exported", "skipped", "failed"]) -> None:
        """Advance the batch once after this knee reaches its final outcome."""
        self.drain()
        for tissue, event in self.events.items():
            if event.status not in {"exported", "failed", "cancelled"}:
                self.tissue(tissue, "Not exported", status="failed")
        self.counts[outcome] += 1
        self.batch.update()
        self._counts()

    def close(self) -> None:
        """Close nested rows before the batch bar."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        for row in reversed(list(self.rows.values())):
            row.close()
        self.batch.close()


def wait_for_results(
    jobs: list[tuple[str, Any]], progress: BatchProgress, pool: Any
) -> list[Any]:
    """Poll every result while draining events, including out-of-order results."""
    remaining = dict(enumerate(jobs))
    results = [None] * len(jobs)
    worker_pids = getattr(
        pool, "_mesh_worker_pids", {worker.pid for worker in pool._pool}
    )
    while remaining:
        progress.drain()
        if worker_pids != {worker.pid for worker in pool._pool} or any(
            worker.exitcode is not None for worker in pool._pool
        ):
            raise RuntimeError("A meshing worker exited before returning its result.")
        for index, (tissue, job) in list(remaining.items()):
            if not job.ready():
                continue
            try:
                results[index] = job.get()
            except Exception:
                progress.tissue(tissue, "Worker failed", status="failed")
                logging.exception("Meshing failed for %s", tissue)
            else:
                if results[index] is None:
                    progress.tissue(tissue, "Meshing failed", status="failed")
            del remaining[index]
        if remaining:
            sleep(0.1)
    progress.drain()
    return results
