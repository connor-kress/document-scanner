"""Create timestamped training directories and resolve completed runs."""
from __future__ import annotations

import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}\.\d{6}Z(?:-\d+)?$")


@contextmanager
def run_directory(results_root: str | Path, model_type: str) -> Iterator[Path]:
    """Create one timestamped run and mark it complete on successful exit."""
    if not model_type or Path(model_type).name != model_type:
        raise ValueError(f"invalid model type: {model_type!r}")
    parent = Path(results_root) / model_type
    parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = parent / run_id
    suffix = 1
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            path = parent / f"{run_id}-{suffix}"
            suffix += 1
    yield path
    (path / "COMPLETE").touch()


def latest_run(
    results_root: str | Path,
    model_type: str,
    *,
    required: tuple[str, ...] = (),
) -> Path:
    """Return the newest completed run containing all required artifacts.

    A fixed-layout directory from older versions is returned only when no
    timestamped completed run is available.
    """
    model_root = Path(results_root) / model_type
    if model_root.is_dir():
        candidates = sorted(
            (
                path
                for path in model_root.iterdir()
                if path.is_dir() and _RUN_ID_RE.fullmatch(path.name)
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        for path in candidates:
            if (path / "COMPLETE").is_file() and all((path / name).is_file() for name in required):
                return path
        if all((model_root / name).is_file() for name in required):
            return model_root
    raise FileNotFoundError(
        f"no completed {model_type!r} run containing {required} under {model_root}"
    )
