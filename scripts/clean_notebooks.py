#!/usr/bin/env python3
"""Check or clear outputs from Jupyter notebooks without third-party packages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ALLOWLIST_FILE = ".notebook-output-allowlist"


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed")
    return result.stdout


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("run this command from inside the Git repository")
    return Path(result.stdout.strip())


def nul_separated_paths(output: bytes) -> list[str]:
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in output.split(b"\0")
        if item
    ]


def load_allowlist(repo: Path) -> set[str]:
    path = repo / ALLOWLIST_FILE
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def selected_paths(repo: Path, paths: list[str], staged: bool) -> list[str]:
    if staged:
        output = git(
            repo,
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            "--",
            "*.ipynb",
        )
        return nul_separated_paths(output)

    if not paths:
        return nul_separated_paths(git(repo, "ls-files", "-z", "--", "*.ipynb"))

    selected = []
    for value in paths:
        path = Path(value)
        if path.is_absolute():
            try:
                path = path.relative_to(repo)
            except ValueError as error:
                raise ValueError(f"path is outside the repository: {value}") from error
        selected.append(path.as_posix().removeprefix("./"))
    return selected


def notebook_bytes(repo: Path, path: str, staged: bool) -> bytes:
    if staged:
        return git(repo, "show", f":{path}")
    return (repo / path).read_bytes()


def clean_notebook(data: bytes) -> tuple[dict, list[str], bool]:
    notebook = json.loads(data)
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("expected a notebook object containing a cells list")

    problems = []
    changed = False

    for index, cell in enumerate(notebook["cells"], start=1):
        if not isinstance(cell, dict):
            raise ValueError(f"cell {index} is not an object")
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            problems.append(f"cell {index} has outputs")
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            problems.append(f"cell {index} has an execution count")
            cell["execution_count"] = None
            changed = True

    return notebook, problems, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="notebooks to process (defaults to all notebooks tracked by Git)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fix",
        action="store_true",
        help="clear outputs and execution counts in place",
    )
    mode.add_argument(
        "--staged",
        action="store_true",
        help="check notebook contents staged for commit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = repo_root()
        paths = selected_paths(repo, args.paths, args.staged)
        allowlist = load_allowlist(repo)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    failures = 0
    changed_files = 0
    checked_files = 0

    for path in paths:
        if path in allowlist:
            continue
        checked_files += 1
        try:
            data = notebook_bytes(repo, path, args.staged)
            notebook, problems, changed = clean_notebook(data)
        except (
            OSError,
            RuntimeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            print(f"{path}: could not read notebook: {error}", file=sys.stderr)
            failures += 1
            continue

        if args.fix and changed:
            output = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
            try:
                (repo / path).write_text(output, encoding="utf-8")
            except OSError as error:
                print(f"{path}: could not write notebook: {error}", file=sys.stderr)
                failures += 1
                continue
            print(f"cleaned {path}")
            changed_files += 1
        elif problems:
            print(f"{path}: {'; '.join(problems)}", file=sys.stderr)
            failures += 1

    if failures:
        print(
            "Notebook outputs must be cleared. Run "
            "`python scripts/clean_notebooks.py --fix`, then stage the files again.",
            file=sys.stderr,
        )
        return 1

    if args.fix:
        print(f"Cleaned {changed_files} of {checked_files} checked notebook(s).")
    else:
        print(f"Checked {checked_files} notebook(s); all outputs are clear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
