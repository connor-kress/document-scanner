#!/usr/bin/env python
"""Download and install the pretrained model release into models/."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath


PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
REPOSITORY = "connor-kress/document-scanner"
DEFAULT_MODEL_VERSION = "1.0.0"
VERSION_FILE = ".release-version"
PINNED_SHA256 = {
    "1.0.0": "d465c1d0c59a51d730870a3eab09902a871a60b71b9e08330f9d6233524a1cfe",
}
REQUIRED_FILES = (
    "LICENSE",
    "ridge/model.joblib",
    "ridge/config.yaml",
    "mlp/model.pt",
    "mlp/pca.joblib",
    "mlp/config.yaml",
    "cnn_gray/model.pt",
    "cnn_gray/config.yaml",
    "cnn_rgb/model.pt",
    "cnn_rgb/config.yaml",
    "evaluation/final_comparison.csv",
    "evaluation/by_background.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_version(version: str) -> str:
    if version.startswith("v-"):
        version = version[2:]
    elif version.startswith("v"):
        version = version[1:]
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]*", version):
        raise ValueError(f"invalid model version: {version}")
    return version


def archive_root(version: str) -> str:
    return f"document-scanner-models-v{version}"


def release_asset_url(version: str, asset: str) -> str:
    return f"https://github.com/{REPOSITORY}/releases/download/models-v{version}/{asset}"


def read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "document-scanner"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def latest_model_version() -> str:
    releases = json.loads(read_url(f"https://api.github.com/repos/{REPOSITORY}/releases?per_page=100"))
    for release in releases:
        tag = release.get("tag_name", "")
        if tag.startswith("models-v") and not release.get("draft") and not release.get("prerelease"):
            return normalize_version(tag.removeprefix("models-"))
    raise RuntimeError("GitHub does not have a published model release")


def release_checksum(version: str, archive_name: str) -> str:
    if version in PINNED_SHA256:
        return PINNED_SHA256[version]
    checksum_url = release_asset_url(version, "SHA256SUMS")
    for line in read_url(checksum_url).decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == archive_name:
            if re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
                return parts[0].lower()
    raise RuntimeError(f"SHA256SUMS does not contain a checksum for {archive_name}")


def installed_version(destination: Path) -> str | None:
    marker = destination / VERSION_FILE
    if not marker.is_file():
        return None
    version = marker.read_text().strip()
    return normalize_version(version) if version else None


def models_ready(destination: Path = MODELS) -> bool:
    return all((destination / relative).is_file() for relative in REQUIRED_FILES)


def download_archive(url: str, destination: Path, version: str) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading pretrained models {version} from {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "document-scanner"})
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe path in model archive: {member.name}")
            target = destination.joinpath(*relative.parts).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in model archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read model archive entry: {member.name}")
                with source, open(target, "wb") as output:
                    shutil.copyfileobj(source, output)
            else:
                raise RuntimeError(f"unsupported entry in model archive: {member.name}")


def download_models(
    destination: Path = MODELS,
    *,
    version: str = DEFAULT_MODEL_VERSION,
    force: bool = False,
    url: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    version = normalize_version(version)
    destination = destination.resolve()
    if models_ready(destination) and not force:
        current_version = installed_version(destination)
        if current_version == version:
            print(f"Using existing pretrained models {version}: {destination}")
            return destination
        if current_version is None:
            raise RuntimeError(
                f"model directory has no release version: {destination}; "
                "pass --force to replace it"
            )
        print(f"Replacing pretrained models {current_version} with {version}")
    if destination.exists() and not force:
        if not models_ready(destination):
            raise RuntimeError(
                f"model directory exists but is incomplete: {destination}; "
                "pass --force to replace it"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".models-", dir=destination.parent) as temporary:
        temporary_path = Path(temporary)
        root_name = archive_root(version)
        archive_name = f"{root_name}.tar.gz"
        archive = temporary_path / archive_name
        extracted = temporary_path / "extracted"
        url = url or release_asset_url(version, archive_name)
        expected_sha256 = expected_sha256 or release_checksum(version, archive_name)
        download_archive(url, archive, version)
        if sha256(archive) != expected_sha256:
            raise RuntimeError("downloaded model archive failed SHA-256 verification")

        safe_extract(archive, extracted)
        source = extracted / root_name
        if not models_ready(source):
            raise RuntimeError("model archive does not contain the expected files")
        (source / VERSION_FILE).write_text(f"{version}\n")

        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        source.replace(destination)

    print(f"Installed pretrained models in {destination}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS,
        help=f"installation directory (default: {MODELS})",
    )
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument(
        "--version",
        default=DEFAULT_MODEL_VERSION,
        help=f"model release version, with or without a leading 'v' (default: {DEFAULT_MODEL_VERSION})",
    )
    version_group.add_argument(
        "--latest",
        action="store_true",
        help="download the newest published model release",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing model directory")
    args = parser.parse_args(argv)
    version = latest_model_version() if args.latest else args.version
    download_models(args.output, version=version, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
