#!/usr/bin/env bash
# Download and verify the SmartDoc 2015 Challenge 1 v2.0.0 frame archive.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
download_dir="${project_root}/data/downloads"
raw_dir="${project_root}/data/raw"
frames_dir="${raw_dir}/frames"
archive="${download_dir}/frames-v2.0.0.tar.gz"
partial_archive="${archive}.part"
complete_marker="${raw_dir}/.smartdoc-frames-v2.0.0.complete"

release_url="https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/frames.tar.gz"
# From the release's official sha256.chksum asset.
expected_sha256="3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae"
expected_frames=24889

mkdir -p "${download_dir}" "${raw_dir}"

verify_archive() {
    printf '%s  %s\n' "${expected_sha256}" "${archive}" | sha256sum --check --status
}

verify_extraction() {
    [[ -f "${frames_dir}/metadata.csv.gz" ]] || return 1
    local frame_count
    frame_count="$(find "${frames_dir}" -type f \( -iname '*.jpeg' -o -iname '*.jpg' \) | wc -l)"
    [[ "${frame_count}" -eq "${expected_frames}" ]]
}

if [[ -f "${complete_marker}" ]] && verify_extraction; then
    echo "SmartDoc v2.0.0 frames are already extracted; download step is complete."
    exit 0
fi

if [[ -f "${archive}" ]] && ! verify_archive; then
    bad_archive="${archive}.bad.$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Existing archive failed SHA-256 verification; preserving it as ${bad_archive}."
    mv "${archive}" "${bad_archive}"
fi

if [[ ! -f "${archive}" ]]; then
    echo "Downloading SmartDoc v2.0.0 frames (1,019,404,933 bytes)..."
    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 5 \
        --continue-at - \
        --output "${partial_archive}" \
        "${release_url}"
    mv "${partial_archive}" "${archive}"
fi

echo "Verifying official SHA-256 checksum..."
verify_archive
tar -tzf "${archive}" >/dev/null

echo "Extracting frames into ${frames_dir}..."
mkdir -p "${frames_dir}"
tar -xzf "${archive}" -C "${frames_dir}"

if ! verify_extraction; then
    echo "Extraction validation failed: expected metadata.csv.gz and ${expected_frames} JPEG frames." >&2
    exit 1
fi

touch "${complete_marker}"
echo "SmartDoc v2.0.0 is ready at ${frames_dir}."
