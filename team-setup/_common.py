"""Shared paths and integrity checks for the pinned team text release."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = REPO_ROOT / "project/data/processed/cmv-text/v1"
LINKS_FILE = REPO_ROOT / "project/data/cmv-text-download-links.json"
PROJECT_URL = "https://xofhtpbusoutxtvrdynb.supabase.co"
BUCKET = "cmv-text"
PREFIX = "cmv-text/v1"
MANIFEST_SHA256 = "324e4ef93bf16d2928abd89e1dab1ed34652c2a98ff81d4f17741a64495391c0"


def load_environment():
    """Load this checkout's .env without expanding password text such as ${x}."""
    load_dotenv(REPO_ROOT / ".env", interpolate=False)


def local_path(root, relative):
    """Resolve a release-relative path, rejecting traversal and escaping symlinks."""
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or "\\" in relative
            or ":" in relative or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError(f"Invalid release path: {relative!r}")
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Release path escapes destination: {relative!r}")
    return target


def read_manifest(data):
    """Accept only the published v1 manifest; return it and required file entries."""
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Manifest does not match the pinned CMV text v1 release")
    manifest = json.loads(data)
    if manifest["version"] != "v1":
        raise ValueError("Expected text release v1")
    entries = [manifest["discussion_index"], *manifest["files"]]
    seen = {"manifest.json"}
    for entry in entries:
        relative = entry["path"]
        local_path(TEXT_DIR, relative)
        if relative in seen:
            raise ValueError(f"Duplicate release path: {relative}")
        seen.add(relative)
    return manifest, entries


def file_matches(path, expected):
    """Check cached bytes without loading a Parquet shard into memory."""
    if not path.is_file() or ("bytes" in expected and path.stat().st_size != expected["bytes"]):
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"]


def validate_download_url(relative, url):
    """Allow only HTTPS signed URLs for this project's exact release object."""
    parsed = urlsplit(url)
    expected_path = f"/storage/v1/object/sign/{BUCKET}/{PREFIX}/{relative}"
    if (parsed.scheme != "https" or parsed.netloc != urlsplit(PROJECT_URL).netloc
            or unquote(parsed.path) != expected_path or not parsed.query or parsed.fragment):
        raise ValueError(f"Invalid signed download URL for {relative}")
    return url
