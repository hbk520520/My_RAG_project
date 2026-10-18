"""Restore validated runtime bundles into a new D-drive directory."""
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from .contracts import local_path


def restore_bundle(source, destination):
    source, destination = local_path(source), local_path(destination)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != 1 or manifest["run_db"] not in manifest["files"]:
        raise ValueError("Unsupported runtime backup manifest")
    if destination.exists():
        raise ValueError("Restore destination must be a new directory")
    for name, digest in manifest["files"].items():
        if local_path(source / name).parent != source or name != Path(name).name:
            raise ValueError("Invalid backup filename")
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Backup checksum mismatch")
    destination.mkdir(parents=True)
    for name, digest in manifest["files"].items():
        path = destination / name
        shutil.copyfile(source / name, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Restored checksum mismatch")
        connection = sqlite3.connect(path)
        try:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Restored database integrity check failed")
        finally:
            connection.close()
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return destination / manifest["run_db"]
