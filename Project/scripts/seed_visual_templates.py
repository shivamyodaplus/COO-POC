#!/usr/bin/env python3
"""
Seed signature and stamp visual templates from image files.

Filename convention expected:
    {num}_{DocType}__{template_type}__{name}.png
    e.g.  01_AfCFTA__signature__ahmed_alsayed.png
          01_AfCFTA__stamp__goeic_origin.png

Usage (inside the running app container):
    docker exec <app-container> python scripts/seed_visual_templates.py

Usage (local, with backend reachable at http://localhost:8000):
    BACKEND_URL=http://localhost:8000 python scripts/seed_visual_templates.py

You can also point to a custom images directory:
    TEMPLATES_DIR=/path/to/images python scripts/seed_visual_templates.py
"""
from __future__ import annotations

import mimetypes
import os
import re
import sys
from pathlib import Path

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "yolostructuredetection" / "test" / "test" / "files"
TEMPLATES_DIR = Path(os.getenv("TEMPLATES_DIR", str(DEFAULT_TEMPLATES_DIR)))

VALID_TYPES = {"signature", "sign", "stamp", "logo"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}

# Mapping from numeric prefix to doc_type
_DOC_TYPE_MAP: dict[str, str] = {
    "01": "AfCFTA",
    "02": "COMESA",
    "03": "AGADIR",
    "04": "GAFTA",
}


def _parse_filename(stem: str) -> dict[str, str] | None:
    """
    Parse a filename stem like '01_AfCFTA__signature__ahmed_alsayed'
    Returns dict with keys: doc_type, template_type, name
    or None if the filename doesn't match the expected pattern.
    """
    # Pattern: {num}_{DocType}__{type}__{rest}
    m = re.match(r"^(\d+)_([^_]+)__(\w+)__(.+)$", stem)
    if m:
        num, doc_type_raw, ttype, name_raw = m.groups()
        ttype_lower = ttype.strip().lower()
        if ttype_lower not in VALID_TYPES:
            return None
        return {
            "doc_type": _DOC_TYPE_MAP.get(num, doc_type_raw),
            "template_type": ttype_lower,
            "name": name_raw.replace("_", " ").title(),
        }
    return None


def seed_templates(dry_run: bool = False) -> None:
    if not TEMPLATES_DIR.exists():
        print(f"ERROR: Templates directory not found: {TEMPLATES_DIR}", file=sys.stderr)
        sys.exit(1)

    # Check backend health
    try:
        resp = requests.get(f"{BACKEND_URL}/api/v1/status", timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"ERROR: Cannot reach backend at {BACKEND_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Fetch already-existing templates to avoid duplicates
    existing_resp = requests.get(f"{BACKEND_URL}/api/v1/visual-templates", timeout=30)
    existing_names: set[str] = set()
    if existing_resp.ok:
        for item in existing_resp.json():
            if item.get("template_type") in VALID_TYPES:
                existing_names.add(item["name"])

    files = sorted(
        f for f in TEMPLATES_DIR.iterdir()
        if f.suffix.lower() in IMAGE_SUFFIXES and f.is_file()
    )

    uploaded = 0
    skipped = 0
    failed = 0

    for fpath in files:
        parsed = _parse_filename(fpath.stem)
        if parsed is None:
            print(f"  SKIP  {fpath.name}  (filename doesn't match pattern)")
            skipped += 1
            continue

        template_name = f"{parsed['doc_type']} {parsed['name']}"
        if template_name in existing_names:
            print(f"  SKIP  {fpath.name}  (already exists: '{template_name}')")
            skipped += 1
            continue

        content_type = mimetypes.guess_type(str(fpath))[0] or "image/png"
        print(f"  UPLOAD  {fpath.name}  →  '{template_name}' [{parsed['template_type']}]", end=" ")

        if dry_run:
            print("(dry run)")
            continue

        with open(fpath, "rb") as fh:
            image_bytes = fh.read()

        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/v1/visual-templates/upload",
                data={
                    "name": template_name,
                    "template_type": parsed["template_type"],
                    "doc_type": parsed["doc_type"],
                    "doc_category": "coo",
                    "country": "Egypt",
                },
                files={"file": (fpath.name, image_bytes, content_type)},
                timeout=120,
            )
            if resp.ok:
                rec = resp.json()
                print(f"OK  (id={rec['id']})")
                uploaded += 1
            else:
                print(f"FAILED  ({resp.status_code}: {resp.text[:120]})")
                failed += 1
        except Exception as exc:
            print(f"FAILED  ({exc})")
            failed += 1

    print(f"\nDone. Uploaded: {uploaded}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("=== DRY RUN ===")
    print(f"Backend : {BACKEND_URL}")
    print(f"Source  : {TEMPLATES_DIR}\n")
    seed_templates(dry_run=dry_run)
