#!/usr/bin/env python3
"""Audit CSS-local temporary model artifacts.

Default: prints exact smoke checkpoint files/result dirs and byte totals.
--delete-smoke: requires typing DELETE_SMOKE before removing ONLY the
listed known smoke/debug artifacts. No entire formal experiment folder,
data feature pickle, baseline checkpoint, Git file, or report is touched.

Use from the CSS repository; local server artifacts are untracked.
"""
import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SAFE_RESULT_SMOKE_DIRS = {
    "v2_smoke_adaptive_other",
    "v3_supcon_smoke",
    "v8_other_smoke",
}


def in_root(path):
    # Skip symlinks in all cases; resolved paths must stay within CSS.
    if path.is_symlink():
        return False
    try:
        return path.resolve().is_relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return False


def size_bytes(path):
    if path.is_file():
        return path.stat().st_size
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file() and not p.is_symlink() and in_root(p)
    )


def find_smoke_artifacts(root=ROOT):
    checkpoints = root / "checkpoints"
    results = root / "results"
    files = []
    dirs = []
    if checkpoints.is_dir():
        for p in sorted(checkpoints.glob("*.pt")):
            name = p.name.lower()
            if ("smoke" in name or "gradient_probe" in name) and p.is_file() and in_root(p):
                files.append(p)
    if results.is_dir():
        for p in sorted(results.iterdir()):
            if p.name in SAFE_RESULT_SMOKE_DIRS and p.is_dir() and in_root(p):
                dirs.append(p)
    return files, dirs


def format_size(n):
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n = n / 1024
        if n < 1024:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PiB"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete-smoke", action="store_true",
                        help="interactive confirmation; permanently delete only displayed smoke/debug artifacts")
    args = parser.parse_args()
    files, dirs = find_smoke_artifacts()
    items = [*files, *dirs]
    print("CSS temporary artifact inventory (NO deletion by default):")
    total = 0
    for p in items:
        n = size_bytes(p)
        total += n
        print(f"  {format_size(n):>12}  {p.relative_to(ROOT)}")
    print(f"TOTAL: {format_size(total)} across {len(items)} candidates")
    print("Preserved: all data/*.pkl, checkpoints/*baseline*, checkpoints/iemocap.pt,")
    print("checkpoints/meld.pt, ALL formal results, and research documents.")
    if not args.delete_smoke:
        print("DRY RUN ONLY. To remove only these candidates, use --delete-smoke.")
        return
    if not items:
        print("Nothing to delete.")
        return
    print("Permanent local deletion; check the list and back up anything you need.")
    if input("Type DELETE_SMOKE exactly to confirm: ").strip() != "DELETE_SMOKE":
        print("Cancelled. Nothing deleted.")
        return
    for p in items:
        # Revalidate before mutation; don't follow symlinks or delete outside repo.
        if not in_root(p):
            raise RuntimeError(f"path changed to symlink/outside repo: {p}")
        if p.is_file():
            p.unlink()
        elif p.is_dir() and p.name in SAFE_RESULT_SMOKE_DIRS:
            shutil.rmtree(p)
        else:
            raise RuntimeError(f"refusing unexpected type/path: {p}")
        print("Deleted:", p.relative_to(ROOT))
    print(f"Deleted {len(items)} smoke/debug candidates, approx {format_size(total)}.")


if __name__ == "__main__":
    main()
