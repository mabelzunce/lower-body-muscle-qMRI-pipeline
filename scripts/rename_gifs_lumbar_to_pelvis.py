#!/usr/bin/env python3
r"""
Rename GIF files by replacing the word 'lumbar' with 'pelvis' in filenames.

Usage examples (PowerShell):
    # Preview changes (no files modified)
    python .\scripts\rename_gifs_lumbar_to_pelvis.py --path "C:\Users\mabel\OneDrive - NHS\CoreScorePaper\SegmentationChecksPelvis" --dry-run

    # Perform actual rename (no overwrite)
    python .\scripts\rename_gifs_lumbar_to_pelvis.py --path "C:\Users\mabel\OneDrive - NHS\CoreScorePaper\SegmentationChecksPelvis"

    # Force overwrite of existing targets
    python .\scripts\rename_gifs_lumbar_to_pelvis.py --path "C:\Users\mabel\OneDrive - NHS\CoreScorePaper\SegmentationChecksPelvis" --force

This script is safe by default (dry-run available). It supports recursive scanning and
handles case-insensitive occurrences of the word 'lumbar', preserving simple case
patterns (upper-case, title-case, lower-case) when replacing.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import Callable, List, Tuple


def replace_lumbar_with_pelvis_preserve_case(name: str) -> str:
    """Case-insensitive replace of 'lumbar' with 'pelvis', preserving simple case patterns.

    Preserves:
      - all lowercase -> 'pelvis'
      - all uppercase -> 'PELVIS'
      - title case (Lumbar) -> 'Pelvis'
    For mixed-case matches falls back to 'pelvis'.
    """

    def _repl(m: re.Match) -> str:
        s = m.group(0)
        if s.islower():
            return 'pelvis'
        if s.isupper():
            return 'PELVIS'
        if s.istitle():
            return 'Pelvis'
        return 'pelvis'

    return re.sub(r'lumbar', _repl, name, flags=re.IGNORECASE)


def find_gif_files(root: str, recursive: bool = False, pattern: str = '*.gif') -> List[str]:
    """Return a list of file paths to GIFs under root (non-recursive by default).

    pattern is currently ignored except for the implied .gif extension; kept for API parity.
    """
    files: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith('.gif'):
                    files.append(os.path.join(dirpath, fn))
    else:
        for entry in os.listdir(root):
            full = os.path.join(root, entry)
            if os.path.isfile(full) and entry.lower().endswith('.gif'):
                files.append(full)
    return files


def plan_renames(files: List[str]) -> List[Tuple[str, str]]:
    """Return list of (src, dst) pairs for files that contain 'lumbar' in the filename."""
    pairs: List[Tuple[str, str]] = []
    for src in files:
        dirname, fname = os.path.split(src)
        new_fname = replace_lumbar_with_pelvis_preserve_case(fname)
        if new_fname != fname:
            dst = os.path.join(dirname, new_fname)
            pairs.append((src, dst))
    return pairs


def do_renames(pairs: List[Tuple[str, str]], dry_run: bool = True, force: bool = False) -> Tuple[int, int]:
    """Perform or preview renames. Returns (renamed_count, skipped_count)."""
    renamed = 0
    skipped = 0
    for src, dst in pairs:
        if os.path.exists(dst):
            if os.path.samefile(src, dst):
                # same file (unlikely) - skip
                print(f"Skipping (same file): {src}")
                skipped += 1
                continue
            if not force:
                print(f"Target exists, skipping: {dst}")
                skipped += 1
                continue

        print(f"{('Would rename' if dry_run else 'Renaming')}:\n  {src}\n  -> {dst}")
        if not dry_run:
            # ensure target directory exists (it should), then rename/replace
            try:
                if force and os.path.exists(dst):
                    # remove existing target to allow replace on Windows
                    os.remove(dst)
                os.replace(src, dst)
                renamed += 1
            except Exception as e:
                print(f"Failed to rename {src} -> {dst}: {e}")
                skipped += 1

    return renamed, skipped


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rename GIF files replacing 'lumbar' with 'pelvis'")
    p.add_argument('--path', '-p', required=False,
                   default=r"C:/Users/mabel/OneDrive - NHS/CoreScorePaper/SegmentationChecksPelvis",
                   help='Directory containing GIF files to rename (default: provided path)')
    p.add_argument('--recursive', '-r', action='store_true', help='Scan directories recursively')
    p.add_argument('--dry-run', '-n', action='store_true', help='Show what would be renamed without changing files')
    p.add_argument('--force', '-f', action='store_true', help='Overwrite existing target files')
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = args.path

    if not os.path.isdir(root):
        print(f"Error: path not found or not a directory: {root}")
        return 2

    files = find_gif_files(root, recursive=args.recursive)
    if not files:
        print(f"No GIF files found in: {root} (recursive={args.recursive})")
        return 0

    pairs = plan_renames(files)
    if not pairs:
        print("No filenames containing the word 'lumbar' were found. Nothing to do.")
        return 0

    print(f"Found {len(files)} GIF file(s); {len(pairs)} will be renamed.")

    # show summary
    for src, dst in pairs:
        print(f"  {os.path.basename(src)} -> {os.path.basename(dst)}")

    renamed, skipped = do_renames(pairs, dry_run=args.dry_run, force=args.force)

    print('\nSummary:')
    if args.dry_run:
        print(f"  Preview only (dry-run). {len(pairs)} proposed renames; {skipped} would be skipped if run without --force.")
    else:
        print(f"  Renamed: {renamed}")
        print(f"  Skipped/Failed: {skipped}")

    return 0 if (args.dry_run or renamed > 0) else 1


if __name__ == '__main__':
    raise SystemExit(main())
