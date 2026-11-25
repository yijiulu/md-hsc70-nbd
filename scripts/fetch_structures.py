#!/usr/bin/env python3
"""Download public RCSB PDB structures required by the Hsc70 NBD pipeline."""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HSC70_PDBS = ("1KAX", "3HSC", "1YUW")
DNAK_PDBS = ("7RAX", "4B9Q", "4JNE", "7KRT", "7KO2", "5NRO")


def fetch_pdb(pdb_id: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"[SKIP] {dest} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"[GET ] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    struct_dir = ROOT / "structures"
    dnak_dir = struct_dir / "dnak"
    for code in HSC70_PDBS:
        fetch_pdb(code, struct_dir / f"{code}.pdb")
    for code in DNAK_PDBS:
        fetch_pdb(code, dnak_dir / f"{code}.pdb")
    print("[OK] All public crystal structures ready.")


if __name__ == "__main__":
    main()
