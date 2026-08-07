#!/usr/bin/env python
"""Recover correctly named ICML template files from the supplied scrambled bundle."""

from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "TemplatePaperLatex"
DESTINATION = ROOT / "paper"

MAPPING = {
    "icml2024.sty": "icml2024 (1).bst",
    "icml2024.bst": "fancyhdr (1).sty",
    "fancyhdr.sty": "Main.synctex (1).gz",
    "algorithm.sty": "Main (1).out",
    "algorithmic.sty": "algorithm (1).sty",
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for destination, source in MAPPING.items():
        source_path = SOURCE / source
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        copyfile(source_path, DESTINATION / destination)
    print(f"Recovered {len(MAPPING)} template files into {DESTINATION}")


if __name__ == "__main__":
    main()
