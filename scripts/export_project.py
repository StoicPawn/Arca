#!/usr/bin/env python3
"""Utility per creare un archivio ZIP dell'intero progetto.

Lo script genera un archivio che contiene l'intera repository, inclusi README,
configurazioni e codice sorgente. Per impostazione predefinita il file viene
creato nella cartella ``exports`` con un timestamp nel nome per evitare
sovrascritture. È possibile specificare un percorso di destinazione differente
tramite l'opzione ``--output``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

IGNORED_DIRECTORIES = {".git", "__pycache__", "exports"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea un archivio ZIP contenente tutti i file della repository, "
            "escludendo directory temporanee come .git ed exports."
        )
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=(
            "Percorso completo del file ZIP da creare. Se omesso viene creato "
            "un file nella cartella 'exports' con un timestamp nel nome."
        ),
    )
    return parser.parse_args()


def determine_output_path(provided_path: Path | None, repo_root: Path) -> Path:
    if provided_path is not None:
        if provided_path.is_dir():
            raise ValueError("Il percorso di output non può essere una cartella.")
        return provided_path.resolve()

    exports_dir = repo_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return exports_dir / f"{repo_root.name}-{timestamp}.zip"


def add_directory_to_zip(zip_file: ZipFile, repo_root: Path) -> None:
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        # Aggiorna dirs in-place per evitare la discesa nelle cartelle ignorate.
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]

        for file_name in files:
            file_path = root_path / file_name
            relative_path = file_path.relative_to(repo_root)
            zip_file.write(file_path, arcname=relative_path)


def create_archive(repo_root: Path, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_file, mode="w", compression=ZIP_DEFLATED) as zip_file:
        add_directory_to_zip(zip_file, repo_root)
    return output_file


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args()

    try:
        output_path = determine_output_path(args.output, repo_root)
    except ValueError as exc:
        raise SystemExit(str(exc))

    archive = create_archive(repo_root, output_path)
    print(f"Archivio creato: {archive}")


if __name__ == "__main__":
    main()
