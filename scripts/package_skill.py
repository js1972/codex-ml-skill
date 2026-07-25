#!/usr/bin/env python3
"""Build a deterministic .skill archive from the canonical skill source."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

FIXED_TIME = (2020, 1, 1, 0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="skills/ml-model-builder",
        help="Skill source directory relative to the repository",
    )
    parser.add_argument(
        "--output",
        default="dist/ml-model-builder.skill",
        help="Output archive relative to the repository",
    )
    return parser.parse_args()


def validate_source(source: Path) -> list[Path]:
    if not (source / "SKILL.md").is_file():
        raise ValueError(f"{source} does not contain SKILL.md")
    files = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if (
            not path.is_file()
            or path.is_symlink()
            or "__pycache__" in relative.parts
            or any(part.startswith(".") for part in relative.parts)
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(source).as_posix())


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    source = (repository / args.source).resolve()
    output = (repository / args.output).resolve()
    try:
        files = validate_source(source)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = path.relative_to(source)
            info = zipfile.ZipInfo(
                f"ml-model-builder/{relative.as_posix()}", date_time=FIXED_TIME
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.suffix == ".py" else 0o644) << 16
            archive.writestr(info, path.read_bytes())
    shutil.move(temporary, output)
    print(f"Packaged {len(files)} files into {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
