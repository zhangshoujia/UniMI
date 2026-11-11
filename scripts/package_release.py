#!/usr/bin/env python3
"""Build a code archive from the public source, docs, configs, and examples."""

import argparse
import tempfile
import zipfile
from pathlib import Path


ROOT_FILES = {
    "README.md", "LICENSE", "THIRD_PARTY.md", "pyproject.toml",
    "requirements.txt", ".gitignore",
}
DIRECTORIES = {"src", "scripts", "configs", "docs", "tests", "examples"}
SUFFIXES = {
    ".py", ".sh", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".txt", ".tsv", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jinja",
}


def build_archive(root: Path, output: Path):
    root, output = root.resolve(), output.resolve()
    files = []
    candidates = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in DIRECTORIES:
        candidates.extend((root / name).rglob("*"))
    for path in sorted(candidates):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"Remove symlinks before packaging: {relative}")
        if not path.is_file():
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            if str(relative) not in ROOT_FILES:
                continue
        if str(relative) in ROOT_FILES or (
            relative.parts[0] in DIRECTORIES and path.suffix in SUFFIXES
        ):
            files.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, Path("UniMI") / path.relative_to(root))
        temporary.chmod(0o644)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Packed {len(files)} files: {output}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root.parent / "UniMI.zip")
    args = parser.parse_args()
    build_archive(root, args.output)


if __name__ == "__main__":
    main()
