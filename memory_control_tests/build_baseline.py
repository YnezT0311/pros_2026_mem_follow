import argparse
import shutil
from pathlib import Path

from .common import build_baseline_spec, dump_json, load_json


def iter_conversation_files(source_dir: Path):
    for path in sorted(source_dir.rglob("conversation_*.json")):
        yield path


def build_one(source_path: Path, source_root: Path, dest_root: Path) -> None:
    data = load_json(str(source_path))
    spec = build_baseline_spec(data, str(source_path))

    relative = source_path.relative_to(source_root)
    dest_path = dest_root / relative
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    dump_json(str(dest_path.with_suffix(".memory_control.json")), spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline memory-control bundles from output conversations.")
    parser.add_argument("--source_dir", default="data/output")
    parser.add_argument("--dest_dir", default="data/baseline")
    args = parser.parse_args()

    source_root = Path(args.source_dir)
    dest_root = Path(args.dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    for path in iter_conversation_files(source_root):
        build_one(path, source_root, dest_root)


if __name__ == "__main__":
    main()

