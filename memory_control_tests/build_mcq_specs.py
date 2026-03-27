import argparse
import shutil
from pathlib import Path

from .mcq_specs import build_mcq_spec_bundle


def iter_baseline_files(source_dir: Path):
    for path in sorted(source_dir.rglob("conversation_*.memory_control.json")):
        yield path


def build_one(sidecar_path: Path, source_root: Path, dest_root: Path) -> None:
    bundle = build_mcq_spec_bundle(str(sidecar_path))
    relative = sidecar_path.relative_to(source_root)
    dest_path = dest_root / relative
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if sidecar_path.resolve() != dest_path.resolve():
        shutil.copy2(sidecar_path, dest_path)
    mcq_path = dest_path.with_name(dest_path.name.replace(".memory_control.json", ".mcq_specs.json"))
    mcq_path.write_text(bundle, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MCQ spec bundles from baseline memory-control sidecars.")
    parser.add_argument("--source_dir", default="data/baseline")
    parser.add_argument("--dest_dir", default="data/baseline")
    args = parser.parse_args()

    source_root = Path(args.source_dir)
    dest_root = Path(args.dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    for path in iter_baseline_files(source_root):
        build_one(path, source_root, dest_root)


if __name__ == "__main__":
    main()
