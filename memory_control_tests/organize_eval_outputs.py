import argparse
import shutil
from pathlib import Path


def copy_matching(pattern: str, source_dir: Path, dest_dir: Path) -> int:
    count = 0
    for path in sorted(source_dir.glob(pattern)):
        dest = dest_dir / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        count += 1
    return count


def copy_exact_baseline(source_dir: Path, dest_dir: Path, suffix: str) -> int:
    count = 0
    for path in sorted(source_dir.glob(f"conversation_*{suffix}")):
        stem = path.name
        if ".no_store." in stem or ".forget." in stem or ".no_use." in stem:
            continue
        dest = dest_dir / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize baseline vs test-world eval outputs into separate data directories.")
    parser.add_argument("--source_dir", default="data/baseline/travelPlanning")
    parser.add_argument("--baseline_dest", default="data/eval/baseline/travelPlanning")
    parser.add_argument("--test_dest", default="data/eval/test_worlds/travelPlanning")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    baseline_dest = Path(args.baseline_dest)
    test_dest = Path(args.test_dest)

    baseline_dest.mkdir(parents=True, exist_ok=True)
    test_dest.mkdir(parents=True, exist_ok=True)

    copy_exact_baseline(source_dir, baseline_dest / "plain", ".recall_eval_gpt-5.4-mini.json")
    copy_exact_baseline(source_dir, baseline_dest / "mem0_retrieval", ".mem0_retrieval_eval_gpt-5.4-mini.json")

    for world in ["no_store", "forget", "no_use"]:
        copy_matching(f"conversation_*.{world}.recall_eval_gpt-5.4-mini.json", source_dir, test_dest / world / "plain")
        copy_matching(
            f"conversation_*.{world}.mem0_retrieval_eval_gpt-5.4-mini.json",
            source_dir,
            test_dest / world / "mem0_retrieval",
        )


if __name__ == "__main__":
    main()
