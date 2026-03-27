import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_rendered_files(source_dir: Path) -> List[Path]:
    return sorted(source_dir.rglob("conversation_*.recall_rendered.json"))


def export_one(rendered_path: Path, source_root: Path, dest_root: Path) -> None:
    rendered = load_json(rendered_path)
    relative_dir = rendered_path.parent.relative_to(source_root)
    stem = rendered_path.name.replace(".recall_rendered.json", "")

    whole_items = rendered.get("whole_recall_set", [])
    slot_items = rendered.get("slot_recall_set", [])

    whole_payload = {
        "source_rendered": str(rendered_path),
        "source_sidecar": rendered.get("source_sidecar", ""),
        "source_conversation": rendered.get("source_conversation", ""),
        "model": rendered.get("model", ""),
        "items": whole_items,
    }
    slot_payload = {
        "source_rendered": str(rendered_path),
        "source_sidecar": rendered.get("source_sidecar", ""),
        "source_conversation": rendered.get("source_conversation", ""),
        "model": rendered.get("model", ""),
        "items": slot_items,
    }
    application_payload = {
        "source_rendered": str(rendered_path),
        "status": "TODO_deferred",
        "note": "Application / reasoning benchmarks are not exported yet because the application family is still deferred.",
        "items": [],
    }

    dump_json(dest_root / relative_dir / "whole_recall" / f"{stem}.json", whole_payload)
    dump_json(dest_root / relative_dir / "slot_recall" / f"{stem}.json", slot_payload)
    dump_json(dest_root / relative_dir / "application" / f"{stem}.json", application_payload)


def build_aggregate_files(dest_root: Path) -> None:
    for qa_family in ["whole_recall", "slot_recall", "application"]:
        for topic_dir in sorted(dest_root.iterdir()):
            if not topic_dir.is_dir():
                continue
            family_dir = topic_dir / qa_family
            if not family_dir.exists():
                continue

            aggregate_items: List[Dict] = []
            source_files: List[str] = []
            source_rendered: List[str] = []
            source_sidecars: List[str] = []
            source_conversations: List[str] = []
            model_names: List[str] = []
            status = ""
            note = ""

            for path in sorted(family_dir.glob("conversation_*.json")):
                payload = load_json(path)
                aggregate_items.extend(payload.get("items", []))
                source_files.append(str(path))
                if payload.get("source_rendered"):
                    source_rendered.append(payload["source_rendered"])
                if payload.get("source_sidecar"):
                    source_sidecars.append(payload["source_sidecar"])
                if payload.get("source_conversation"):
                    source_conversations.append(payload["source_conversation"])
                if payload.get("model"):
                    model_names.append(payload["model"])
                if payload.get("status"):
                    status = payload["status"]
                if payload.get("note"):
                    note = payload["note"]

            aggregate_payload = {
                "topic": topic_dir.name,
                "qa_family": qa_family,
                "source_files": source_files,
                "source_rendered": source_rendered,
                "source_sidecars": source_sidecars,
                "source_conversations": source_conversations,
                "models": sorted(set(model_names)),
                "num_persona_files": len(source_files),
                "num_items": len(aggregate_items),
                "items": aggregate_items,
            }
            if status:
                aggregate_payload["status"] = status
            if note:
                aggregate_payload["note"] = note

            dump_json(family_dir / "all_personas.json", aggregate_payload)



def main() -> None:
    parser = argparse.ArgumentParser(description="Export rendered recall benchmarks into a separate test/ directory.")
    parser.add_argument("--source_dir", default="data/baseline")
    parser.add_argument("--dest_dir", default="data/test")
    args = parser.parse_args()

    source_root = Path(args.source_dir)
    dest_root = Path(args.dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    for rendered_path in iter_rendered_files(source_root):
        export_one(rendered_path, source_root, dest_root)

    build_aggregate_files(dest_root)


if __name__ == "__main__":
    main()
