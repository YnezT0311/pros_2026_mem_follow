import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Set


EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+1[-.\s]?)?(?:\d{3}[-.\s]?\d{4}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b")
ID_RE = re.compile(r"\b(?:ID\s*[:#]?\s*)?([A-Z]{1,3}-\d{3,8})\b", re.IGNORECASE)
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9 .'-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln)\b",
    re.IGNORECASE,
)
PERSONA_RE = re.compile(r"_persona(\d+)_")


def iter_conversation_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, fnames in os.walk(root_dir):
        for fname in fnames:
            if fname.startswith("conversation_") and fname.endswith(".json"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def iter_text_values(obj) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_text_values(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_text_values(v)


def extract_pii_values(text: str) -> Dict[str, Set[str]]:
    out = {
        "email": set(EMAIL_RE.findall(text)),
        "phone": set(m.strip() for m in PHONE_RE.findall(text)),
        "id": set(m.upper() for m in ID_RE.findall(text)),
        "address": set(m.strip() for m in ADDRESS_RE.findall(text)),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Validate persona-level synthetic PII consistency.")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        default=["data/output", "data/retention/world"],
        help="Directories to scan for conversation_*.json files.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="data/retention/pii_consistency_report.json",
        help="Output report path.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit code 1 when conflicts are found.")
    args = parser.parse_args()

    persona_declared = {}  # persona -> {"email","phone","id","address"}
    persona_declared_conflicts = []
    observed = defaultdict(lambda: defaultdict(set))  # persona -> type -> values
    seen_in_files = defaultdict(lambda: defaultdict(list))  # persona -> type -> [(value,file)]

    for d in args.input_dirs:
        if not os.path.exists(d):
            continue
        for path in iter_conversation_files(d):
            m = PERSONA_RE.search(os.path.basename(path))
            if not m:
                continue
            persona = m.group(1)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            declared = data.get("Persona PII")
            if isinstance(declared, dict):
                norm_declared = {
                    "email": str(declared.get("email", "")).strip(),
                    "phone": str(declared.get("phone", "")).strip(),
                    "id": str(declared.get("id", "")).strip().upper(),
                    "address": str(declared.get("address", "")).strip(),
                }
                prev = persona_declared.get(persona)
                if prev is None:
                    persona_declared[persona] = norm_declared
                elif prev != norm_declared:
                    persona_declared_conflicts.append({
                        "persona_id": int(persona),
                        "issue": "inconsistent_persona_pii_block",
                        "first": prev,
                        "second": norm_declared,
                        "file": path,
                    })
            for text in iter_text_values(data):
                vals = extract_pii_values(text)
                for k, vset in vals.items():
                    for v in vset:
                        observed[persona][k].add(v)
                        if len(seen_in_files[persona][k]) < 20:
                            seen_in_files[persona][k].append((v, path))

    conflicts = []
    conflicts.extend(persona_declared_conflicts)
    for persona, type_map in observed.items():
        expected = persona_declared.get(persona, {})
        for pii_type in ("email", "phone", "id", "address"):
            values = sorted(type_map.get(pii_type, set()))
            if len(values) > 1:
                conflicts.append({
                    "persona_id": int(persona),
                    "pii_type": pii_type,
                    "issue": "multiple_values",
                    "values": values,
                    "examples": seen_in_files[persona][pii_type][:5],
                })
            exp = expected.get(pii_type)
            if exp:
                normalized_expected = exp.upper() if pii_type == "id" else exp
                mismatched = [v for v in values if v != normalized_expected]
                if mismatched:
                    conflicts.append({
                        "persona_id": int(persona),
                        "pii_type": pii_type,
                        "issue": "mismatch_with_persona_pii",
                        "expected_source": "persona_pii_field",
                        "expected": exp,
                        "observed": values,
                        "examples": seen_in_files[persona][pii_type][:5],
                    })

    report = {
        "input_dirs": args.input_dirs,
        "num_personas_scanned": len(observed),
        "num_conflicts": len(conflicts),
        "conflicts": conflicts,
    }
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("num_personas_scanned", "num_conflicts")}, indent=2))
    print(f"Wrote report: {args.report_path}")

    if args.strict and conflicts:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
