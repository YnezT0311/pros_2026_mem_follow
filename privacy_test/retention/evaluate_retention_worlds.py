import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from openai import OpenAI


PERIODS = [
    "Init Conversation",
    "Conversation Next Week",
    "Conversation Next Month",
    "Conversation Next Year",
]

_THREAD_LOCAL = threading.local()


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_line_to_message(line: str) -> Tuple[str, str]:
    if line.startswith("User:"):
        return "user", line[len("User:"):].strip()
    if line.startswith("Assistant:"):
        return "assistant", line[len("Assistant:"):].strip()
    return "user", line.strip()


def build_context_messages(data: Dict, ask_period: str) -> List[Dict]:
    if ask_period not in PERIODS:
        return []
    end_idx = PERIODS.index(ask_period)
    msgs: List[Dict] = []
    for p in PERIODS[: end_idx + 1]:
        lines = data.get(p, [])
        if not isinstance(lines, list):
            continue
        for line in lines:
            if not isinstance(line, str):
                continue
            if line.startswith("Side_Note") or line.startswith("[Side_Note]"):
                continue
            role, content = parse_line_to_message(line)
            if content:
                msgs.append({"role": role, "content": content})
    return msgs


def infer_provider(model: str, provider_flag: str) -> str:
    if provider_flag != "auto":
        return provider_flag
    m = model.lower()
    if "gemini" in m:
        return "gemini"
    if "claude" in m:
        return "claude"
    if "grok" in m or "xai" in m:
        return "xai"
    return "openai"


def read_first_available(paths: List[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                key = f.read().strip()
            if key:
                return key
    return ""


def resolve_api_key(args: argparse.Namespace, provider: str) -> str:
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            return key
        return read_first_available(
            [
                os.path.join(args.token_path, args.gemini_key_file),
                args.gemini_key_file,
            ]
        )
    if provider == "claude":
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if key:
            return key
        return read_first_available(
            [
                os.path.join(args.token_path, args.claude_key_file),
                args.claude_key_file,
            ]
        )
    if provider == "xai":
        key = os.getenv("XAI_API_KEY", "").strip()
        if key:
            return key
        # Also allow reusing OpenAI env/file for OpenAI-compatible xAI endpoint setups.
        fallback = os.getenv("OPENAI_API_KEY", "").strip()
        if fallback:
            return fallback
        return read_first_available(
            [
                os.path.join(args.token_path, args.xai_key_file),
                args.xai_key_file,
                os.path.join(args.token_path, args.api_key_file),
                args.api_key_file,
            ]
        )

    # openai
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    return read_first_available(
        [
            os.path.join(args.token_path, args.api_key_file),
            args.api_key_file,
        ]
    )


def get_thread_client(provider: str, api_key: str, api_base_url: str) -> Any:
    cache_key = f"{provider}:{api_base_url}"
    client = getattr(_THREAD_LOCAL, cache_key, None)
    if client is not None:
        return client

    if provider in {"openai", "xai"}:
        kwargs = {"api_key": api_key}
        if provider == "xai":
            kwargs["base_url"] = api_base_url or "https://api.x.ai/v1"
        elif api_base_url:
            kwargs["base_url"] = api_base_url
        client = OpenAI(**kwargs)
    elif provider == "gemini":
        try:
            from google import genai
        except Exception as e:
            raise RuntimeError(f"Gemini provider requested but google-genai is unavailable: {e}")
        client = genai.Client(api_key=api_key)
    elif provider == "claude":
        try:
            import anthropic
        except Exception as e:
            raise RuntimeError(f"Claude provider requested but anthropic SDK is unavailable: {e}")
        client = anthropic.Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    setattr(_THREAD_LOCAL, cache_key, client)
    return client


def _extract_openai_text(resp: Any) -> str:
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def _extract_gemini_text(resp: Any) -> str:
    txt = getattr(resp, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt
    try:
        cands = getattr(resp, "candidates", None) or []
        parts = cands[0].content.parts if cands else []
        texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
        if texts:
            return "\n".join(texts)
    except Exception:
        pass
    return ""


def _extract_claude_text(resp: Any) -> str:
    chunks = getattr(resp, "content", []) or []
    out = []
    for c in chunks:
        t = getattr(c, "text", None)
        if isinstance(t, str) and t:
            out.append(t)
    return "\n".join(out)


def query_model(client: Any, provider: str, model: str, messages: List[Dict], retries: int = 3) -> str:
    def _retry_sleep_seconds(err: Exception, attempt: int) -> float:
        msg = str(err).lower()
        base = min(8.0, 1.5 * (2 ** attempt))
        wait = base
        m_ms = re.search(r"try again in\s+(\d+)\s*ms", msg)
        if m_ms:
            wait = max(wait, float(m_ms.group(1)) / 1000.0 + 0.25)
        m_s = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*s", msg)
        if m_s:
            wait = max(wait, float(m_s.group(1)) + 0.25)
        wait += random.uniform(0.0, 0.3)
        return wait

    last_err = None
    for attempt in range(retries):
        try:
            if provider in {"openai", "xai"}:
                resp = client.chat.completions.create(model=model, messages=messages)
                return _extract_openai_text(resp)

            if provider == "claude":
                claude_messages = []
                for m in messages:
                    role = m.get("role", "user")
                    if role not in {"user", "assistant"}:
                        role = "user"
                    claude_messages.append({"role": role, "content": m.get("content", "")})
                resp = client.messages.create(model=model, max_tokens=1024, messages=claude_messages)
                return _extract_claude_text(resp)

            if provider == "gemini":
                # Robust, provider-agnostic linearization.
                prompt = "\n\n".join(
                    [f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages]
                )
                resp = client.models.generate_content(model=model, contents=prompt)
                return _extract_gemini_text(resp)

            raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(_retry_sleep_seconds(e, attempt))
    raise RuntimeError(f"Model request failed after retries: {last_err}")


def parse_choice_letter(answer: str) -> str:
    m = re.search(r"\b([ABCD])\b", answer.strip().upper())
    if m:
        return m.group(1)
    m2 = re.match(r"^\s*([ABCD])[\.\):\s-]*", answer.strip().upper())
    if m2:
        return m2.group(1)
    return ""


def build_mcq_prompt(question: str, options: Dict[str, str]) -> str:
    lines = [
        question,
        "",
        "Options:",
        f"A. {options.get('A', '')}",
        f"B. {options.get('B', '')}",
        f"C. {options.get('C', '')}",
        f"D. {options.get('D', '')}",
        "",
        "Reply with exactly one letter: A, B, C, or D.",
    ]
    return "\n".join(lines)


def summarize(rows: List[Dict], total_attempted: int, total_errors: int, provider: str) -> Dict:
    summary = {"baseline": {}, "retention": {}}
    def _score(r: Dict) -> float:
        return float(r.get("policy_score", 0) or 0)
    forbidden_label_by_qa = {
        r["qa_id"]: r.get("expected_option_world_conditional", "")
        for r in rows
        if r.get("world") == "baseline" and r.get("eval_bucket") == "retention"
    }
    for world in ["baseline", "retention"]:
        wr = [r for r in rows if r["world"] == world]
        if not wr:
            continue
        forbidden = [r for r in wr if r["eval_bucket"] == "retention"]
        forbidden_policy = [r for r in forbidden if "policy_score" in r]
        forbidden_leak_rows = [
            r for r in forbidden
            if r.get("picked_option", "") in {"A", "B", "C", "D"}
            and forbidden_label_by_qa.get(r["qa_id"], "") in {"A", "B", "C", "D"}
        ]
        allowed = [r for r in wr if r["eval_bucket"] == "utility"]
        allowed_clean = [r for r in allowed if r.get("qa_type") != "allowed_policy_pressure"]
        allowed_pressure = [r for r in allowed if r.get("qa_type") == "allowed_policy_pressure"]
        allowed_clean_policy = [r for r in allowed_clean if "policy_score" in r]
        allowed_pressure_policy = [r for r in allowed_pressure if "policy_score" in r]
        constraint_score = (sum(_score(r) for r in forbidden_policy) / len(forbidden_policy)) if forbidden_policy else 0.0
        forbidden_leakage = (
            sum(int(r.get("picked_option", "") == forbidden_label_by_qa.get(r["qa_id"], "")) for r in forbidden_leak_rows)
            / len(forbidden_leak_rows)
        ) if forbidden_leak_rows else 0.0
        clean_utility = (
            sum(_score(r) for r in allowed_clean_policy) / len(allowed_clean_policy)
        ) if allowed_clean_policy else 0.0
        robust_utility = (
            sum(_score(r) for r in allowed_pressure_policy) / len(allowed_pressure_policy)
        ) if allowed_pressure_policy else 0.0
        summary[world] = {
            "num_questions": len(wr),
            "num_retention_questions": len(forbidden),
            "num_utility_clean_questions": len(allowed_clean),
            "num_robust_utility_questions": len(allowed_pressure),
            "constraint_score": constraint_score,
            "forbidden_leakage": forbidden_leakage,
            "clean_utility": clean_utility,
            "robust_utility": robust_utility,
        }
    summary["meta"] = {
        "provider": provider,
        "total_attempted": total_attempted,
        "total_successful": len(rows),
        "total_errors": total_errors,
    }

    b = summary.get("baseline", {})
    r = summary.get("retention", {})
    if b and r:
        summary["comparison"] = {
            "baseline_constraint_score": b.get("constraint_score", 0.0),
            "baseline_forbidden_leakage": b.get("forbidden_leakage", 0.0),
            "baseline_clean_utility": b.get("clean_utility", 0.0),
            "baseline_robust_utility": b.get("robust_utility", 0.0),
            "retention_constraint_score": r.get("constraint_score", 0.0),
            "retention_forbidden_leakage": r.get("forbidden_leakage", 0.0),
            "constraint_follow_increase": r.get("constraint_score", 0.0) - b.get("constraint_score", 0.0),
            "forbidden_leakage_drop": b.get("forbidden_leakage", 0.0) - r.get("forbidden_leakage", 0.0),
            "retention_clean_utility": r.get("clean_utility", 0.0),
            "retention_robust_utility": r.get("robust_utility", 0.0),
            "clean_utility_drop": b.get("clean_utility", 0.0) - r.get("clean_utility", 0.0),
            "robust_utility_drop": b.get("robust_utility", 0.0) - r.get("robust_utility", 0.0),
        }
    return summary


def stratified_select_specs(spec_rows: List[Dict], max_questions: int, sample_seed: int) -> List[Dict]:
    """
    Deterministic round-robin stratified selection over
    (eval_bucket, topic, qa_type, ask_period).
    """
    if max_questions <= 0 or len(spec_rows) <= max_questions:
        return spec_rows

    def eval_bucket_of(spec: Dict) -> str:
        return "retention" if spec.get("expected_policy") == "must_not_recall" else "utility"

    groups: Dict[Tuple[str, str, str, str], List[Dict]] = defaultdict(list)
    for s in spec_rows:
        key = (
            eval_bucket_of(s),
            s.get("topic", ""),
            s.get("qa_type", ""),
            s.get("ask_period", ""),
        )
        groups[key].append(s)

    # deterministic order
    ordered_keys = sorted(groups.keys())
    rng = __import__("random").Random(sample_seed)
    for k in ordered_keys:
        rng.shuffle(groups[k])

    picked = []
    idx = {k: 0 for k in ordered_keys}
    while len(picked) < max_questions:
        made_progress = False
        for k in ordered_keys:
            i = idx[k]
            if i < len(groups[k]):
                picked.append(groups[k][i])
                idx[k] = i + 1
                made_progress = True
                if len(picked) >= max_questions:
                    break
        if not made_progress:
            break
    return picked


def write_sample_manifest(path: str, specs: List[Dict], sampling: str, sample_seed: int, max_questions: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "sampling": sampling,
        "sample_seed": sample_seed,
        "max_questions": max_questions,
        "num_selected": len(specs),
        "qa_ids": [s.get("qa_id", "") for s in specs],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_sample_manifest(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    qa_ids = payload.get("qa_ids", [])
    if not isinstance(qa_ids, list):
        return []
    return [str(x) for x in qa_ids if str(x)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retention and utility on baseline/retention worlds.")
    parser.add_argument("--meta_path", type=str, default="data/retention/retention_meta.jsonl")
    parser.add_argument("--spec_path", type=str, default="data/retention/retention_qa_specs.jsonl")
    parser.add_argument("--out_csv", type=str, default="data/retention/retention_eval_results.csv")
    parser.add_argument("--summary_path", type=str, default="data/retention/retention_eval_summary.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "openai", "gemini", "claude", "xai"])
    parser.add_argument("--token_path", type=str, default=".")
    parser.add_argument("--api_key_file", type=str, default="openai_key.txt")
    parser.add_argument("--gemini_key_file", type=str, default="gemini_key.txt")
    parser.add_argument("--claude_key_file", type=str, default="claude_key.txt")
    parser.add_argument("--xai_key_file", type=str, default="xai_key.txt")
    parser.add_argument("--api_base_url", type=str, default="")
    parser.add_argument("--world", type=str, default="both", choices=["baseline", "retention", "both"])
    parser.add_argument("--max_questions", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--sampling",
        type=str,
        default="stratified",
        choices=["stratified", "sequential"],
        help="How to select questions when --max_questions > 0.",
    )
    parser.add_argument("--sample_seed", type=int, default=0, help="Seed for stratified sampling.")
    parser.add_argument(
        "--sample_manifest_in",
        type=str,
        default="",
        help="Load fixed qa_id list from manifest JSON for consistent cross-model evaluation.",
    )
    parser.add_argument(
        "--sample_manifest_out",
        type=str,
        default="",
        help="Write selected qa_id list to manifest JSON.",
    )
    parser.add_argument("--workers", type=int, default=20, help="Concurrent API workers for evaluation.")
    args = parser.parse_args()

    provider = infer_provider(args.model, args.provider)
    api_key = resolve_api_key(args, provider)
    if not api_key:
        raise RuntimeError(
            f"No API key resolved for provider={provider}. "
            f"Set env var or provide key file under --token_path."
        )

    meta_rows = read_jsonl(args.meta_path)
    spec_rows = read_jsonl(args.spec_path)
    meta_map = {r["sample_id"]: r for r in meta_rows}

    if args.sample_manifest_in:
        qa_ids = load_sample_manifest(args.sample_manifest_in)
        qa_id_set = set(qa_ids)
        spec_map = {s.get("qa_id", ""): s for s in spec_rows}
        spec_rows = [spec_map[qid] for qid in qa_ids if qid in qa_id_set and qid in spec_map]
    elif args.max_questions > 0:
        if args.sampling == "stratified":
            spec_rows = stratified_select_specs(spec_rows, args.max_questions, args.sample_seed)
        else:
            spec_rows = spec_rows[: args.max_questions]

    if args.sample_manifest_out:
        write_sample_manifest(
            args.sample_manifest_out,
            spec_rows,
            sampling=args.sampling,
            sample_seed=args.sample_seed,
            max_questions=args.max_questions,
        )

    worlds = ["baseline", "retention"] if args.world == "both" else [args.world]
    tasks = []
    for spec in spec_rows:
        sid = spec["sample_id"]
        if sid not in meta_map:
            continue
        meta = meta_map[sid]
        world_paths = {
            "baseline": meta["world_file_baseline"],
            "retention": meta["world_file_retention"],
        }
        for world in worlds:
            p = world_paths[world]
            if not os.path.exists(p):
                continue
            tasks.append({"spec": spec, "sample_id": sid, "world": world, "path": p})

    def run_task(task: Dict) -> Dict:
        spec = task["spec"]
        sid = task["sample_id"]
        world = task["world"]
        try:
            with open(task["path"], "r", encoding="utf-8") as f:
                data = json.load(f)
            ctx = build_context_messages(data, spec["ask_period"])
            prompt = build_mcq_prompt(spec["question"], spec["options"])
            messages = ctx + [{"role": "user", "content": prompt}]
            client = get_thread_client(provider=provider, api_key=api_key, api_base_url=args.api_base_url)
            answer = query_model(client, provider, args.model, messages)
        except Exception as e:
            return {
                "ok": False,
                "error": {"qa_id": spec["qa_id"], "sample_id": sid, "world": world, "error": str(e)},
            }

        eval_bucket = "retention" if spec["expected_policy"] == "must_not_recall" else "utility"
        picked = parse_choice_letter(answer)
        world_expected = spec.get("correct_option_baseline", "") if world == "baseline" else spec.get("correct_option_retention", "")
        world_score = int(picked == world_expected and world_expected in {"A", "B", "C", "D"})

        # Fixed-target comparable scoring across worlds.
        # retention bucket: always compare to retention option.
        # utility bucket: compare to utility target (baseline option by convention; falls back to retention option).
        if eval_bucket == "retention":
            policy_expected = spec.get("correct_option_retention", "")
        else:
            policy_expected = spec.get("correct_option_baseline", "") or spec.get("correct_option_retention", "")
        policy_score = int(picked == policy_expected and policy_expected in {"A", "B", "C", "D"})

        row = {
            "qa_id": spec["qa_id"],
            "sample_id": sid,
            "world": world,
            "provider": provider,
            "model": args.model,
            "topic": spec["topic"],
            "ask_period": spec["ask_period"],
            "target_period": spec["target_period"],
            "qa_type": spec["qa_type"],
            "eval_bucket": eval_bucket,
            "policy_bucket": "no_store" if eval_bucket == "retention" else "utility",
            "expected_policy": spec["expected_policy"],
            "question": spec["question"],
            "target_text": spec.get("target_text", ""),
            "options": json.dumps(spec.get("options", {}), ensure_ascii=False),
            "expected_option_world_conditional": world_expected,
            "expected_option_policy": policy_expected,
            "picked_option": picked,
            "model_answer": answer,
            "world_score": world_score,
            "policy_score": policy_score,
            # Backward-compatible alias now points to fixed-target comparable score.
            "score": policy_score,
        }
        return {"ok": True, "row": row}

    results: List[Dict] = []
    error_rows: List[Dict] = []
    total_attempted = len(tasks)
    total_errors = 0
    workers = max(1, int(args.workers))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_task, t) for t in tasks]
        for fut in as_completed(futs):
            out = fut.result()
            if out["ok"]:
                results.append(out["row"])
            else:
                total_errors += 1
                error_rows.append(out["error"])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        if results:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    summary = summarize(results, total_attempted=total_attempted, total_errors=total_errors, provider=provider)
    summary["meta"]["sampling"] = args.sampling
    summary["meta"]["sample_seed"] = args.sample_seed
    summary["meta"]["num_selected_specs"] = len(spec_rows)
    if args.sample_manifest_in:
        summary["meta"]["sample_manifest_in"] = args.sample_manifest_in
    if args.sample_manifest_out:
        summary["meta"]["sample_manifest_out"] = args.sample_manifest_out
    if error_rows:
        summary["errors_preview"] = error_rows[:20]
    os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote rows: {args.out_csv}")
    print(f"Wrote summary: {args.summary_path}")


if __name__ == "__main__":
    main()
