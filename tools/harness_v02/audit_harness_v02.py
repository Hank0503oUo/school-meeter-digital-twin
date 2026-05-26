"""
Audit harness_v02 against the SFT data-factory rules.

Run:
  cd D:\idf優化\demo
  python tools\harness_v02\audit_harness_v02.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent

TRAIN = BASE / "harness_v02_train.jsonl"
VAL = BASE / "harness_v02_val.jsonl"
SMOKE = BASE / "harness_v02_smoke.jsonl"
MANIFEST = BASE / "harness_v02_manifest.json"
AUDIT_JSON = BASE / "harness_v02_audit.json"
AUDIT_MD = BASE / "harness_v02_audit.md"

SOURCE_FILES = [
    BASE / "router_sft.jsonl",
    BASE / "safety_sft.jsonl",
    BASE / "explainer_sft.jsonl",
    BASE / "legacy_cleaned.jsonl",
]

REAL_ENTITY_PATTERNS = [
    "NTU",
    "台大",
    "保健中心",
    "化學工程館",
    "土木研究大樓",
    "土木大樓",
]

RAW_ID_PATTERN = re.compile(r"\b(?:point|meter|sensor|building|uid|id)[_\-]?\d{3,}\b", re.I)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            item["_path"] = str(path)
            rows.append(item)
    return rows


def message_content(sample: dict[str, Any], role: str) -> str:
    return "\n".join(
        m.get("content", "")
        for m in sample.get("messages", [])
        if m.get("role") == role
    )


def parse_assistant_json(sample: dict[str, Any]) -> dict[str, Any] | None:
    text = message_content(sample, "assistant").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "unknown")) for row in rows).most_common())


def audit() -> dict[str, Any]:
    train = load_jsonl(TRAIN)
    val = load_jsonl(VAL)
    smoke = load_jsonl(SMOKE)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        manifest = {}
    active_source_names = set((manifest.get("sources") or {}).keys())
    sources = {
        path.name: load_jsonl(path)
        for path in SOURCE_FILES
        if not active_source_names or path.name in active_source_names
    }
    all_rows = train + val

    issues: list[dict[str, Any]] = []

    def add_issue(severity: str, title: str, detail: str) -> None:
        issues.append({"severity": severity, "title": title, "detail": detail})

    if not MANIFEST.exists():
        add_issue("fail", "missing_manifest", "harness_v02_manifest.json is missing.")

    for name, rows in {"train": train, "val": val}.items():
        if not rows:
            add_issue("fail", f"empty_{name}", f"{name} split has no records.")
        for row in rows:
            messages = row.get("messages", [])
            roles = [m.get("role") for m in messages]
            if roles[:2] != ["system", "user"] or "assistant" not in roles:
                add_issue("fail", "bad_message_roles", f"{name}:{row.get('_line_no')} roles={roles}")
            parsed = parse_assistant_json(row)
            if parsed is None:
                add_issue("fail", "assistant_not_json", f"{name}:{row.get('_line_no')} assistant target is not JSON.")
            elif parsed.get("tool") != row.get("expected_tool"):
                add_issue(
                    "fail",
                    "expected_tool_mismatch",
                    f"{name}:{row.get('_line_no')} expected_tool={row.get('expected_tool')} assistant_tool={parsed.get('tool')}",
                )

    train_prompts = [message_content(row, "user") for row in train]
    val_prompts = [message_content(row, "user") for row in val]
    train_val_overlap = sorted(set(train_prompts) & set(val_prompts))
    if train_val_overlap:
        add_issue("fail", "train_val_prompt_overlap", f"{len(train_val_overlap)} user prompts appear in both splits.")

    duplicate_train = len(train_prompts) - len(set(train_prompts))
    if duplicate_train:
        add_issue("warn", "duplicate_train_prompts", f"{duplicate_train} duplicate user prompts in train split.")

    val_categories = set(distribution(val, "category"))
    source_categories = set(distribution([row for rows in sources.values() for row in rows], "category"))
    missing_val_categories = sorted(source_categories - val_categories)
    if missing_val_categories:
        add_issue("warn", "missing_val_categories", f"Validation split misses categories: {missing_val_categories}")

    full_text = "\n".join(
        "\n".join(m.get("content", "") for m in row.get("messages", []))
        for row in all_rows
    )
    entity_hits = [p for p in REAL_ENTITY_PATTERNS if p in full_text]
    if entity_hits:
        add_issue(
            "warn",
            "real_entity_names_present",
            "Real campus/building names are present. This is OK for a closed demo, but not for a reusable public SFT set: "
            + ", ".join(entity_hits),
        )

    raw_id_hits = RAW_ID_PATTERN.findall(full_text)
    if raw_id_hits:
        add_issue("warn", "raw_ids_present", f"Potential raw IDs found: {raw_id_hits[:10]}")

    if not smoke:
        add_issue("warn", "missing_smoke_split", "No harness_v02_smoke.jsonl found.")

    if not any(path.name.endswith("rejected.jsonl") for path in SOURCE_FILES):
        add_issue(
            "warn",
            "no_rejected_variants",
            "harness_v02 has no rejected variants/preference pairs; keep it as SFT router data, not DPO data.",
        )

    gates = manifest.get("eval_gates", {})
    if "tool_accuracy" not in gates or "malformed_json" not in gates:
        add_issue("warn", "incomplete_eval_gates", "Manifest should define tool_accuracy and malformed_json gates.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "train": len(train),
            "val": len(val),
            "smoke": len(smoke),
            "sources": {name: len(rows) for name, rows in sources.items()},
        },
        "distribution": {
            "train_category": distribution(train, "category"),
            "val_category": distribution(val, "category"),
            "train_difficulty": distribution(train, "difficulty"),
            "val_difficulty": distribution(val, "difficulty"),
            "train_tool_top": dict(list(distribution(train, "expected_tool").items())[:15]),
            "val_tool_top": dict(list(distribution(val, "expected_tool").items())[:15]),
        },
        "factory_alignment": {
            "manifest": MANIFEST.exists(),
            "deterministic_split": "split_method" in manifest,
            "smoke_split": bool(smoke),
            "risk_refusal_samples": any(row.get("category") == "safety" for row in all_rows),
            "rejected_variants": False,
            "preference_pairs": False,
            "downstream_validation_report": False,
        },
        "issues": issues,
        "assessment": "PASS_WITH_WARNINGS" if not any(i["severity"] == "fail" for i in issues) else "FAIL",
    }
    return report


def write_report(report: dict[str, Any]) -> None:
    AUDIT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# harness_v02 SFT Data-Factory Audit",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- assessment: `{report['assessment']}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in report["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Factory Alignment", ""]
    for key, value in report["factory_alignment"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Issues", ""]
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- `{issue['severity']}` `{issue['title']}`: {issue['detail']}")
    else:
        lines.append("- No issues found.")
    lines += ["", "## Distributions", "", "```json", json.dumps(report["distribution"], ensure_ascii=False, indent=2), "```", ""]
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = audit()
    write_report(result)
    print(json.dumps({"assessment": result["assessment"], "issues": result["issues"]}, ensure_ascii=False, indent=2))
