"""HARNESS-first long-term memory: event store, procedure store, and retrieval.

Three memory tiers:
- Semantic memory: building facts, campus facts, curated findings (existing wiki + knowledge workbench)
- Event memory: every useful interaction trace (harness_events.jsonl)
- Procedure memory: reusable tool plans promoted from successful events (harness_procedures.jsonl)

Retrieval uses deterministic + lexical scoring (Phase 1).

Safety invariants:
- Procedure tool_plan stores **argument templates** with $variable placeholders, never concrete
  building names.  At search time, rebind_templates() fills placeholders from the *current* query
  entities so a procedure learned on "保健中心" is safe to suggest for "圖書館".
- Procedure de-duplication uses **plan_signature** (sorted tool name sequence) instead of coarse
  intent, preventing different tool flows from merging under "compare" / "strategy".
- auto_execute_threshold defaults to a very high value (effectively disabled).  The caller must
  explicitly lower it to enable automatic tool-plan reuse.  suggest-only is the safe default.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.project_paths import data_dir

_SCHEMA_VERSION = 2
_TOKEN_RE = re.compile(r"[A-Za-z0-9_一-鿿][A-Za-z0-9_\-一-鿿]*")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with", "by", "as", "at", "be",
    "this", "that", "are", "from", "but", "not", "into", "over", "under", "more", "less", "using", "use", "used",
    "you", "your", "we", "our", "they", "their",
    "於", "與", "和", "在", "是", "這", "那", "了", "也", "就", "都", "會", "以及", "或", "及", "為", "所", "由", "對",
}

SUGGEST_ONLY_THRESHOLD = 0.55
AUTO_EXECUTE_THRESHOLD = 0.95

BUILDING_ALIASES: dict[str, dict[str, str]] = {
    "保健中心": {"uid": "AT2045", "name": "保健中心"},
    "化學工程館": {"uid": "AT2007", "name": "化學工程館"},
    "土木研究大樓": {"uid": "AT5043", "name": "土木研究大樓"},
    "AT2045": {"uid": "AT2045", "name": "保健中心"},
    "AT2007": {"uid": "AT2007", "name": "化學工程館"},
    "AT5043": {"uid": "AT5043", "name": "土木研究大樓"},
}

_KNOWN_ARG_KEYS = {
    "building_name", "buildings", "building_uid", "campus",
    "meter_name", "name", "uid",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_event_id() -> str:
    ts = datetime.now().strftime("%Y%m%d")
    import uuid
    return f"evt_{ts}_{uuid.uuid4().hex[:6]}"


def _new_procedure_id(intent: str) -> str:
    import uuid
    return f"proc_{intent}_{uuid.uuid4().hex[:6]}"


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2 and t.lower() not in _STOPWORDS]


def _keyword_overlap(query_tokens: list[str], target_tokens: list[str]) -> float:
    if not query_tokens or not target_tokens:
        return 0.0
    q = Counter(query_tokens)
    t = Counter(target_tokens)
    overlap = sum(min(q[w], t.get(w, 0)) for w in q)
    return overlap / max(len(query_tokens), 1)


def _entity_overlap(query_entities: list[dict], target_entities: list[dict]) -> float:
    if not query_entities or not target_entities:
        return 0.0
    q_uids = {e.get("uid", "").upper() for e in query_entities if e.get("uid")}
    q_names = {e.get("name", "").lower() for e in query_entities if e.get("name")}
    t_uids = {e.get("uid", "").upper() for e in target_entities if e.get("uid")}
    t_names = {e.get("name", "").lower() for e in target_entities if e.get("name")}
    uid_match = len(q_uids & t_uids)
    name_match = len(q_names & t_names)
    total = max(len(q_uids | q_names), 1)
    return (uid_match + name_match) / total


def _intent_match(query_intent: str, target_intent: str) -> float:
    if not query_intent or not target_intent:
        return 0.0
    return 1.0 if query_intent == target_intent else 0.0


def _recency_score(ts: str, max_age_days: int = 90) -> float:
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return 0.0
    age = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    if age < 0:
        age = 0
    if age > max_age_days:
        return 0.0
    return 1.0 - (age / max_age_days)


def _success_quality(event: dict) -> float:
    if event.get("outcome") != "success":
        return 0.0
    quality = event.get("quality", {})
    score = quality.get("judge_score", 0.0)
    if isinstance(score, (int, float)):
        return min(float(score), 1.0)
    return 0.0


def _score_event(query_tokens: list[str], query_entities: list[dict], query_intent: str, event: dict) -> float:
    kw = _keyword_overlap(query_tokens, _tokenize(event.get("normalized_query", "") or event.get("user_query", "")))
    ent = _entity_overlap(query_entities, event.get("entities", []))
    intent = _intent_match(query_intent, event.get("intent", ""))
    sq = _success_quality(event)
    rec = _recency_score(event.get("ts", ""))
    return 0.35 * kw + 0.25 * ent + 0.20 * intent + 0.10 * sq + 0.10 * rec


def _score_procedure(query_tokens: list[str], query_intent: str, procedure: dict) -> float:
    trigger_text = " ".join(procedure.get("trigger_keywords", []))
    kw = _keyword_overlap(query_tokens, _tokenize(trigger_text))
    intent = _intent_match(query_intent, procedure.get("intent", ""))
    conf = float(procedure.get("confidence", 0.0))
    return 0.40 * kw + 0.30 * intent + 0.30 * conf


def _can_promote(event: dict) -> bool:
    quality = event.get("quality", {})
    return bool(
        quality.get("tool_correct")
        and quality.get("numbers_correct")
        and quality.get("answer_grounded")
        and float(quality.get("judge_score", 0)) >= 0.75
        and len(event.get("tool_trace", [])) >= 1
        and event.get("outcome") == "success"
    )


def _plan_signature(tool_plan: list[dict]) -> str:
    return "|".join(sorted(step.get("tool", "") for step in tool_plan))


def _templating_map(entities: list[dict]) -> dict[str, str]:
    m: dict[str, str] = {}
    for entity in entities:
        name = entity.get("name", "")
        uid = entity.get("uid", "")
        if name:
            m[name] = f"$entity_name_{len(m)}"
        if uid:
            m[uid] = f"$entity_uid_{len(m)}"
    return m


def _templatize_args(args: dict, tmap: dict[str, str]) -> dict:
    out: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and value in tmap:
            out[key] = tmap[value]
        elif isinstance(value, list):
            out[key] = [tmap.get(v, v) if isinstance(v, str) else v for v in value]
        else:
            out[key] = value
    return out


def _collect_entity_values(entities: list[dict]) -> set[str]:
    vals: set[str] = set()
    for e in entities:
        if e.get("name"):
            vals.add(e["name"])
        if e.get("uid"):
            vals.add(e["uid"])
    return vals


def rebind_templates(
    tool_plan: list[dict],
    query_entities: list[dict],
) -> list[dict]:
    if not query_entities:
        return [{"tool": step.get("tool", ""), "arguments": step.get("arguments_template", step.get("arguments", {})), "needs_rebind": True} for step in tool_plan]

    first_building: dict[str, str] | None = None
    for e in query_entities:
        if e.get("type") == "building":
            first_building = e
            break

    rebound: list[dict] = []
    for step in tool_plan:
        raw_args = step.get("arguments_template", step.get("arguments", {}))
        filled: dict[str, Any] = {}
        for key, value in raw_args.items():
            if isinstance(value, str) and value.startswith("$"):
                if first_building and key in _KNOWN_ARG_KEYS:
                    if "uid" in value or "uid" in key:
                        filled[key] = first_building.get("uid", value)
                    else:
                        filled[key] = first_building.get("name", value)
                else:
                    filled[key] = value
            elif isinstance(value, list):
                filled[key] = [first_building.get("name", v) if isinstance(v, str) and v.startswith("$") and first_building else v for v in value]
            else:
                filled[key] = value
        rebound.append({"tool": step.get("tool", ""), "arguments": filled, "needs_rebind": False})
    return rebound


class HarnessMemory:
    def __init__(self, state_dir: Path | str | None = None) -> None:
        if state_dir is None:
            state_dir = data_dir("knowledge_workbench", "state")
        self.state_dir = Path(state_dir)
        self.events_path = self.state_dir / "harness_events.jsonl"
        self.procedures_path = self.state_dir / "harness_procedures.jsonl"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_file(self.events_path)
        self._ensure_file(self.procedures_path)

    @staticmethod
    def _ensure_file(path: Path) -> None:
        if not path.exists():
            path.write_text("", encoding="utf-8")

    def _append_jsonl(self, path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _read_events(self) -> list[dict]:
        return self._read_jsonl(self.events_path)

    def _read_procedures(self) -> list[dict]:
        return self._read_jsonl(self.procedures_path)

    def extract_keywords(self, query: str) -> dict[str, Any]:
        tokens = _tokenize(query)
        entities: list[dict] = []
        seen_uids: set[str] = set()
        for alias, info in BUILDING_ALIASES.items():
            if alias in query:
                uid = info["uid"]
                if uid not in seen_uids:
                    entities.append({"type": "building", "name": info["name"], "uid": uid})
                    seen_uids.add(uid)

        intent_hints: list[str] = []
        if any(kw in query for kw in ("比較", "哪個", "哪棟", "哪一", "對比")):
            intent_hints.append("compare")
        if any(kw in query for kw in ("節能", "省電", "節電", "潛力")):
            intent_hints.append("savings")
        if any(kw in query for kw in ("異常", "突波", "不正常")):
            intent_hints.append("anomaly")
        if any(kw in query for kw in ("策略", "建議", "方案", "推薦")):
            intent_hints.append("strategy")
        if any(kw in query for kw in ("趨勢", "歷年", "變化")):
            intent_hints.append("trend")
        if any(kw in query for kw in ("預測", "模擬", "假設", "如果")):
            intent_hints.append("prediction")
        if any(kw in query for kw in ("ROI", "投資", "預算", "優先")):
            intent_hints.append("portfolio")
        if any(kw in query for kw in ("上次", "之前", "之前那", "剛才")):
            intent_hints.append("recall")

        intent_hint = intent_hints[0] if intent_hints else ""
        normalized = " ".join(tokens)
        return {
            "keywords": tokens,
            "entities": entities,
            "intent_hint": intent_hint,
            "normalized_query": normalized,
        }

    def append_event(self, event: dict) -> str:
        event_id = event.get("event_id") or _new_event_id()
        event.setdefault("schema_version", _SCHEMA_VERSION)
        event["event_id"] = event_id
        event.setdefault("ts", _now_iso())
        event.setdefault("outcome", "unknown")
        event.setdefault("quality", {})
        event.setdefault("tool_trace", [])
        event.setdefault("memory_hits", [])
        event.setdefault("training_tags", [])
        event.setdefault("promote_to_procedure", False)
        self._append_jsonl(self.events_path, event)
        return event_id

    def list_recent_events(self, limit: int = 20, outcome: str = "") -> list[dict]:
        events = self._read_events()
        if outcome:
            events = [e for e in events if e.get("outcome") == outcome]
        return events[-limit:]

    def search_events(self, query_tokens: list[str], query_entities: list[dict], query_intent: str, top_k: int = 3) -> list[dict]:
        events = self._read_events()
        scored = []
        for event in events:
            score = _score_event(query_tokens, query_entities, query_intent, event)
            if score >= SUGGEST_ONLY_THRESHOLD:
                scored.append({"event": event, "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search_procedures(self, query_tokens: list[str], query_intent: str, top_k: int = 3) -> list[dict]:
        procedures = self._read_procedures()
        scored = []
        for proc in procedures:
            score = _score_procedure(query_tokens, query_intent, proc)
            if score >= SUGGEST_ONLY_THRESHOLD:
                scored.append({"procedure": proc, "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search_memory(self, query: str, top_k: int = 3) -> dict[str, Any]:
        extracted = self.extract_keywords(query)
        tokens = extracted["keywords"]
        entities = extracted["entities"]
        intent = extracted["intent_hint"]

        hits: list[dict] = []

        event_hits = self.search_events(tokens, entities, intent, top_k)
        for hit in event_hits:
            event = hit["event"]
            rebound_plan = rebind_templates(event.get("selected_tool_plan", []), entities)
            action = "suggest_only"
            if hit["score"] >= AUTO_EXECUTE_THRESHOLD and event.get("outcome") == "success":
                action = "auto_execute"
            hits.append({
                "memory_type": "event",
                "id": event.get("event_id", ""),
                "score": hit["score"],
                "summary": event.get("final_answer_summary", event.get("user_query", ""))[:200],
                "suggested_tool_plan": rebound_plan,
                "action": action,
            })

        proc_hits = self.search_procedures(tokens, intent, top_k)
        for hit in proc_hits:
            proc = hit["procedure"]
            rebound_plan = rebind_templates(proc.get("tool_plan", []), entities)
            action = "suggest_only"
            if hit["score"] >= AUTO_EXECUTE_THRESHOLD and float(proc.get("confidence", 0)) >= 0.75:
                action = "auto_execute"
            hits.append({
                "memory_type": "procedure",
                "id": proc.get("procedure_id", ""),
                "score": hit["score"],
                "summary": f"Procedure for {proc.get('intent', '')}: " + ", ".join(
                    step.get("tool", "") for step in proc.get("tool_plan", [])
                ),
                "suggested_tool_plan": rebound_plan,
                "action": action,
            })

        hits.sort(key=lambda x: x["score"], reverse=True)
        return {"query": query, "hits": hits[:top_k]}

    def promote_to_procedure(self, event_id: str, procedure_hint: str = "") -> dict[str, Any]:
        events = self._read_events()
        target = None
        for event in events:
            if event.get("event_id") == event_id:
                target = event
                break
        if target is None:
            return {"status": "error", "error": f"Event not found: {event_id}"}
        if not _can_promote(target):
            return {"status": "rejected", "reason": "Event does not meet quality gates for promotion"}

        intent = target.get("intent", procedure_hint or "unknown")
        raw_plan = target.get("selected_tool_plan", [])
        sig = _plan_signature(raw_plan)

        tmap = _templating_map(target.get("entities", []))
        templatized_plan = []
        for step in raw_plan:
            args = step.get("arguments", {})
            templatized_plan.append({
                "tool": step.get("tool", ""),
                "arguments_template": _templatize_args(args, tmap),
            })

        existing_procs = self._read_procedures()
        existing = None
        for proc in existing_procs:
            if proc.get("plan_signature") == sig:
                existing = proc
                break

        if existing:
            existing["tool_plan"] = templatized_plan
            existing["trigger_keywords"] = list(set(
                existing.get("trigger_keywords", []) + target.get("keywords", [])
            ))
            existing["confidence"] = min(
                1.0,
                (float(existing.get("confidence", 0)) * existing.get("success_count", 0) + 0.8)
                / (existing.get("success_count", 0) + 1),
            )
            existing["supporting_event_ids"] = list(set(
                existing.get("supporting_event_ids", []) + [event_id]
            ))
            existing["success_count"] = existing.get("success_count", 0) + 1
            existing["updated_at"] = _now_iso()
            all_procs = [p for p in existing_procs if p.get("procedure_id") != existing["procedure_id"]]
            all_procs.append(existing)
            self._write_all_procedures(all_procs)
            return {"status": "ok", "procedure_id": existing["procedure_id"], "action": "updated"}
        else:
            procedure_id = _new_procedure_id(intent)
            new_proc = {
                "schema_version": _SCHEMA_VERSION,
                "procedure_id": procedure_id,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "intent": intent,
                "plan_signature": sig,
                "trigger_keywords": target.get("keywords", []),
                "required_entities": list({e.get("type", "building") for e in target.get("entities", [])}) or ["building"],
                "optional_entities": ["year", "strategy"],
                "tool_plan": templatized_plan,
                "confidence": 0.8,
                "supporting_event_ids": [event_id],
                "success_count": 1,
                "failure_count": 0,
            }
            self._append_jsonl(self.procedures_path, new_proc)
            return {"status": "ok", "procedure_id": procedure_id, "action": "created"}

    def _write_all_procedures(self, procedures: list[dict]) -> None:
        with self.procedures_path.open("w", encoding="utf-8") as f:
            for proc in procedures:
                f.write(json.dumps(proc, ensure_ascii=False, default=str) + "\n")

    def get_startup_context(self, campus: str = "ntu", limit: int = 8) -> dict[str, Any]:
        recent = self.list_recent_events(limit=limit, outcome="success")
        procedures = self._read_procedures()
        recent_procs = sorted(procedures, key=lambda p: p.get("updated_at", ""), reverse=True)[:limit]

        building_counter: Counter[str] = Counter()
        for event in recent:
            for entity in event.get("entities", []):
                name = entity.get("name", "")
                if name:
                    building_counter[name] += 1

        failure_modes: list[str] = []
        recent_failures = self.list_recent_events(limit=10, outcome="failure")
        for fail in recent_failures[-3:]:
            reason = fail.get("quality", {}).get("answer_grounded", "unknown")
            failure_modes.append(f"{fail.get('user_query', '')[:60]}: {reason}")

        summary_parts: list[str] = []
        if recent_procs:
            proc_names = [p.get("intent", "") for p in recent_procs[:3]]
            summary_parts.append(f"Known procedures: {', '.join(proc_names)}")
        if building_counter:
            top_buildings = building_counter.most_common(3)
            summary_parts.append(f"Frequent buildings: {', '.join(n for n, _ in top_buildings)}")
        summary_parts.append(f"Recent successful events: {len(recent)}")

        return {
            "recent_successful_procedures": [
                {
                    "procedure_id": p.get("procedure_id", ""),
                    "intent": p.get("intent", ""),
                    "tool_plan": p.get("tool_plan", []),
                    "confidence": p.get("confidence", 0),
                }
                for p in recent_procs
            ],
            "frequent_buildings": [
                {"name": name, "count": count}
                for name, count in building_counter.most_common(5)
            ],
            "known_failure_modes": failure_modes[-3:],
            "memory_summary": ". ".join(summary_parts),
        }

    def build_training_candidates(self) -> dict[str, int]:
        events = self._read_events()
        counts = {
            "router_sft_candidate": 0,
            "answer_sft_candidate": 0,
            "preference_pair_candidate": 0,
            "hard_negative_router": 0,
            "procedure_memory_candidate": 0,
        }
        new_events = []
        for event in events:
            tags = set(event.get("training_tags", []))
            quality = event.get("quality", {})
            outcome = event.get("outcome")

            if outcome == "success" and quality.get("tool_correct"):
                tags.add("router_sft_candidate")
            if (
                outcome == "success"
                and event.get("tool_trace")
                and quality.get("answer_grounded")
                and float(quality.get("judge_score", 0)) >= 0.8
            ):
                tags.add("answer_sft_candidate")
            if outcome == "failure":
                tags.add("hard_negative_router")
            if _can_promote(event):
                tags.add("procedure_memory_candidate")

            event["training_tags"] = sorted(tags)
            new_events.append(event)
            for tag in tags:
                if tag in counts:
                    counts[tag] += 1

        with self.events_path.open("w", encoding="utf-8") as f:
            for event in new_events:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

        return counts
