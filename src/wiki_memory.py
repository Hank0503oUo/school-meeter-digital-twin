"""Wiki + knowledge-graph memory for the energy assistant.

Mirrors the multi-ai-collab `memory.js` module so the energy assistant has the
same shared-memory primitives (incrementally maintained markdown wiki + a
graphify-style knowledge graph) writable by the LLM via its MCP tools or by
``lm_studio_client.chat_with_mcp`` directly.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

PAGE_KINDS: tuple[str, ...] = ("source", "entity", "concept", "session")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_一-鿿][A-Za-z0-9_\-一-鿿]*")
_WIKILINK_RE = re.compile(r"\[\[([^\]\n|]+?)(?:\|[^\]\n]+?)?\]\]")
_FRONTMATTER_FIELD_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\-]*):\s*(.*)$")
_STOPWORDS = {
    "the","a","an","and","or","of","to","in","is","it","for","on","with","by","as","at","be","this","that","are",
    "from","but","not","into","over","under","more","less","using","use","used","you","your","we","our","they","their",
    "於","與","和","在","是","這","那","了","也","就","都","會","以及","或","及","為","所","由","對","對於",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    text = re.sub(r"[\s\\/]+", "-", str(value or "").strip().lower())
    text = re.sub(r"[^0-9a-z\-_一-鿿]", "", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "page"


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2 and t.lower() not in _STOPWORDS]


def _read_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _strip_frontmatter(body: str) -> str:
    lines = body.splitlines()
    if not lines or lines[0].strip() != "---":
        return body
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1:])
    return body


class WikiMemory:
    """Persistent wiki + knowledge graph that the energy assistant maintains."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            env_root = os.getenv("ENERGY_WIKI_ROOT")
            if env_root:
                root = Path(env_root)
            else:
                demo_root = Path(__file__).resolve().parent.parent
                root = demo_root / "data" / "knowledge_workbench" / "memory"
        self.root = Path(root)
        self.wiki_root = self.root / "wiki"
        self.graph_root = self.root / "graph"
        self.index_path = self.wiki_root / "index.md"
        self.log_path = self.wiki_root / "log.md"
        self.graph_json_path = self.graph_root / "graph.json"
        self.graph_report_path = self.graph_root / "GRAPH_REPORT.md"
        self._ensure_dirs()

    @staticmethod
    def _plural_kind(kind: str) -> str:
        return "entities" if kind == "entity" else f"{kind}s"

    def _ensure_dirs(self) -> None:
        for kind in PAGE_KINDS:
            (self.wiki_root / self._plural_kind(kind)).mkdir(parents=True, exist_ok=True)
        self.graph_root.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(
                "# Wiki Index\n\n_Maintained by the energy assistant. Each entry: page link + one-line summary._\n",
                encoding="utf-8",
            )
        if not self.log_path.exists():
            self.log_path.write_text(
                "# Wiki Log\n\n_Chronological record of ingest / query / build events._\n",
                encoding="utf-8",
            )

    def _kind_dir(self, kind: str) -> Path:
        if kind not in PAGE_KINDS:
            raise ValueError(f"kind must be one of {PAGE_KINDS}, got {kind!r}")
        return self.wiki_root / self._plural_kind(kind)

    def _rel_for(self, kind: str, slug: str) -> str:
        return f"{self._plural_kind(kind)}/{slug}.md"

    def _list_kind(self, kind: str) -> list[dict[str, Any]]:
        directory = self._kind_dir(kind)
        if not directory.exists():
            return []
        pages = []
        for path in sorted(directory.glob("*.md")):
            slug = path.stem
            pages.append({"kind": kind, "slug": slug, "path": path, "rel": self._rel_for(kind, slug)})
        return pages

    def _list_pages(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            return self._list_kind(kind)
        out: list[dict[str, Any]] = []
        for k in PAGE_KINDS:
            out.extend(self._list_kind(k))
        return out

    def _append_log(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## [{_now_iso()}] {line}\n")

    def _read_frontmatter_field(self, path: Path, field: str) -> str:
        body = _read_safe(path)
        if not body.startswith("---"):
            return ""
        for raw in body.splitlines()[1:]:
            if raw.strip() == "---":
                break
            match = _FRONTMATTER_FIELD_RE.match(raw)
            if match and match.group(1) == field:
                return match.group(2).strip()
        return ""

    def extract_summary(self, body: str) -> str:
        in_frontmatter = False
        seen_fence = False
        for raw in body.splitlines():
            line = raw.strip()
            if line == "---":
                if not seen_fence:
                    in_frontmatter = True
                    seen_fence = True
                    continue
                if in_frontmatter:
                    in_frontmatter = False
                    continue
                continue
            if in_frontmatter:
                continue
            if not line or line.startswith("#"):
                continue
            return line if len(line) <= 140 else line[:137] + "..."
        return "(no summary)"

    def rebuild_index(self) -> Path:
        lines = [
            "# Wiki Index",
            "",
            "_Maintained by the energy assistant. Each entry: page link + one-line summary._",
            "",
        ]
        for kind in PAGE_KINDS:
            pages = self._list_kind(kind)
            if not pages:
                continue
            lines.append(f"## {kind}s ({len(pages)})")
            lines.append("")
            for page in pages:
                summary = self.extract_summary(_read_safe(page["path"]))
                lines.append(f"- [{page['slug']}]({page['rel']}) — {summary}")
            lines.append("")
        self.index_path.write_text("\n".join(lines), encoding="utf-8")
        return self.index_path

    def ingest(
        self,
        *,
        title: str,
        content: str,
        kind: str = "source",
        tags: Sequence[str] | str = (),
        links: Sequence[str] = (),
    ) -> dict[str, Any]:
        title = str(title or "").strip()
        if not title:
            raise ValueError("title is required")
        if not str(content or "").strip():
            raise ValueError("content is required")
        if kind not in PAGE_KINDS:
            raise ValueError(f"kind must be one of {PAGE_KINDS}")

        slug = _slugify(title)
        path = self._kind_dir(kind) / f"{slug}.md"
        existed = path.exists()
        tag_list = [t.strip() for t in (tags.split(",") if isinstance(tags, str) else list(tags)) if str(t).strip()]
        link_list = [str(link).strip() for link in (links or []) if str(link).strip()]

        created = self._read_frontmatter_field(path, "created") if existed else _now_iso()
        if not created:
            created = _now_iso()

        frontmatter = "\n".join([
            "---",
            f"title: {title}",
            f"kind: {kind}",
            f"slug: {slug}",
            f"created: {created}",
            f"updated: {_now_iso()}",
            "tags: [" + ", ".join(json.dumps(t, ensure_ascii=False) for t in tag_list) + "]",
            "---",
            "",
        ])
        link_block = ""
        if link_list:
            link_block = "\n\n## Related\n" + "\n".join(f"- [[{name}]]" for name in link_list) + "\n"

        path.write_text(frontmatter + str(content).strip() + link_block + "\n", encoding="utf-8")
        self.rebuild_index()
        self._append_log(f"{'update' if existed else 'ingest'} | {kind} | {title}")
        return {
            "kind": kind,
            "slug": slug,
            "path": str(path),
            "rel": self._rel_for(kind, slug),
            "updated": existed,
        }

    def query(self, query: str, *, kind: str | None = None, limit: int = 6) -> list[dict[str, Any]]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        wanted = Counter(tokens)
        pages = self._list_pages(kind)
        hits: list[dict[str, Any]] = []
        q_lower = str(query or "").lower()
        for page in pages:
            body = _read_safe(page["path"])
            page_tokens = Counter(_tokenize(body))
            overlap = sum(min(weight, page_tokens.get(token, 0)) for token, weight in wanted.items())
            if overlap == 0:
                continue
            density = overlap / max(sum(page_tokens.values()), 1)
            phrase = 1.0 if q_lower and q_lower in body.lower() else 0.0
            hits.append({
                "kind": page["kind"],
                "slug": page["slug"],
                "rel": page["rel"],
                "score": round(overlap + density + phrase, 4),
                "summary": self.extract_summary(body),
                "excerpt": self._excerpt(body, query),
            })
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[: max(1, int(limit))]

    def list_pages(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "kind": page["kind"],
                "slug": page["slug"],
                "rel": page["rel"],
                "summary": self.extract_summary(_read_safe(page["path"])),
            }
            for page in self._list_pages(kind)
        ]

    def _excerpt(self, body: str, query: str, radius: int = 180) -> str:
        text = body or ""
        lower = text.lower()
        q = str(query or "").strip().lower()
        pos = lower.find(q) if q else -1
        if pos < 0:
            for token in _tokenize(q):
                pos = lower.find(token)
                if pos >= 0:
                    break
        if pos < 0:
            return text[: radius * 2]
        start = max(pos - radius, 0)
        stop = min(pos + radius, len(text))
        snippet = text[start:stop].strip()
        if start > 0:
            snippet = "..." + snippet
        if stop < len(text):
            snippet += "..."
        return snippet

    def build_graph(self) -> dict[str, Any]:
        pages = self._list_pages()
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        page_tokens: dict[str, set[str]] = {}
        token_doc_freq: Counter[str] = Counter()

        for page in pages:
            page_id = f"{page['kind']}:{page['slug']}"
            body = _read_safe(page["path"])
            nodes[page_id] = {
                "id": page_id,
                "kind": page["kind"],
                "slug": page["slug"],
                "rel": page["rel"],
                "summary": self.extract_summary(body),
                "degree": 0,
            }
            tokens = set(_tokenize(_strip_frontmatter(body)))
            page_tokens[page_id] = tokens
            for token in tokens:
                token_doc_freq[token] += 1

        for page in pages:
            page_id = f"{page['kind']}:{page['slug']}"
            body = _read_safe(page["path"])
            for raw_link in {match.strip() for match in _WIKILINK_RE.findall(body) if match.strip()}:
                target_slug = _slugify(raw_link)
                target = next((n for n in nodes.values() if n["slug"] == target_slug), None)
                if target and target["id"] != page_id:
                    edges.append({"source": page_id, "target": target["id"], "type": "link"})
                    nodes[page_id]["degree"] += 1
                    target["degree"] += 1

        total_docs = len(pages)
        if total_docs >= 4:
            upper = max(2, total_docs // 2)
            significant = [
                token for token, df in sorted(token_doc_freq.items(), key=lambda kv: -kv[1])
                if 2 <= df <= upper
            ][:60]
            seen_pair: set[tuple[str, str]] = set()
            for token in significant:
                ids = [pid for pid, toks in page_tokens.items() if token in toks]
                if not (2 <= len(ids) <= 6):
                    continue
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        key = tuple(sorted((ids[i], ids[j])))
                        if key in seen_pair:
                            continue
                        seen_pair.add(key)
                        edges.append({"source": ids[i], "target": ids[j], "type": f"co:{token}", "weight": 1})
                        nodes[ids[i]]["degree"] += 1
                        nodes[ids[j]]["degree"] += 1

        node_list = list(nodes.values())
        god_nodes = sorted(node_list, key=lambda n: -n["degree"])[:8]
        orphans = [n for n in node_list if n["degree"] == 0]
        co_tokens = sorted({e["type"][3:] for e in edges if e["type"].startswith("co:")})[:20]

        graph = {
            "generated_at": _now_iso(),
            "node_count": len(node_list),
            "edge_count": len(edges),
            "nodes": node_list,
            "edges": edges,
        }
        self.graph_json_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

        report_lines = [
            "# Graph Report",
            "",
            f"_Generated {graph['generated_at']}_",
            "",
            f"- Pages indexed: {len(node_list)}",
            f"- Edges: {len(edges)} (wikilinks + token co-occurrence)",
            "",
            "## God nodes (most connected)",
            "",
        ]
        report_lines += [
            f"- **[{n['slug']}](../wiki/{n['rel']})** ({n['kind']}, degree {n['degree']}) — {n['summary']}"
            for n in god_nodes
        ] or ["_No nodes yet — ingest a page first._"]
        report_lines += ["", "## Orphan pages", ""]
        if orphans:
            report_lines += [f"- [{n['slug']}](../wiki/{n['rel']}) — {n['summary']}" for n in orphans[:12]]
        else:
            report_lines.append("_None — every page has at least one connection._")
        report_lines += ["", "## Cross-cutting tokens", ""]
        report_lines += [f"- `{tok}`" for tok in co_tokens] or ["_No cross-cutting tokens detected yet._"]
        report_lines.append("")
        self.graph_report_path.write_text("\n".join(report_lines), encoding="utf-8")

        self._append_log(f"graph_build | nodes={len(node_list)} edges={len(edges)}")
        return {
            "nodes": len(node_list),
            "edges": len(edges),
            "god_nodes": [{"id": n["id"], "degree": n["degree"], "summary": n["summary"]} for n in god_nodes],
            "orphans": len(orphans),
            "report_path": str(self.graph_report_path),
            "graph_path": str(self.graph_json_path),
        }

    def query_graph(self, query: str, *, limit: int = 8) -> dict[str, Any]:
        if not self.graph_json_path.exists():
            self.build_graph()
        try:
            graph = json.loads(_read_safe(self.graph_json_path) or "{\"nodes\":[],\"edges\":[]}")
        except json.JSONDecodeError:
            graph = {"nodes": [], "edges": []}
        tokens = _tokenize(query)
        if not tokens:
            return {"matched": [], "related": []}
        want = set(tokens)
        q_lower = str(query or "").lower()
        scored: list[dict[str, Any]] = []
        for node in graph.get("nodes", []):
            node_tokens = set(_tokenize(f"{node.get('slug','')} {node.get('summary','')}"))
            score = sum(1 for token in want if token in node_tokens)
            if q_lower and q_lower in str(node.get("slug", "")).lower():
                score += 2
            if score:
                scored.append({**node, "score": score})
        scored.sort(key=lambda n: -n["score"])
        matched = scored[:limit]
        matched_ids = {n["id"] for n in matched}
        related_ids: set[str] = set()
        for edge in graph.get("edges", []):
            if edge["source"] in matched_ids:
                related_ids.add(edge["target"])
            if edge["target"] in matched_ids:
                related_ids.add(edge["source"])
        related_ids -= matched_ids
        related = [n for n in graph.get("nodes", []) if n["id"] in related_ids][:limit]
        self._append_log(f'graph_query | "{query[:40]}" | matched={len(matched)}')
        return {"matched": matched, "related": related}

    def snapshot(self) -> dict[str, Any]:
        graph_info: dict[str, Any] | None = None
        if self.graph_json_path.exists():
            try:
                graph = json.loads(_read_safe(self.graph_json_path))
                graph_info = {
                    "nodes": graph.get("node_count", 0),
                    "edges": graph.get("edge_count", 0),
                    "generated_at": graph.get("generated_at"),
                }
            except json.JSONDecodeError:
                graph_info = None
        return {
            "root": str(self.root),
            "pages_total": sum(len(self._list_kind(k)) for k in PAGE_KINDS),
            "by_kind": {self._plural_kind(k): len(self._list_kind(k)) for k in PAGE_KINDS},
            "graph": graph_info,
            "index_path": str(self.index_path),
            "log_path": str(self.log_path),
        }

    def context_preamble(self, *, query: str = "", max_chars: int = 4000) -> str:
        """Render a compact system message describing the wiki + graph state.

        Used by ``lm_studio_client.chat_with_mcp`` to seed the LLM with the
        accumulated knowledge before answering.
        """
        sections: list[str] = ["# Energy Assistant Memory"]
        snap = self.snapshot()
        sections.append(
            f"_Wiki root_: {snap['root']} | pages={snap['pages_total']} "
            f"({', '.join(f'{k}={v}' for k, v in snap['by_kind'].items() if v)})"
        )
        if snap["graph"]:
            sections.append(
                f"_Graph_: {snap['graph']['nodes']} nodes / {snap['graph']['edges']} edges "
                f"(built {snap['graph']['generated_at']})"
            )
        if self.graph_report_path.exists():
            report = _read_safe(self.graph_report_path).strip()
            if report:
                sections.append("## Graph Report\n" + report)
        if query:
            hits = self.query(query, limit=4)
            if hits:
                lines = ["## Relevant wiki pages"]
                for hit in hits:
                    lines.append(f"- [{hit['slug']}]({hit['rel']}) — {hit['summary']}")
                    lines.append(f"  > {hit['excerpt']}")
                sections.append("\n".join(lines))
        log_excerpt = self._tail(self.log_path, 6)
        if log_excerpt:
            sections.append("## Recent log entries\n" + log_excerpt)
        rendered = "\n\n".join(sections).strip()
        if len(rendered) > max_chars:
            rendered = rendered[:max_chars - 3] + "..."
        return rendered

    @staticmethod
    def _tail(path: Path, count: int) -> str:
        if not path.exists():
            return ""
        lines = _read_safe(path).splitlines()
        headings = [line for line in lines if line.startswith("## [")]
        return "\n".join(headings[-count:])


def default_memory() -> WikiMemory:
    return WikiMemory()


__all__ = ["WikiMemory", "PAGE_KINDS", "default_memory"]
