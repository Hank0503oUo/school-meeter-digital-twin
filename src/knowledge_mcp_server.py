from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server import FastMCP

from src.algorithm_mcp_backend import AlgorithmMCPBackend
from src.knowledge_mcp_backend import KnowledgeMCPBackend
from src.proactive_alerts import (
    create_energy_alert_impl,
    list_active_energy_alerts_impl,
    notify_energy_manager_impl,
    recommend_anomaly_decision_impl,
    scan_iot_snapshot_for_alerts_impl,
    update_energy_alert_status_impl,
)

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("knowledge-mcp-server")

def build_server(*, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    """
    構建並配置 MCP 伺服器，整合建築能源知識庫與演算法後端。
    """
    # 初始化後端
    try:
        backend = KnowledgeMCPBackend()
        algo_backend = AlgorithmMCPBackend()
        logger.info("Successfully initialized knowledge and algorithm backends.")
    except Exception as e:
        logger.error(f"Failed to initialize backends: {e}")
        raise

    server = FastMCP(
        name="building-energy-knowledge",
        instructions=(
            "所有回答必須基於本地建築能源知識庫。優先使用本體 (Ontology) 和記憶 (Memory)，"
            "其次查詢文件和 CSV 摘要。僅將經核准的輸出保存至精選追蹤 (Curated Traces) 中。"
        ),
        host=host,
        port=port,
    )

    # --- 知識檢索工具 ---

    @server.tool(description="在本地建築能源知識庫中搜尋索引的文件分塊。支援篩選建築 ID 與特定文件。")
    def search_docs(
        query: str,
        building_id: str = "",
        selected_docs: list[str] | None = None,
        selected_csvs: list[str] | None = None,
        top_k: int = 6,
    ) -> dict:
        return backend.search_docs(
            query=query,
            building_id=building_id,
            selected_docs=selected_docs or [],
            selected_csvs=selected_csvs or [],
            top_k=top_k,
        )

    @server.tool(description="根據 chunk_id 從知識庫中獲取特定的文本分塊內容。")
    def fetch_chunk(chunk_id: str) -> dict:
        return backend.fetch_chunk(chunk_id=chunk_id)

    @server.tool(description="依文件標題、doc_id、路徑或 building_id 找知識庫文件。不讀內容，只回傳文件清單。")
    def find_docs(
        query: str = "",
        building_id: str = "",
        source_type: str = "",
        limit: int = 20,
    ) -> dict:
        return backend.find_docs(query=query, building_id=building_id, source_type=source_type, limit=limit)

    @server.tool(description="在知識庫原始/解析文件中做精確關鍵字或 regex 搜尋。適合法規條號、OpenBSE、EUI、CV-RMSE、HVAC 等精確線索。")
    def grep_docs(
        pattern: str,
        building_id: str = "",
        selected_docs: list[str] | None = None,
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 50,
    ) -> dict:
        return backend.grep_docs(
            pattern=pattern,
            building_id=building_id,
            selected_docs=selected_docs or [],
            regex=regex,
            case_sensitive=case_sensitive,
            limit=limit,
        )

    @server.tool(description="讀取指定文件片段。可用 doc_id 或 path 定位，支援 start_line 和 max_lines。")
    def read_doc_chunk(
        doc_id: str = "",
        path: str = "",
        start_line: int = 1,
        max_lines: int = 80,
    ) -> dict:
        return backend.read_doc_chunk(doc_id=doc_id, path=path, start_line=start_line, max_lines=max_lines)

    @server.tool(description="檢查某個關鍵字 match 的前後文，用於定位證據。回傳 before/match/after 結構。")
    def inspect_doc_context(
        pattern: str,
        doc_id: str = "",
        path: str = "",
        before: int = 5,
        after: int = 8,
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 10,
    ) -> dict:
        return backend.inspect_doc_context(
            pattern=pattern,
            doc_id=doc_id,
            path=path,
            before=before,
            after=after,
            regex=regex,
            case_sensitive=case_sensitive,
            limit=limit,
        )

    @server.tool(description="統計關鍵字在各文件中的出現次數，用於縮小搜尋範圍。")
    def count_doc_matches(
        pattern: str,
        building_id: str = "",
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 50,
    ) -> dict:
        return backend.count_doc_matches(
            pattern=pattern,
            building_id=building_id,
            regex=regex,
            case_sensitive=case_sensitive,
            limit=limit,
        )

    @server.tool(description="查詢特定建築的本體實體資訊，包括文件、計量表 (Meters) 與 KPI。")
    def lookup_building_entity(building_id: str) -> dict:
        return backend.lookup_building_entity(building_id=building_id)

    @server.tool(description="對特定建築或選定的 CSV 檔案進行數據摘要與統計分析。")
    def query_meter_or_kpi(building_id: str = "", selected_csvs: list[str] | None = None) -> dict:
        return backend.query_meter_or_kpi(building_id=building_id, selected_csvs=selected_csvs or [])

    # --- 分析與追蹤工具 ---

    @server.tool(description="執行雲端優先的分析任務，若雲端不可用則回退至本地工作台。")
    def run_analysis(
        building_id: str,
        task_type: str,
        user_query: str,
        selected_docs: list[str] | None = None,
        selected_csvs: list[str] | None = None,
    ) -> dict:
        return backend.run_analysis(
            building_id=building_id,
            task_type=task_type,
            user_query=user_query,
            selected_docs=selected_docs or [],
            selected_csvs=selected_csvs or [],
        )

    @server.tool(description="將核准的分析結果保存至精選追蹤 (Curated Traces)，並可選擇同步至 MEMORY.md。")
    def save_curated_trace(
        building_id: str,
        task_type: str,
        user_query: str,
        answer_markdown: str,
        extracted_json: dict | None = None,
        cited_chunks: list[dict] | None = None,
        confidence: float = 0.7,
        followups: list[str] | None = None,
        adapter_name: str = "manual",
        reviewer_notes: str = "",
        save_to_memory: bool = False,
        memory_title: str = "",
    ) -> dict:
        return backend.save_curated_trace(
            building_id=building_id,
            task_type=task_type,
            user_query=user_query,
            answer_markdown=answer_markdown,
            extracted_json=extracted_json,
            cited_chunks=cited_chunks,
            confidence=confidence,
            followups=followups or [],
            adapter_name=adapter_name,
            reviewer_notes=reviewer_notes,
            save_to_memory=save_to_memory,
            memory_title=memory_title,
        )

    # --- 能源演算法工具 ---

    @server.tool(
        description=(
            "運行 PI-VD 四層架構能源推論。支援指定開始時間與時數，"
            "若未提供天氣序列，系統將自動從 models/weather 加載對應的 EPW 數據。"
        )
    )
    def run_pvid(
        building_uid: str = "",
        hours: int = 24,
        t_out_series: list[float] = [],
        humidity_series: list[float] = [],
        start_time: str = "",
    ) -> dict:
        return algo_backend.run_pvid(
            building_uid=building_uid,
            hours=hours,
            t_out_series=t_out_series,
            humidity_series=humidity_series,
            start_time=start_time,
        )

    @server.tool(description="對多個演算法結果進行相關性分析，並推理算法間的交叉關係。")
    def correlate_algorithms(
        results: list[dict],
        question: str = "",
        building_uid: str = "",
    ) -> dict:
        return algo_backend.correlate_algorithms(
            results=results,
            question=question,
            building_uid=building_uid,
        )

    # --- 主動異常告警 / 人類決策支援工具 ---

    @server.tool(description=(
        "Scan a scheduled IoT/RTEM/BMS snapshot for anomaly candidates and optionally "
        "create alert events. Use this for proactive background monitoring before "
        "notifying a human manager."
    ))
    def scan_iot_snapshot_for_alerts(
        snapshot: dict,
        source: str = "scheduled_scan",
        create_alerts: bool = True,
    ) -> dict:
        return scan_iot_snapshot_for_alerts_impl(
            snapshot=snapshot,
            source=source,
            create_alerts=create_alerts,
        )

    @server.tool(description="Create a persistent energy anomaly alert event for operator review.")
    def create_energy_alert(
        title: str,
        summary: str,
        severity: str = "medium",
        event_type: str = "anomaly",
        building_uid: str = "",
        meter_name: str = "",
        anomaly_type: str = "",
        evidence: dict | None = None,
        recommended_actions: list[str] | str | None = None,
        source: str = "agent",
    ) -> dict:
        return create_energy_alert_impl(
            title=title,
            summary=summary,
            severity=severity,
            event_type=event_type,
            building_uid=building_uid,
            meter_name=meter_name,
            anomaly_type=anomaly_type,
            evidence=evidence,
            recommended_actions=recommended_actions,
            source=source,
        )

    @server.tool(description="List open or acknowledged energy alerts for the dashboard or operator triage.")
    def list_active_energy_alerts(
        severity_min: str = "low",
        building_uid: str = "",
        limit: int = 20,
    ) -> dict:
        return list_active_energy_alerts_impl(
            severity_min=severity_min,
            building_uid=building_uid,
            limit=int(limit),
        )

    @server.tool(description="Mark an energy alert as acknowledged by a human operator.")
    def acknowledge_energy_alert(alert_id: str, operator: str = "", note: str = "") -> dict:
        return update_energy_alert_status_impl(
            alert_id=alert_id,
            status="acknowledged",
            operator=operator,
            note=note,
        )

    @server.tool(description="Close an energy alert after resolution or mark it as false_positive.")
    def close_energy_alert(
        alert_id: str,
        operator: str = "",
        note: str = "",
        false_positive: bool = False,
    ) -> dict:
        return update_energy_alert_status_impl(
            alert_id=alert_id,
            status="false_positive" if false_positive else "closed",
            operator=operator,
            note=note,
        )

    @server.tool(description=(
        "Queue a notification for an energy manager. Current implementation writes to "
        "outputs/energy_manager/notification_outbox.jsonl; external email/LINE/Teams "
        "senders can consume that outbox."
    ))
    def notify_energy_manager(
        alert_id: str = "",
        channel: str = "outbox",
        recipients: list[str] | str | None = None,
        message: str = "",
        dry_run: bool = True,
    ) -> dict:
        return notify_energy_manager_impl(
            alert_id=alert_id,
            channel=channel,
            recipients=recipients,
            message=message,
            dry_run=dry_run,
        )

    @server.tool(description=(
        "Recommend an operator decision for an anomaly alert: notify, create ticket, "
        "monitor, or mark as lower priority."
    ))
    def recommend_anomaly_decision(alert: dict | None = None, alert_id: str = "") -> dict:
        return recommend_anomaly_decision_impl(alert=alert, alert_id=alert_id)

    # --- HARNESS 長期記憶工具 ---

    @server.tool(description=(
        "Extract keywords, entities (buildings), and intent hints from a user query. "
        "Uses deterministic building alias matching and keyword rules. "
        "Use this before searching HARNESS memory on each user query."
    ))
    def extract_harness_keywords(query: str) -> dict:
        return backend.extract_harness_keywords(query=query)

    @server.tool(description=(
        "Search HARNESS long-term memory for similar events and reusable tool plans. "
        "Returns scored hits from both event memory and procedure memory. "
        "Use this after extract_harness_keywords to find prior successful interactions."
    ))
    def search_harness_memory(query: str, top_k: int = 3) -> dict:
        return backend.search_harness_memory(query=query, top_k=int(top_k))

    @server.tool(description=(
        "Record a HARNESS interaction event including the query, tool trace, results, "
        "and quality metadata. Automatically promotes to procedure memory if quality gates pass "
        "and promote_to_procedure=true."
    ))
    def record_harness_event(
        user_query: str,
        keywords: list[str] | None = None,
        entities: list[dict] | None = None,
        intent: str = "",
        selected_tool_plan: list[dict] | None = None,
        tool_trace: list[dict] | None = None,
        final_answer_summary: str = "",
        quality: dict | None = None,
        outcome: str = "unknown",
        promote_to_procedure: bool = False,
        training_tags: list[str] | None = None,
    ) -> dict:
        event = {
            "user_query": user_query,
            "keywords": keywords or [],
            "entities": entities or [],
            "intent": intent,
            "selected_tool_plan": selected_tool_plan or [],
            "tool_trace": tool_trace or [],
            "final_answer_summary": final_answer_summary,
            "quality": quality or {},
            "outcome": outcome,
            "promote_to_procedure": promote_to_procedure,
            "training_tags": training_tags or [],
        }
        return backend.record_harness_event(event)

    @server.tool(description=(
        "Promote a recorded HARNESS event to a reusable procedure in procedure memory. "
        "The event must pass quality gates: tool_correct, numbers_correct, answer_grounded, "
        "judge_score >= 0.75, and at least one successful tool call."
    ))
    def promote_harness_procedure(event_id: str, procedure_hint: str = "") -> dict:
        return backend.promote_harness_procedure(event_id=event_id, procedure_hint=procedure_hint)

    @server.tool(description=(
        "Get HARNESS startup context: recent successful procedures, frequent buildings, "
        "known failure modes, and a compact memory summary for Agent context. "
        "Call this at app/session startup before the first user turn."
    ))
    def get_harness_startup_context(campus: str = "ntu", limit: int = 8) -> dict:
        return backend.get_harness_startup_context(campus=campus, limit=int(limit))

    # --- 資源 (Resources) ---

    @server.resource("knowledge://status", name="knowledge-status", mime_type="application/json")
    def knowledge_status() -> dict:
        return backend.status()

    @server.resource("knowledge://buildings", name="knowledge-buildings", mime_type="application/json")
    def buildings() -> list[dict]:
        return backend.list_buildings()

    @server.resource("knowledge://building/{building_id}/ontology", name="building-ontology", mime_type="text/plain")
    def building_ontology(building_id: str) -> str:
        return backend.read_ontology(building_id)

    @server.resource("knowledge://building/{building_id}/memory", name="building-memory", mime_type="text/markdown")
    def building_memory(building_id: str) -> str:
        return backend.read_memory(building_id)

    # --- 直接 LLM 推論工具 (Gemini 2.0/3.1) ---

    @server.tool(description="直接調用 Gemini 模型進行能源分析推論，支援從本地知識庫獲取脈絡。")
    def ask_gemini_inference(
        user_query: str,
        building_id: str = "general",
        task_type: str = "qa",
    ) -> dict:
        import json

        # 從環境變數獲取配置，確保安全性
        api_key = os.getenv("ENERGY_LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = os.getenv("ENERGY_LLM_MODEL", "gemini-2.0-flash-exp").strip()
        max_tokens = int(os.getenv("ENERGY_LLM_MAX_TOKENS", "4096"))

        if not api_key:
            logger.error("GEMINI_API_KEY is not configured in environment.")
            return {"error": "GEMINI_API_KEY is not set. Please check environment variables.", "model": model}

        # 嘗試豐富背景脈絡
        context_lines: list[str] = []
        try:
            entity = backend.lookup_building_entity(building_id=building_id)
            if entity:
                context_lines.append(f"Building context from KB: {json.dumps(entity, ensure_ascii=False)[:3000]}")
        except Exception as e:
            logger.warning(f"Failed to fetch building context for {building_id}: {e}")

        system_prompt = (
            "You are a building energy analysis assistant for a university campus digital twin. "
            "Your answers must be concise, accurate, and grounded in provided building data. "
            "Always respond in the same language as the user query."
        )
        
        user_content = user_query
        if context_lines:
            user_content = "\n".join(context_lines) + "\n\nUser Question: " + user_query

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    max_output_tokens=max_tokens,
                ),
            )
            return {
                "answer": response.text or "",
                "model": model,
                "building_id": building_id,
                "status": "success",
            }
        except Exception as exc:
            logger.error(f"Gemini inference failed: {exc}")
            return {"error": str(exc), "model": model, "status": "failed"}

    return server

def main() -> None:
    # 支援 stdio 模式，方便作為 MCP server 調用
    server = build_server()
    logger.info("Starting building-energy-knowledge MCP server on stdio...")
    server.run("stdio")

if __name__ == "__main__":
    main()
