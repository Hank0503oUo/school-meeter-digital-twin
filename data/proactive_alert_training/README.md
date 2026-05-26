# Proactive Alert Training Pack

這包語料是給其他 agent 擴寫「主動異常通報後，人類管理者追問時 agent 應該怎麼回覆」。

目前 MCP 已新增工具：

- `scan_iot_snapshot_for_alerts`
- `create_energy_alert`
- `list_active_energy_alerts`
- `acknowledge_energy_alert`
- `close_energy_alert`
- `notify_energy_manager`
- `recommend_anomaly_decision`

## 其他 agent 的任務

請讀 `response_authoring_prompts.jsonl`，為每一筆補上 `assistant_response`。

回答必須固定包含：

1. 異常類型
2. 嚴重度
3. 證據
4. 可能原因
5. 建議處置
6. 是否通知 / 是否開工單
7. 下一個建議工具

禁止事項：

- 不要捏造不存在的數值。
- 不要說已經派人或已經控制設備，除非 user 明確說已完成。
- 不要直接建議自動改控制參數；只能建議人工確認或開工單。
- 不要輸出與能源/RTEM/BMS 無關的內容。

## 檔案

- `router_seed.jsonl`：router 訓練用，格式是 user + expected_tool。
- `response_authoring_prompts.jsonl`：給其他 agent 擴寫自然語言回覆。
