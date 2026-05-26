# -*- coding: utf-8 -*-
"""
NTU 校園能源數位分身 — Panel Dashboard 入口

啟動方式：
    panel serve app.py --show --port 5006 --autoreload

匯出靜態 HTML：
    python app.py --export campus_dashboard.html
"""

import sys
import argparse
from pathlib import Path

# ── 確保 demo/ 為工作目錄 ─────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard import create_dashboard


def main_export():
    """CLI 匯出模式：python app.py --export output.html"""
    parser = argparse.ArgumentParser(description="NTU Campus Energy Digital Twin")
    parser.add_argument("--export", default=None,
                        help="匯出靜態 HTML 路徑 (不啟動伺服器)")
    args = parser.parse_args()

    if args.export:
        dashboard = create_dashboard()
        # FastListTemplate 不支援 embed=True
        dashboard.save(args.export)
        print(f"Dashboard 已匯出: {args.export}")
    else:
        print("提示: 請使用 'panel serve app.py --show' 啟動 Dashboard")
        print("      或使用 'python app.py --export output.html' 匯出 HTML")


# ── panel serve 入口 ──────────────────────────────────────
# `panel serve app.py` 執行此模組時 __name__ != "__main__"
# 此處建立 Dashboard 並標記為 servable
if __name__ != "__main__":
    dashboard = create_dashboard()
    dashboard.servable()

if __name__ == "__main__":
    main_export()
