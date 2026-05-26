# -*- coding: utf-8 -*-
"""
Knowledge workbench launcher.

Run with:
    panel serve app_workbench.py --show --port 5007 --autoreload
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard import create_knowledge_workbench


def main_export():
    parser = argparse.ArgumentParser(description="Building Energy Knowledge Workbench")
    parser.add_argument("--export", default=None, help="Export the workbench as an HTML file")
    args = parser.parse_args()

    if args.export:
        dashboard = create_knowledge_workbench()
        dashboard.save(args.export)
        print(f"Workbench exported to {args.export}")
    else:
        print("Use 'panel serve app_workbench.py --show --port 5007' to launch the knowledge workbench.")


if __name__ != "__main__":
    dashboard = create_knowledge_workbench()
    dashboard.servable()

if __name__ == "__main__":
    main_export()
