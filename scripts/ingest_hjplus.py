import sys
from pathlib import Path
import os

# 強制加入 demo 根目錄到 sys.path
demo_root = Path("D:/idf優化/demo")
sys.path.append(str(demo_root))

from src.knowledge_base import KnowledgeWorkbench

def ingest_hjplus():
    wb = KnowledgeWorkbench()
    hjplus_root = Path("D:/idf優化/HJPLUS_Taiwan_Architect_KB-main/HJPLUS_Taiwan_Architect_KB-main")
    
    # 搜尋 markdown 檔案
    md_files = list(hjplus_root.glob("*.md")) + list(hjplus_root.glob("raw/**/*.md"))
    
    print(f"Found {len(md_files)} markdown files in HJPLUS KB.")
    
    count = 0
    for md_path in md_files[:10]: # 先嘗試前 10 個
        try:
            content = md_path.read_bytes()
            wb.ingest_upload(
                filename=md_path.name,
                content=content,
                building_id="hjplus_kb",
                title=f"HJPLUS: {md_path.stem}",
                tags=["hjplus", "architectural"]
            )
            print(f"Ingested: {md_path.name}")
            count += 1
        except Exception as e:
            print(f"Failed to ingest {md_path.name}: {e}")
            
    print(f"Total ingested: {count}")
    
    # 測試檢索
    query = "Contribute"
    hits = wb.search_chunks(query=query, building_id="hjplus_kb")
    print(f"\nTesting search for '{query}':")
    for hit in hits:
        print(f"- {hit['title']} (Score: {hit['score']})")

if __name__ == "__main__":
    ingest_hjplus()
