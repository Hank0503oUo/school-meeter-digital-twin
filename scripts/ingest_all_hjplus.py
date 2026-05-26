import sys
from pathlib import Path
import os
import json

# 強制加入 demo 根目錄到 sys.path
demo_root = Path("D:/idf優化/demo")
sys.path.append(str(demo_root))

from src.knowledge_base import KnowledgeWorkbench

def ingest_all_hjplus():
    wb = KnowledgeWorkbench()
    hjplus_root = Path("D:/idf優化/HJPLUS_Taiwan_Architect_KB-main/HJPLUS_Taiwan_Architect_KB-main")
    
    # 搜尋所有 markdown 檔案
    md_files = list(hjplus_root.glob("*.md")) + list(hjplus_root.glob("raw/**/*.md"))
    
    print(f"Starting full ingestion of {len(md_files)} files from HJPLUS KB...")
    
    count = 0
    errors = 0
    for md_path in md_files:
        try:
            content = md_path.read_bytes()
            # 使用相對路徑作為 title 的一部分以保留階層觀念
            rel_path = md_path.relative_to(hjplus_root)
            title = f"HJPLUS: {rel_path}"
            
            wb.ingest_upload(
                filename=md_path.name,
                content=content,
                building_id="hjplus_kb",
                title=title,
                tags=["hjplus", "architectural", "concept"]
            )
            count += 1
            if count % 10 == 0:
                print(f"Progress: {count}/{len(md_files)} files ingested...")
        except Exception as e:
            print(f"Failed to ingest {md_path}: {e}")
            errors += 1
            
    print(f"\nIngestion Complete!")
    print(f"Total Success: {count}")
    print(f"Total Errors: {errors}")
    
    # 強制重建索引以確保所有分塊都已就緒
    print("Rebuilding search index...")
    chunk_count = wb.rebuild_index()
    print(f"Total chunks in knowledge base: {chunk_count}")

    # 進行觀念檢索測試
    test_queries = ["建築法", "防火避難", "結構安全", "容積率"]
    print("\nVerifying concepts with test queries:")
    for query in test_queries:
        hits = wb.search_chunks(query=query, building_id="hjplus_kb", top_k=2)
        print(f"\nQuery: '{query}'")
        if hits:
            for h in hits:
                print(f"  - [{h['title']}] Score: {h['score']}")
        else:
            print("  - No results found.")

if __name__ == "__main__":
    ingest_all_hjplus()
