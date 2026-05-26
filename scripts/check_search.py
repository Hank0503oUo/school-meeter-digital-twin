import sys
from pathlib import Path
demo_root = Path("D:/idf優化/demo")
sys.path.append(str(demo_root))
from src.knowledge_base import KnowledgeWorkbench, _tokenize
from collections import Counter

wb = KnowledgeWorkbench()
query = "建築法"
query_tokens = _tokenize(query)
print(f"Query: '{query}'")
print(f"Query tokens: {query_tokens}")

chunks = wb.list_chunks(building_id="hjplus_kb")
print(f"Total chunks: {len(chunks)}")

found_count = 0
for c in chunks:
    if query in c.text:
        found_count += 1
        if found_count == 1:
            print(f"First match found in: {c.title}")
            text_tokens = _tokenize(c.text)
            print(f"Text tokens (partial): {text_tokens[:20]}")
            overlap = [t for t in query_tokens if t in text_tokens]
            print(f"Overlap: {overlap}")

print(f"Total matching chunks (raw find): {found_count}")

hits = wb.search_chunks(query=query, building_id="hjplus_kb")
print(f"Search hits: {len(hits)}")
