import sys
from pathlib import Path
demo_root = Path("D:/idf優化/demo")
sys.path.append(str(demo_root))
from src.knowledge_base import KnowledgeWorkbench, _tokenize

wb = KnowledgeWorkbench()
query = "建築法"
# 手動嘗試在腳本內定義
import re
LOCAL_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")
def local_tokenize(text):
    return [token.lower() for token in LOCAL_RE.findall(text or "")]

print(f"Query: '{query}'")
print(f"Src tokenize: {_tokenize(query)}")
print(f"Local tokenize: {local_tokenize(query)}")

chunks = wb.list_chunks(building_id="hjplus_kb")
for c in chunks:
    if query in c.text:
        print(f"Found match in {c.title}")
        print(f"Src tokenize text: {_tokenize(c.text)[:5]}")
        break
