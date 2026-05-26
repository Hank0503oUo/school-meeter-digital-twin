import json  
sp = "你是 NTU 校園能源助理。"  
data = [{"u":"test","a":"test answer","s":5}]  
with open("D:/idf優化/demo/data/lora/harness_output.jsonl","w",encoding="utf-8") as f:  
    for d in data:  
        f.write(json.dumps({"messages":[{"role":"system","content":sp},{"role":"user","content":d["u"]},{"role":"assistant","content":d["a"]}]},ensure_ascii=False)+"\n")  
print("done")  
