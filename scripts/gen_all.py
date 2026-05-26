import json, os  
os.makedirs("data/lora", exist_ok=True)  
def w(name, data):  
    with open(f"data/lora/{name}.jsonl","w",encoding="utf-8") as f:  
        for r in data: f.write(json.dumps(r,ensure_ascii=False)+"\n")  
    print(f"Wrote {len(data)} records to {name}")  
b1 = []  
print("script ready")  
w("test_batch", [{"test": 1}]) 
