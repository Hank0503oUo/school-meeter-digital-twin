import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.makedirs("data/lora", exist_ok=True)
def w(name, data):
    with open(f"data/lora/{name}.jsonl","w",encoding="utf-8") as f:
        for r in data: f.write(json.dumps(r,ensure_ascii=False)+"\\n")
    print(f"Wrote {len(data)} to {name}")
b1 = []

b1.append({'role':'總務處能源管理員','q':'今年全校總用電量是多少？和去年比差多少？'})
print('done')
b1.append(('總務處能源管理員','今年全校總用電量是多少？和去年比差多少？','compare_campus_years','{"years":[2025,2024],"campus":"NTU","metric":"electricity_usage"}','["get_campus_annual_usage"]','easy','明確的跨年比較'))