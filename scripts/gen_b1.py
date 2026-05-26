import json, os  
os.makedirs("data/lora", exist_ok=True)  
b1 = []  
def r(role, query, tool, args, avoid, diff, reason):  
    b1.append({"user_role":role,"user_query":query,"expected_intent":tool,"expected_tool":tool,"expected_arguments":args,"should_not_use":avoid,"difficulty":diff,"reason":reason})  
print(f"Batch 1: {len(b1)} records")  
    r("總務處能源管理員","今年全校總用電量是多少？和去年比差多少？","compare_campus_years",{"years":[2025,2024],"campus":"NTU","metric":"electricity_usage"},["get_campus_annual_usage"],"easy","明確的跨年比較") 
