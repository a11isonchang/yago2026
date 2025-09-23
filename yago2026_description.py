import json

# 讀取原始 JSON 檔案
input_path = "/Users/allisonchang/Downloads/BeyondKnownFacts/yago2026.json"
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 只保留 description 欄位
descriptions = [{"description": item["description"]} for item in data if "description" in item]

# 輸出新的 JSON 檔案
output_path = "yago2026_descriptions.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(descriptions, f, ensure_ascii=False, indent=2)
