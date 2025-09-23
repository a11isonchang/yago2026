import json

# 輸入檔案與輸出檔案路徑
input_path = "2026_likelihood_output.json"
output_path = "2026_possible_false.json"

def filter_false_entries(input_file, output_file):
    # 讀取原始 JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 過濾出 possible_in_2026 = false 的資料
    false_entries = [
        {
            "id": item["id"],
            "likelihood": item["likelihood"],
            "rationale": item["rationale"]
        }
        for item in data["results"]
        if item.get("possible_in_2026") is False
    ]

    # 存成新的 JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(false_entries, f, ensure_ascii=False, indent=2)

    print(f"已輸出 {len(false_entries)} 筆資料到 {output_file}")

# 執行
filter_false_entries(input_path, output_path)
