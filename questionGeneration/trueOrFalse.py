import os
import json
import time
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# === 設定 ===
API_URL = "https://outer-medusa.genai.nchc.org.tw/v1/chat/completions"
MODEL = "gpt-oss-120b"  # 依 NCHC 實際可用模型調整
BATCH_SIZE = 20
TEMPERATURE = 0.2
TIMEOUT = 60
RETRY = 3
RETRY_WAIT = 2  # 秒

# 載入 .env（建議在 .env 內放 API_KEY=xxxxx）
load_dotenv()
API_KEY = os.getenv("NCHC_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "accept": "application/json",
}

# === Prompts ===
SYSTEM_PROMPT = (
    "You are a data transformation assistant. "
    "Always follow the rules strictly and output valid JSON only."
)

# ⛳ 重點：這版 user 指令假設「我已經把要處理的 items 以 JSON 貼在 Input: 後面」，
# 不再要求模型自己去讀附件，避免訊息衝突。
USER_INSTRUCTIONS = (
    """TASK
You will receive an array named "Input" that contains objects with fields:
- id (string or number)
- description (string)

For EACH item in Input, generate FOUR true/false statements with likelihood labels:
- "Highly likely"
- "Possible"
- "Unlikely"
- "Highly unlikely"

GENERATION RULES
- Use ONLY information present in the item's 'description'; do not add external facts.
- Preserve key entities, dates, organizations, and roles.
- Create statements as follows:
  • Highly likely: a faithful, tight paraphrase that would be true if the description is true.
  • Possible: a softened variant (e.g., “around 2026”, “may have”), still consistent with the description.
  • Unlikely: a plausible-sounding inversion of the core relation (e.g., joined↔left, continued↔ended).
  • Highly unlikely: a strong contradiction or role reversal that clearly conflicts with the description.
- Avoid hedging words in “Highly likely” and “Highly unlikely”.
- Keep each statement ≤ 30 words.
- Do NOT include analysis or explanations—output JSON only.

OUTPUT FORMAT
- Return a single JSON ARRAY (not prose). The array length must be exactly 4 × len(Input).
- Each element is an object:
  {
    "id": "<original-id>_<suffix>",  // suffix ∈ {"highly_likely","possible","unlikely","highly_unlikely"}
    "statement": "<the generated true/false statement>",
    "label": "Highly likely" | "Possible" | "Unlikely" | "Highly unlikely"
  }

VALIDATION
- If the description lacks enough detail to invert safely, keep entities/timeframe but invert the main relation reasonably.
- If exact dates appear (e.g., “2026-01-01”), keep them exact in “Highly likely”; in “Possible” you may relax to “around 2026”.
"""
)

def call_model(items_batch):
    """
    傳入一批 items（list of dict: {id, description}），要求模型回傳 JSON array，
    其中每個 input 產出 4 筆（共 4*len(batch) 筆）。
    """
    user_content = USER_INSTRUCTIONS + "\nInput:\n" + json.dumps(items_batch, ensure_ascii=False, indent=2)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": TEMPERATURE,
        # 若對方支援 JSON 模式可考慮啟用（NCHC 端如支援 OpenAI json_object）
        # "response_format": {"type": "json_object"}  # 這行改為 array 就不適用，故先關閉
    }

    last_err = None
    for attempt in range(1, RETRY + 1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            # 取出文字
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                content = json.dumps(data, ensure_ascii=False)

            # 解析：預期是一個 JSON array；同時容忍 {"results":[...]} 的情況
            parsed = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                start = content.find("[")
                end = content.rfind("]")
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(content[start:end+1])
                else:
                    # 再嘗試 dict
                    start = content.find("{")
                    end = content.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        maybe = json.loads(content[start:end+1])
                        if isinstance(maybe, dict) and "results" in maybe:
                            parsed = maybe["results"]
                        else:
                            raise ValueError(f"Invalid JSON content:\n{content}")
                    else:
                        raise ValueError(f"Invalid JSON content:\n{content}")

            # 標準化為 array
            if isinstance(parsed, dict) and "results" in parsed:
                parsed = parsed["results"]
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got: {type(parsed)}")

            return parsed, data  # (本批結果 array, 原始回應)

        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < RETRY:
                time.sleep(RETRY_WAIT)
            else:
                raise
        except Exception as e:
            last_err = e
            if attempt < RETRY:
                time.sleep(RETRY_WAIT)
            else:
                raise

    raise RuntimeError(f"呼叫模型失敗：{last_err}")

def load_input(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input JSON 必須是陣列，每筆包含 {id, description}")
    for i, item in enumerate(data):
        if "id" not in item or "description" not in item:
            raise ValueError(f"第 {i} 筆缺少 id 或 description：{item}")
    return data

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def main(
    input_path="yago2026_possible.json",
    output_path="true_false_output.json",
    raw_log_path="true_false_raw_responses.jsonl",
    only_first_10=False
):
    try:
        items = load_input(input_path)

        # ⛳ 若只想跑前 10：
        if only_first_10:
            items = items[:10]

        all_results = []

        with open(raw_log_path, "w", encoding="utf-8") as raw_fp:
            batches = list(chunked(items, BATCH_SIZE))
            for batch in tqdm(batches, desc="分析進度"):
                parsed_array, raw = call_model(batch)

                # 期望 4 × len(batch) 筆
                expected = 4 * len(batch)
                if len(parsed_array) != expected:
                    # 不終止流程，但警告
                    print(f"⚠️ 批次輸出數量不符：got {len(parsed_array)} expected {expected}")

                all_results.extend(parsed_array)
                raw_fp.write(json.dumps(raw, ensure_ascii=False) + "\n")

        save_json(output_path, all_results)  # 直接存 array，比較通用
        print(f"\n✅ 完成：{output_path}\n📝 原始回應紀錄：{raw_log_path}")

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")

if __name__ == "__main__":
    # 只跑前 10 的範例：把 only_first_10=True 即可
    main(only_first_10=True)
