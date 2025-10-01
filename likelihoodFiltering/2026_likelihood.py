import os
import json
import time
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# === 設定 ===
API_URL = "https://outer-medusa.genai.nchc.org.tw/v1/chat/completions"
MODEL = "gpt-oss-120b"  # 請依 NCHC 可用模型調整
BATCH_SIZE = 20  # 每批處理幾條 description，可視情況調整
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

# 系統提示：要求只用我們定義的 JSON 結構回答
SYSTEM_PROMPT = (
   "Analyze whether the following event description could realistically occur in 2026. Return STRICT JSON ONLY with the required schema. Do not include extra text."

    "Please evaluate:"
    "1. **Factual Accuracy**: Are the people, organizations, and relationships mentioned realistic and factually plausible?"
    "2. **Timeline Feasibility**: Is the specified date/timeframe reasonable?"
    "3. **Real-world Plausibility**: Could this scenario actually happen given current knowledge of the people/organizations involved? Even though the possibility is low, if it is not impossible, please consider it as possible."

    )

    # 要模型輸出的結構
USER_INSTRUCTIONS = (
    "Given an array of items, each with fields {id, description}, "
    "assess each item and output a JSON object with this exact shape:\n\n"
    "{\n"
    '  "results": [\n'
    "    {\n"
    '      "id": string,\n'
    '      "possible_in_2026": boolean,\n'
    '      "likelihood": "impossible" | "low" | "medium" | "high",\n'
    '      "rationale": string\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Base the judgment on general plausibility by 2026 (not certainty).\n"
    "- If the scenario is absolutely impossible in 2026, then possible_in_2026 = false, otherwise true.\n"
    "- Use concise, concrete rationale (<= 2 sentences).\n"
    "- The array order in results must follow the input order.\n"
    "- Answer in English.\n"
)

def call_model(items_batch):
    """
    給模型一批 items（list of dict: {id, description}），回傳解析後的 JSON dict：
    { "results": [ {id, possible_in_2026, likelihood, rationale}, ... ] }
    """
    # 組合用戶訊息：把批次的 items 放進去
    user_content = USER_INSTRUCTIONS + "\nInput:\n" + json.dumps(items_batch, ensure_ascii=False, indent=2)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": TEMPERATURE,
        # 若對方支援 JSON 模式可以啟用（不確定 NCHC 是否支援，先保留）
        # "response_format": {"type": "json_object"}
    }

    last_err = None
    for attempt in range(1, RETRY + 1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()

            # 嘗試解析 JSON 回應
            data = resp.json()

            # 依 OpenAI 相容格式取文字
            content = None
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                # 不符合預期格式，改存 raw
                content = json.dumps(data, ensure_ascii=False)

            # 解析 content 成 JSON（模型應該回 STRICT JSON）
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # 若模型沒遵守，退而求其次：嘗試從回應裡找第一個 { 開始到最後一個 } 的區段
                # 以提高健壯性（但仍可能失敗）
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(content[start:end+1])
                    except json.JSONDecodeError as e:
                        raise ValueError(f"回應不是有效 JSON，raw content:\n{content}") from e
                else:
                    raise ValueError(f"回應不是有效 JSON，raw content:\n{content}")

            # 期望有 results 陣列
            if not isinstance(parsed, dict) or "results" not in parsed or not isinstance(parsed["results"], list):
                raise ValueError(f"回應 JSON 結構不符預期：{json.dumps(parsed, ensure_ascii=False)}")

            return parsed, data  # (解析後, 原始完整回應)

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

    # 理論上不會到這裡
    raise RuntimeError(f"呼叫模型失敗：{last_err}")

def load_input(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input.json 需為陣列，每一筆包含 {id, description}")
    # 基本檢查
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

def main(input_path="/Users/allisonchang/Desktop/yago2026/yago2026_descriptions.json", output_path="2026_likelihood_output.json", raw_log_path="2026_likelihood_raw_responses.jsonl"):
    try:
        items = load_input(input_path)
        all_results = []

        with open(raw_log_path, "w", encoding="utf-8") as raw_fp:
            # ✅ tqdm 進度條：總批次數量
            for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="分析進度"):
                parsed, raw = call_model(batch)
                all_results.extend(parsed["results"])
                raw_fp.write(json.dumps(raw, ensure_ascii=False) + "\n")

        output = {"results": all_results}
        save_json(output_path, output)
        print(f"\n✅ 已完成，輸出：{output_path}；原始回應紀錄：{raw_log_path}")

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")

if __name__ == "__main__":
    main()
