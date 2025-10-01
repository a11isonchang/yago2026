import os
import json
import time
import requests
from dotenv import load_dotenv

# === 設定 ===
API_URL = "https://outer-medusa.genai.nchc.org.tw/v1/chat/completions"
MODEL = "gpt-oss-120b"  # 依 NCHC 實際可用模型調整
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


# === System prompt ===
SYSTEM_PROMPT = ("You are a reasoning assistant."
                 "Always follow the rules strictly and output valid JSON only."
)

# === User prompt template ===
USER_TEMPLATE = """TASK
- You will be given two inputs:
  1) A context passage (retrieved text).
  2) A true/false statement (the claim).

- Consider BOTH:
  • The context passage (RAG input)
  • Your own built-in knowledge

- Your goal is to decide whether the claim is true or false, and explain whether your judgment comes from the context passage, your own knowledge, or both.

OUTPUT FORMAT
Return your answer as a JSON object with the following fields:
{{
  "statement": "<the input statement>",
  "answer": "True" | "False",
  "reasoning": "<explain whether you relied on RAG, your own knowledge, or noticed a conflict between them>"
}}

INPUT
Context passage:
<<<
{context}
>>>

Statement:
{statement}
"""


def call_model(context: str, statement: str):
    # 把 context 和 statement 填入模板
    user_content = USER_TEMPLATE.format(context=context, statement=statement)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": TEMPERATURE,
    }

    last_err = None
    for attempt in range(1, RETRY + 1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            # 嘗試取出模型輸出文字
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                content = json.dumps(data, ensure_ascii=False)

            # 🖨️ 印出模型的原始輸出
            print("\n=== Model Output (raw content) ===")
            print(content)
            print("=================================\n")

            # 嘗試解析成 JSON
            parsed = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                print("⚠️ 輸出不是合法 JSON，請檢查模型回覆")
                parsed = None

            return parsed, content

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

def main(context: str, statement: str, output_path="RAGtest/rag_conflict_results.json"):
    try:
        parsed, raw_text = call_model(context, statement)
        result = {
            "context": context,
            "statement": statement,
            "model_output": parsed if parsed else raw_text
        }

        # 以 "追加模式" 寫入，每次新增一行
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"✅ 已追加結果到 {output_path}")
    except Exception as e:
        print(f"❌ 發生錯誤：{e}")

if __name__ == "__main__":
    # 在這裡換不同的 context 和 statement 測試
    context = (
       "Wednesday, January 7th, 2026: Local News Update\nMatt Read is no longer a member of the University of Western Ontario as of Thursday, January 1, 2026, marking a significant change in his affiliation with the institution just days ago."
    )
    statement = "Matt Read was appointed chancellor of the University of Western Ontario on January 1, 2026."
    main(context, statement)
