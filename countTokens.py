import tiktoken
import requests
import datetime

import tiktoken

def isEnoughTokens(corpus_name):
    # 取得 tokenizer
    enc = tiktoken.encoding_for_model("gpt-4o")

    # 讀取指定檔案
    path = f"reproduce/dataset/{corpus_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 計算 token 數
    tokens = len(enc.encode(text))
    leave_tokens = 2_500_000 - tokens

    # 檢查剩餘額度
    if leave_tokens >= 0:
        return True
    else:
        return False


def countTokens():
    # === [1] 請填入你的 API 金鑰 ===
    API_KEY = "sk-FtWr6TPbLwCyq0uwGpYPQUDhIPiOEH1IcL-njESFqVT3BlbkFJxgam6aWocs3qaq-uD9J69EJ9GOrADz7BL_K4VklQwA"

    # === [2] 設定查詢時間區間 (今日 UTC 時間) ===
    today = datetime.date.today()
    start_date = today.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    # === [3] 呼叫 OpenAI Usage API ===
    url = "https://api.openai.com/v1/usage"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"start_date": start_date, "end_date": end_date}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    # === [4] 累加 token 使用量 ===
    total_prompt = 0
    total_completion = 0

    for item in data.get("data", []):
        total_prompt += item.get("n_context_tokens_total", 0)
        total_completion += item.get("n_generated_tokens_total", 0)

    total_tokens = total_prompt + total_completion

    # === [5] 免費額度上限（根據你的帳號類型）===
    FREE_LIMIT_LIGHT = 2_500_000  # 輕量模型 (gpt-4o-mini, gpt-5-mini, etc.)

    # === [6] 計算剩餘額度 ===
    remaining_light = max(FREE_LIMIT_LIGHT - total_tokens, 0)

    # === [7] 換算中文字數 (1 token ≈ 1.3 中文字) ===
    chars_light = int(remaining_light * 1.3)

    # === [8] 輸出結果 ===
    print("📅 日期：", today)
    print("📊 今日已用 Tokens：", total_tokens)
    print(f"⚡ 輕量模型剩餘：{remaining_light:,} tokens（約 {chars_light:,} 字）")
    
    return remaining_light
