import json
import requests

from common.env import ENV


def example_chat_completion() -> dict:
    """
    Простой пример вызова чат‑модели RouterAI.
    """
    url = ENV.get("ROUTERAI_BASE_URL", "https://routerai.ru/api/v1") + "/chat/completions"
    api_key = ENV.get("ROUTERAI_API_KEY", "sk-...")

    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    data = {
        "model": "qwen/qwen3-vl-30b-a3b-thinking",
        "messages": [
            {"role": "user", "content": "1+1=? только число"},
        ],
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    result = example_chat_completion()
    print(json.dumps(result, indent=2, ensure_ascii=False))

