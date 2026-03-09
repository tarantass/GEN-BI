import json
import requests

from common.env import ENV


def example_embeddings() -> dict:
    """
    Простой пример вызова эмбеддингов RouterAI.
    """
    url = ENV.get("ROUTERAI_BASE_URL", "https://routerai.ru/api/v1") + "/embeddings"
    api_key = ENV.get("ROUTERAI_API_KEY", "sk-...")

    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    data = {
        "model": "sentence-transformers/all-minilm-l6-v2",
        "input": "Your text to embed goes here",
        "encoding_format": "float",
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    result = example_embeddings()
    print(json.dumps(result, indent=2, ensure_ascii=False))

