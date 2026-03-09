from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings


ROUTERAI_BASE_URL = "https://routerai.ru/api/v1"
ROUTERAI_API_KEY: Optional[str] = None


def set_routerai_api_key(key: str) -> None:
    """
    Устанавливает API-ключ RouterAI во время выполнения (например, из UI Streamlit).
    """
    global ROUTERAI_API_KEY
    ROUTERAI_API_KEY = key


def _get_headers() -> dict:
    if not ROUTERAI_API_KEY:
        raise ValueError(
            "ROUTERAI_API_KEY is not set. Please provide it via set_routerai_api_key()."
        )

    return {
        "Authorization": ROUTERAI_API_KEY,
        "Content-Type": "application/json",
    }


class RouterAIEmbeddings(Embeddings):
    """
    LangChain совместимые эмбеддинги через RouterAI API.
    """

    def __init__(self, model: str):
        self.model = model

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
        }

        response = requests.post(
            f"{ROUTERAI_BASE_URL}/embeddings",
            headers=_get_headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        vectors = [item["embedding"] for item in data["data"]]
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def routerai_chat_completion(prompt: str, model: str) -> str:
    """
    Вызов чат‑модели RouterAI в стиле OpenAI Chat Completions.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(
        f"{ROUTERAI_BASE_URL}/chat/completions",
        headers=_get_headers(),
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]

