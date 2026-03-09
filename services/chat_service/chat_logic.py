import atexit
import yaml

from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient

from common.env import ENV
from common.routerai_langchain import (
    RouterAIEmbeddings,
    routerai_chat_completion,
    set_routerai_api_key,
)


# Загрузка конфигурации
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Использование конфигурации
EMBED_MODEL = config["models"]["embed_model"]
LLM_MODEL = config["models"]["llm_model"]
VECTOR_STORE_PATH = config["vector_store"]["path"]

# Настройка RouterAI API ключа из .env один раз при старте
_api_key = ENV.get("ROUTERAI_API_KEY")
if _api_key:
    set_routerai_api_key(_api_key)

# Создание клиентов (общие для чата — консоль и Streamlit)
global_client = QdrantClient(path=VECTOR_STORE_PATH)
global_embeddings = RouterAIEmbeddings(model=EMBED_MODEL)


def get_questions(questions):
    """Пока простая обёртка — при необходимости можно доработать."""
    return questions


def generate_promt():
    """
    Формирует шаблон промпта для RAG-диалога.
    """
    prompt_template = PromptTemplate.from_template(
        "Текущий контекст: {context}\n Текущий вопрос: {question}\nИстория переписки: {history}"
    )

    return prompt_template


def get_answer(prompt, question, context, history=""):
    """
    Формирует текстовый промпт и вызывает чат‑модель RouterAI.
    """
    prompt_text = prompt.format(context=context, question=question, history=history)
    return routerai_chat_completion(prompt_text, model=LLM_MODEL)


def cleanup():
    global global_client
    if global_client:
        try:
            global_client.close()
        except Exception:
            pass


atexit.register(cleanup)

