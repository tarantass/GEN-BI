import sys
from pathlib import Path

import streamlit as st

# Добавляем корень проекта в sys.path, чтобы работал импорт пакета services
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.chat_service.chat_logic import (  # noqa: E402
    generate_promt,
    get_answer,
    global_client,
    global_embeddings,
)
from services.chat_service.context import generate_context  # noqa: E402
from common.routerai_langchain import set_routerai_api_key  # noqa: E402


COLLECTION_NAME = "rag_joto_test_v1"


def get_history_text(history_messages):
    """
    Превращает историю из session_state в текстовый формат,
    похожий на тот, что использовался в консольной версии.
    """
    if not history_messages:
        return ""

    parts = []
    for i, item in enumerate(history_messages, start=1):
        if item["role"] != "user":
            # сохраняем только пары вопрос/ответ, поэтому собираем по двум сообщениям
            continue
        question = item["content"]

        # ищем следующий ассистентский ответ
        answer = ""
        for j in range(i, len(history_messages)):
            if history_messages[j]["role"] == "assistant":
                answer = history_messages[j]["content"]
                break

        parts.append(
            f""" Переписка №{i}
Контекст: [история в Streamlit]
Вопрос: {question}
Ответ: {answer}"""
        )

    return "\n\n".join(parts)


def main():
    st.set_page_config(page_title="GEN BI Chat", page_icon="💬", layout="wide")
    st.title("GEN BI — чат с RAG")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "routerai_api_key" not in st.session_state:
        st.session_state.routerai_api_key = ""

    # Первый шаг: запросить ключ RouterAI прямо в чате
    if not st.session_state.routerai_api_key:
        with st.chat_message("assistant"):
            st.markdown(
                "Перед началом работы вставьте, пожалуйста, ваш **RouterAI API‑ключ** "
                "(формат `sk-...`). Ключ используется только для запросов к RouterAI "
                "и не отправляется в саму LLM‑модель."
            )
            api_key = st.text_input(
                "RouterAI API ключ",
                type="password",
                key="routerai_api_key_input",
            )

        if api_key:
            st.session_state.routerai_api_key = api_key
            set_routerai_api_key(api_key)
            # Перезапускаем, чтобы перейти к основному чату
            st.rerun()

        # Пока ключ не введён, дальше не идём
        return

    # Отображаем историю
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Поле ввода нового вопроса
    question = st.chat_input("Введите вопрос...")

    if question:
        # Показываем вопрос пользователя
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Генерирую ответ..."):
                # Получаем контекст из Qdrant
                context = generate_context(
                    question, COLLECTION_NAME, global_client, global_embeddings
                )

                # Генерируем промпт
                prompt = generate_promt()

                # История переписки в текстовом виде
                history_text = get_history_text(st.session_state.messages)

                # Получаем ответ модели
                answer = get_answer(prompt, question, context, history_text)

                st.markdown(answer)

        # Сохраняем ответ ассистента в историю
        st.session_state.messages.append(
            {"role": "assistant", "content": str(answer)}
        )


if __name__ == "__main__":
    main()

