import yaml
from qdrant_client import QdrantClient

from common.env import ENV
from services.gen_database_service.get_docs import (
    get_chunks_from_csv,
    get_chunks_from_pdf,
    get_chunks_from_sql,
    get_chunks_from_txt,
)
from services.gen_database_service.export_docs_to_Qdrant import (
    get_chunks,
    export_chunks,
)
from common.routerai_langchain import RouterAIEmbeddings, set_routerai_api_key


def main() -> None:
    # Загрузка конфигурации
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Использование конфигурации
    embed_model = config["models"]["embed_model"]
    vector_store_path = config["vector_store"]["path"]

    # Настройка RouterAI API ключа из .env
    api_key = ENV.get("ROUTERAI_API_KEY")
    if not api_key:
        raise ValueError(
            "ROUTERAI_API_KEY не задан. Укажите его в env и загрузите в окружение."
        )
    set_routerai_api_key(api_key)

    # Создание клиентов
    client = QdrantClient(path=vector_store_path)
    embeddings = RouterAIEmbeddings(model=embed_model)

    collection_name = "rag_joto_test_v1"

    # CSV (по необходимости раскомментировать)
    # doc_csv = get_chunks_from_csv("raw/raw/csv/")
    # chunk_csv = get_chunks(doc_csv)
    # export_chunks(
    #     client, collection_name, embeddings, chunk_csv, doc_type="source_csv"
    # )

    # PDF
    doc_pdf = get_chunks_from_pdf("raw/raw/pdf/")
    chunk_pdf = get_chunks(doc_pdf)
    export_chunks(
        client, collection_name, embeddings, chunk_pdf, doc_type="source_pdf"
    )

    # SQL
    doc_sql = get_chunks_from_sql("raw/raw/sql/")
    chunk_sql = get_chunks(doc_sql)
    export_chunks(
        client, collection_name, embeddings, chunk_sql, doc_type="source_sql"
    )

    # TXT
    doc_txt = get_chunks_from_txt("raw/raw/txt/")
    chunk_txt = get_chunks(doc_txt)
    export_chunks(
        client, collection_name, embeddings, chunk_txt, doc_type="source_txt"
    )


if __name__ == "__main__":
    main()

