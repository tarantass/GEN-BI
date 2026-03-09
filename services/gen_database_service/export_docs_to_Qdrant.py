from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient, models


#3 Формируются чанки
def get_chunks(docs):
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    return chunks


#4 Чанки индексируются в векторном хранилище
def export_chunks(global_client, QDRANT_COLLECTION, global_embeddings, chunks, doc_type="general"):
    """
    Экспортирует чанки в Qdrant с указанием типа документа
    """
    
    # Добавляем doc_type в метаданные каждого чанка
    for chunk in chunks:
        # Создаем metadata если его нет
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            chunk.metadata = {}
        
        # Добавляем тип документа в метаданные
        chunk.metadata["doc_type"] = doc_type
    
        # ПРОВЕРЯЕМ И СОЗДАЕМ КОЛЛЕКЦИЮ
    try:
        global_client.get_collection(QDRANT_COLLECTION)
        print(f"📚 Коллекция '{QDRANT_COLLECTION}' уже существует")
    except ValueError:
        print(f"🆕 Создаем коллекцию '{QDRANT_COLLECTION}'...")
        
        VECTOR_SIZE = len(global_embeddings.embed_query("test"))

        global_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "size": VECTOR_SIZE,
                "distance": models.Distance.COSINE
            }
        )

    # Создаем векторное хранилище
    insert_vectorstore = QdrantVectorStore(
        client=global_client,
        collection_name=QDRANT_COLLECTION,
        embedding=global_embeddings
    )
    
    # Добавляем документы
    insert_vectorstore.add_documents(chunks)
    
    print(f"✅ Добавлено {len(chunks)} чанков с типом '{doc_type}' в коллекцию {QDRANT_COLLECTION}")
    
    return None