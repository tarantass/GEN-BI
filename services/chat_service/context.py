from langchain_qdrant import QdrantVectorStore


def generate_context(question, qdrant_collection, client, embeddings):
    """
    Получает релевантный контекст из Qdrant по вопросу.
    """
    select_vectorstore = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection,
        embedding=embeddings,
    )

    relevant_docs = select_vectorstore.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    return context

