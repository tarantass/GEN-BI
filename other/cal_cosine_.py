import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common.routerai_langchain import RouterAIEmbeddings


def get_cosine_similarity(text1, text2):
    """Минимальная функция для получения косинусного сходства"""

    embeddings = RouterAIEmbeddings(model="sentence-transformers/all-minilm-l6-v2")

    vec1 = np.array(embeddings.embed_query(text1)).reshape(1, -1)
    vec2 = np.array(embeddings.embed_query(text2)).reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]

# Использование:
sim = get_cosine_similarity("-- 6. Вычисление среднего балла (сумма, деленная на количество). SQL", "-- 6. Вычисление среднего балла (сумма, деленная на количество)SELECT (5 + 4 + 3 + 5) / 4.0 AS average_score")
print(f"Сходство: {sim:.4f}")


# -- 6. Вычисление среднего балла (сумма, деленная на количество)
# SELECT (5 + 4 + 3 + 5) / 4.0 AS average_score

sim = get_cosine_similarity("-- 6. Вычисление среднего балла (сумма, деленная на количество). SQL", "# Первый множитель;Второй множитель;Произведение: 4;6;24")
print(f"Сходство: {sim:.4f}")

