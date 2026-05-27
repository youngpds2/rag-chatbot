from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# CONNECT ELASTICSEARCH
es = Elasticsearch(
    "http://localhost:9200",
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)

# EMBEDDING MODEL
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

while True:

    user_question = input("\nBạn hỏi: ")

    if user_question.lower() == "exit":
        print("Tạm biệt!")
        break

    # EMBEDDING QUESTION
    query_vector = model.encode(user_question).tolist()

    # SEARCH
    response = es.search(
    index="traffic_law",
    body={
        "size": 3,
            "query": {
                "script_score": {
                    "query": {
                        "match": {
                            "content": user_question
                        }
                    },
                    "script": {
                        "source": """
                        cosineSimilarity(
                            params.query_vector,
                            'embedding'
                        ) + 1.0
                        """,
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            }
        }
    )

    hits = response["hits"]["hits"]

    print("\n===== CONTEXT =====")

    for hit in hits:
        print(hit["_source"]["content"])
        print()

    # LẤY KẾT QUẢ TỐT NHẤT
    best_answer = hits[0]["_source"]["content"]

    print("CHATBOT:\n")
    print(best_answer)