from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# KẾT NỐI ELASTICSEARCH
es = Elasticsearch(
    "http://localhost:9200",
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)

# MODEL EMBEDDING
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# NHẬP CÂU HỎI
query = input("Bạn hỏi: ")

# EMBEDDING CÂU HỎI
query_vector = embedding_model.encode(query).tolist()

# SEARCH
response = es.search(
    index="traffic_law",
    body={
        "size": 5,
        "query": {
            "script_score": {
                "query": {
                    "match": {
                        "content": query
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

# IN KẾT QUẢ
print("\n===== KẾT QUẢ =====\n")

for hit in response["hits"]["hits"]:

    print("Score:", hit["_score"])
    print(hit["_source"]["content"])
    print("-" * 50)