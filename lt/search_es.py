from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Kết nối ES
es = Elasticsearch("http://localhost:9200")

# Load model embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Câu hỏi người dùng
query = input("Bạn hỏi: ")

# Chuyển câu hỏi thành vector
query_vector = model.encode(query).tolist()

# Search vector
response = es.search(
    index="traffic_law",
    body={
        "size": 3,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }
)

# In kết quả
print("\nKẾT QUẢ:\n")

for hit in response["hits"]["hits"]:
    print(hit["_source"]["content"])