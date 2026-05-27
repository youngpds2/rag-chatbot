from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Kết nối Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Model embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Dữ liệu luật giao thông
docs = [
    "Vượt đèn đỏ bị phạt từ 4 đến 6 triệu đồng.",
    "Không đội mũ bảo hiểm bị phạt từ 200 đến 300 nghìn đồng.",
    "Đi ngược chiều bị phạt từ 1 đến 2 triệu đồng.",
    "Không mang giấy phép lái xe bị phạt từ 100 đến 200 nghìn đồng.",
    "Sử dụng điện thoại khi lái xe bị phạt từ 800 nghìn đến 1 triệu đồng."
]

# Đưa dữ liệu vào Elasticsearch
for i, doc in enumerate(docs):

    embedding = model.encode(doc).tolist()

    es.index(
        index="traffic_law",
        id=i,
        document={
            "content": doc,
            "embedding": embedding
        }
    )

print("Done ingest!")