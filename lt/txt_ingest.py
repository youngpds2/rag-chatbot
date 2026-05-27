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

# ĐỌC FILE TXT
with open("traffic_data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# LỌC DÒNG RỖNG
lines = [line.strip() for line in lines if line.strip()]

print("Tổng dữ liệu:", len(lines))

# INSERT TỪNG DÒNG
for line in lines:

    embedding = embedding_model.encode(line).tolist()

    doc = {
    "content": line,
    "source": "traffic_data.txt",
    "embedding": embedding
}
    es.index(
        index="traffic_law",
        document=doc
    )

print("DONE INGEST TXT!")