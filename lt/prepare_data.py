from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

with open(
    "traffic_data.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

# SPLIT CHUNKS
chunks = text.split("\n")

for i, chunk in enumerate(chunks):

    if chunk.strip() == "":
        continue

    embedding = model.encode(chunk).tolist()

    doc = {
        "content": chunk,
        "embedding": embedding,
        "source": "traffic_data.txt"
    }

    es.index(
        index="traffic_law",
        id=i,
        document=doc
    )

print("Done indexing")