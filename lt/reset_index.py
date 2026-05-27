from elasticsearch import Elasticsearch

# KẾT NỐI ELASTICSEARCH
es = Elasticsearch(
    "http://localhost:9200",
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)

# XÓA INDEX CŨ
es.indices.delete(
    index="traffic_law",
    ignore_unavailable=True
)

print("Deleted old index!")

# TẠO INDEX MỚI
mapping = {
    "mappings": {
        "properties": {
            "content": {
                "type": "text"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 384
            }
        }
    }
}

es.indices.create(
    index="traffic_law",
    body=mapping
)

print("Created new clean index!")