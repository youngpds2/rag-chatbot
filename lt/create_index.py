from elasticsearch import Elasticsearch

es = Elasticsearch(
    "http://localhost:9200",
    verify_certs=False
)

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

try:
    es.indices.delete(index="traffic_law")
except:
    pass

es.indices.create(index="traffic_law", body=mapping)

print("Index created!")