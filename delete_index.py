from elasticsearch import Elasticsearch

# Khởi tạo kết nối
es = Elasticsearch(
    "http://localhost:9200",
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)

# Tên index bạn đang dùng trong chatbot
index_name = "traffic_law"

# XÓA INDEX
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"✅ Đã xóa index '{index_name}' thành công!")
else:
    print(f"ℹ️ Index '{index_name}' không tồn tại, không cần xóa.")