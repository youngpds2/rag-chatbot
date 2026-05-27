from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# kết nối elasticsearch
es = Elasticsearch("http://localhost:9200")

# model embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# đọc file pdf
reader = PdfReader("luat_giao_thong.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

print("Độ dài text:", len(text))

# chia chunk đơn giản
chunks = text.split("\n")

# bỏ dòng rỗng
chunks = [c.strip() for c in chunks if len(c.strip()) > 20]

print("Số chunk:", len(chunks))

# lưu elasticsearch
for i, chunk in enumerate(chunks):

    embedding = model.encode(chunk).tolist()

    doc = {
        "content": chunk,
        "embedding": embedding
    }

    es.index(
        index="traffic_law",
        id=i,
        document=doc
    )

print("DONE INGEST PDF!")