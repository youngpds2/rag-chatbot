import json
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# =========================
# ELASTICSEARCH
# =========================

es = Elasticsearch("http://localhost:9200")

# =========================
# MODEL
# =========================

model = SentenceTransformer("keepitreal/vietnamese-sbert")

# =========================
# LOAD JSON
# =========================

with open("data/traffic_laws.json", "r", encoding="utf-8") as f:
    laws = json.load(f)

# =========================
# INGEST
# =========================

for law in laws:

    # Xử lý fine: {"min": "4.000.000 đồng", "max": "6.000.000 đồng"} → chuỗi đẹp
    fine_raw = law.get("fine", "")
    if isinstance(fine_raw, dict):
        fine_str = f"{fine_raw.get('min', '')} - {fine_raw.get('max', '')}"
    else:
        fine_str = str(fine_raw)

    # Xử lý additional_penalty: list → chuỗi
    additional = law.get("additional_penalty", [])
    additional_str = ", ".join(additional) if additional else ""

    # Xử lý legal_basis
    legal_basis = law.get("legal_basis", {})
    article   = legal_basis.get("article", "")
    clause    = legal_basis.get("clause", "")
    point     = legal_basis.get("point", "")

    # law_reference nằm ở root, KHÔNG phải trong legal_basis
    law_reference = law.get("law_reference", "")

    # TEXT CHO EMBEDDING — thêm additional_penalty vào để tìm kiếm tốt hơn
    text_for_embedding = f"""
    {law['title']}
    {' '.join(law.get('keywords', []))}
    {law.get('content', '')}
    {fine_str}
    {additional_str}
    """.lower()

    # VECTOR
    embedding = model.encode(text_for_embedding).tolist()

    # DOCUMENT
    doc = {
        "id":                law.get("id", ""),
        "type":              law.get("type", ""),
        "title":             law.get("title", ""),
        "vehicle":           law.get("vehicle", ""),
        "keywords":          law.get("keywords", []),
        "fine":              fine_str,           # ✅ chuỗi thay vì dict
        "additional_penalty": additional_str,    # ✅ field mới
        "content":           law.get("content", ""),
        "article":           article,
        "clause":            clause,
        "point":             point,
        "law_reference":     law_reference,      # ✅ lấy đúng field
        "embedding":         embedding
    }

    es.index(index="traffic_law", document=doc)
    print(f"✅ Indexed: {law['title']} | Phạt: {fine_str} | Bổ sung: {additional_str or 'không có'}")

print("\n🚀 DONE INGEST!")