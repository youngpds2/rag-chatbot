import os
from pypdf import PdfReader
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# ELASTICSEARCH
# =========================

es = Elasticsearch(
    "http://localhost:9200",
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)

# =========================
# EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# TEXT SPLITTER
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)

# =========================
# PDF FOLDER
# =========================

PDF_FOLDER = "pdfs"

# =========================
# ALL CHUNKS
# =========================

all_chunks = []

# =========================
# READ PDF
# =========================

for filename in os.listdir(PDF_FOLDER):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(
            PDF_FOLDER,
            filename
        )

        print(f"\n📄 Đang đọc: {filename}")

        try:

            reader = PdfReader(pdf_path)

            text = ""

            # READ EACH PAGE
            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    page_text = page_text.replace("\n", " ")

                    page_text = " ".join(page_text.split())

                    text += page_text + " "

            # CHECK EMPTY
            if len(text.strip()) == 0:

                print("❌ PDF không có text")
                continue

            print("🧠 Độ dài text:", len(text))

            # =========================
            # SPLIT CHUNKS
            # =========================

            chunks = text_splitter.split_text(text)

            print("✂️ Số chunk:", len(chunks))

            # SAVE CHUNKS
            for chunk in chunks:

                chunk = chunk.strip()

                # BỎ chunk quá ngắn
                if len(chunk) < 30:
                    continue

                all_chunks.append({
                    "content": chunk,
                    "source": filename
                })

        except Exception as e:

            print(f"❌ Lỗi đọc PDF {filename}: {e}")

# =========================
# TOTAL CHUNKS
# =========================

print(f"\n🔥 Tổng chunk: {len(all_chunks)}")

# =========================
# INSERT ELASTICSEARCH
# =========================

for i, chunk_data in enumerate(all_chunks):

    try:

        # EMBEDDING
        embedding = embedding_model.encode(
            chunk_data["content"]
        ).tolist()

        # DOCUMENT
        doc = {
            "content": chunk_data["content"],
            "source": chunk_data["source"],
            "embedding": embedding
        }

        # INDEX
        es.index(
            index="traffic_law",
            document=doc
        )

        print(f"✅ Indexed chunk {i+1}")

    except Exception as e:

        print(f"❌ Lỗi chunk {i+1}: {e}")

print("\n🚀 DONE MULTI PDF INGEST!")