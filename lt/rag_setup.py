from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("🚀 Bắt đầu chạy...")

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Độ dài text:", len(text))

print("📄 Đã đọc file")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_text(text)

print("Số đoạn:", len(chunks))

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("🧠 Đang tạo vector...")

vector_db = FAISS.from_texts(chunks, embedding)

print("💾 Đang lưu...")

vector_db.save_local("faiss_index")

print("✅ DONE")