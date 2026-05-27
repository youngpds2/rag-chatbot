from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load vector DB
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# query thử
query = "vượt đèn đỏ phạt bao nhiêu?"

docs = db.similarity_search(query, k=2)

context = "\n".join([doc.page_content for doc in docs])

print("\n📄 Context tìm được:")
print(context)

print("\n🤖 Trả lời ngắn gọn:")

# xử lý đơn giản
if "vượt đèn đỏ" in context:
    print("Vượt đèn đỏ bị phạt từ 4 đến 6 triệu đồng.")
elif "mũ bảo hiểm" in context:
    print("Không đội mũ bảo hiểm bị phạt từ 200 đến 300 nghìn đồng.")
else:
    print(context.split("\n")[0])

docs = db.similarity_search(query, k=2)

for i, doc in enumerate(docs):
    print(f"\nKết quả {i+1}:")
    print(doc.page_content)