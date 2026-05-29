import streamlit as st
import time
import os
import uuid
import io
import json
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# =========================
# KHỞI TẠO
# =========================

# Đọc API key: .env (local) hoặc st.secrets (Streamlit Cloud)
from dotenv import load_dotenv
load_dotenv()

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="Traffic Law RAG Chatbot",
    page_icon="🚦",
    layout="wide"
)

if "threads" not in st.session_state:
    st.session_state.threads = {}
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

# =========================
# HÀM BỔ TRỢ
# =========================

def generate_chat_title(question):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tóm tắt yêu cầu thành tiêu đề dưới 5 từ."},
                {"role": "user", "content": question}
            ],
            temperature=0.5
        )
        return res.choices[0].message.content.strip().replace('"', '')
    except:
        return question[:20] + "..."

@st.dialog("Xác nhận xóa")
def confirm_delete_dialog(tid):
    st.write("Bạn có chắc chắn muốn xóa cuộc hội thoại này?")
    st.info(f"**Tiêu đề:** {st.session_state.threads[tid]['title']}")
    col1, col2 = st.columns(2)
    if col1.button("Có, xóa ngay", type="primary", use_container_width=True):
        del st.session_state.threads[tid]
        if st.session_state.current_thread_id == tid:
            st.session_state.current_thread_id = None
        st.rerun()
    if col2.button("Không", use_container_width=True):
        st.rerun()

# =========================
# LOAD MODEL & DATA (cache)
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("keepitreal/vietnamese-sbert")

@st.cache_resource
def load_data_and_index():
    """Load JSON và build FAISS index trong RAM — chạy 1 lần duy nhất."""
    import faiss

    # Tìm file JSON
    json_paths = [
        "data/traffic_laws.json",
        "traffic_laws.json",
    ]
    laws = None
    for path in json_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                laws = json.load(f)
            break

    if laws is None:
        st.error("❌ Không tìm thấy file traffic_laws.json")
        st.stop()

    model = load_model()

    # Build embeddings
    texts = []
    for law in laws:
        fine = law.get("fine", "")
        if isinstance(fine, dict):
            fine = f"{fine.get('min','')} - {fine.get('max','')}"
        additional = ", ".join(law.get("additional_penalty", []))
        text = f"""
        {law.get('title', '')}
        {' '.join(law.get('keywords', []))}
        {law.get('content', '')}
        {fine}
        {additional}
        """.lower().strip()
        texts.append(text)

    embeddings = model.encode(texts, show_progress_bar=False).astype("float32")

    # Normalize để dùng cosine similarity
    faiss.normalize_L2(embeddings)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sau normalize
    index.add(embeddings)

    # Chuẩn hóa fine thành string cho mỗi law
    for law in laws:
        fine = law.get("fine", "")
        if isinstance(fine, dict):
            law["fine"] = f"{fine.get('min','')} - {fine.get('max','')}"
        law["additional_penalty_str"] = ", ".join(law.get("additional_penalty", []))

    return index, laws, model

model = load_model()
faiss_index, laws_data, _ = load_data_and_index()

# =========================
# HÀM SEARCH FAISS
# =========================

def search_faiss(query: str, top_k: int = 5):
    import faiss
    vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(vec)
    scores, indices = faiss_index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({"score": float(score), "source": laws_data[idx]})
    return results

def search_keyword(query: str, top_k: int = 5):
    """BM25-style keyword search đơn giản."""
    query_words = set(query.lower().split())
    scored = []
    for law in laws_data:
        text = f"{law.get('title','')} {' '.join(law.get('keywords',[]))} {law.get('content','')}".lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, law))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "source": law} for s, law in scored[:top_k]]

def hybrid_search(query: str, original: str, top_k: int = 5):
    """Kết hợp vector + keyword, fallback về keyword nếu vector không đủ."""
    vector_hits = search_faiss(query, top_k)
    # Lọc score thấp (< 0.3 thường không liên quan)
    vector_hits = [h for h in vector_hits if h["score"] >= 0.3]

    if vector_hits:
        return vector_hits

    # Fallback 1: keyword với query expanded
    kw_hits = search_keyword(query, top_k)
    if kw_hits:
        return kw_hits

    # Fallback 2: keyword với câu gốc
    return search_keyword(original, top_k)

# =========================
# VOICE (WHISPER)
# =========================

def transcribe_audio(audio_bytes: bytes) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="vi"
    )
    return transcript.text.strip()

# =========================
# CSS
# =========================

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0b1120;
    color: white;
    font-family: 'Inter', sans-serif;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}
.sub-title {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
    font-size: 16px;
}
.chat-user {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 14px 0;
    margin-left: auto;
    width: fit-content;
    max-width: 75%;
    color: white;
    line-height: 1.7;
    font-size: 15px;
    box-shadow: 0 4px 18px rgba(37,99,235,0.25);
}
.chat-bot {
    background: #1e293b;
    padding: 14px;
    border-radius: 16px;
    margin-bottom: 12px;
    color: white;
    line-height: 1.7;
    font-size: 15px;
    border-left: 4px solid #22c55e;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}
.stButton button {
    border-radius: 10px;
    border: none;
    background: #1e293b;
    color: white;
}
.stButton button:hover {
    background: #334155;
    color: white;
}
.stChatInput input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid #334155 !important;
}
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }

.main .block-container { padding-bottom: 90px !important; }

div[data-testid="stBottom"] {
    position: sticky !important;
    bottom: 0 !important;
    z-index: 999 !important;
    background-color: #0b1120 !important;
    border-top: 1px solid #1f2937 !important;
    padding: 10px 1.5rem 14px 1.5rem !important;
    width: 100% !important;
}

div[data-testid="stAudioInput"] {
    position: fixed !important;
    bottom: 70px !important;
    right: 50px !important;
    width: 50px !important;
    height: 48px !important;
    overflow: hidden !important;
    z-index: 1002 !important;
    background: transparent !important;
    
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
div[data-testid="stAudioInput"] > label { display: none !important; }
div[data-testid="stAudioInput"] > div { overflow: hidden !important; height: 38px !important; }
div[data-testid="stAudioInput"] button {
    width: 50px !important;
    height: 48px !important;
    border-radius: 8px !important;
    background: #2b313e !important;
    border: none !important;
    color: white !important;
    font-size: 16px !important;
    padding: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    position: relative !important;
    z-index: 1003 !important;
}
div[data-testid="stAudioInput"] button:hover {
    background: #2b313e  !important;
    color: white !important;
}

div[data-testid="stChatInputContainer"] {
    padding-right: 50px !important;
    margin-right: 0px !important;
}

div[data-testid="stChatInputContainer"] > div {
    padding-right: 50px !important;
}

textarea[data-testid="stChatInputTextArea"] {
    padding-right: 50px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("📜 Lịch sử")
    if st.button("+ Đoạn chat mới", use_container_width=True):
        st.session_state.current_thread_id = None
        st.rerun()
    st.divider()
    for tid, tdata in list(st.session_state.threads.items()):
        cols = st.columns([0.82, 0.18])
        with cols[0]:
            if st.button(tdata['title'], key=f"select_{tid}", use_container_width=True):
                st.session_state.current_thread_id = tid
                st.rerun()
        with cols[1]:
            if st.button("✕", key=f"del_{tid}"):
                confirm_delete_dialog(tid)

# =========================
# MAIN UI
# =========================

st.markdown("""
<div class="main-title">🚦 Traffic Law AI</div>
<div class="sub-title">Trợ lý AI tư vấn luật giao thông Việt Nam</div>
""", unsafe_allow_html=True)

if st.session_state.current_thread_id in st.session_state.threads:
    for msg in st.session_state.threads[st.session_state.current_thread_id]["messages"]:
        role_class = "chat-user" if msg["role"] == "user" else "chat-bot"
        st.markdown(
            f'<div class="{role_class}">{msg["content"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

# =========================
# VOICE INPUT
# =========================

typed_question = st.chat_input("Hỏi về mức phạt, biển báo...")
audio_input = st.audio_input(" ", label_visibility="collapsed", key="mic_hidden")

voice_question = ""
if audio_input is not None:
    audio_bytes = audio_input.getvalue()
    audio_id = hash(audio_bytes)
    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id
        with st.spinner("🎧 Đang nhận dạng giọng nói..."):
            try:
                voice_question = transcribe_audio(audio_bytes)
                st.toast(f"🎤 {voice_question}", icon="✅")
            except Exception as e:
                st.error(f"❌ Lỗi Whisper: {e}")

user_question = voice_question if voice_question else typed_question

# =========================
# XỬ LÝ CÂU HỎI
# =========================

if user_question:
    if st.session_state.current_thread_id is None:
        new_id = str(uuid.uuid4())
        st.session_state.threads[new_id] = {"title": "Cuộc trò chuyện mới...", "messages": []}
        st.session_state.current_thread_id = new_id

    st.markdown(f'<div class="chat-user">{user_question}</div>', unsafe_allow_html=True)

    with st.spinner("🔍 Đang tra cứu luật..."):

        # Lịch sử gần nhất
        recent_messages = st.session_state.threads[
            st.session_state.current_thread_id
        ]["messages"][-4:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent_messages
        )

        # Query expansion
        try:
            expansion_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"""Nhiệm vụ: Liệt kê các từ khóa tìm kiếm liên quan đến câu hỏi về luật giao thông.

Lịch sử hội thoại (nếu có):
{history_text}

Câu hỏi: {user_question}

Chỉ trả về các từ khóa cách nhau bằng dấu cách. Không giải thích. Không JSON. Tối đa 15 từ. Viết thường có dấu tiếng Việt.

Ví dụ đầu ra: vượt đèn đỏ mức phạt xe máy ô tô tín hiệu giao thông"""}],
                temperature=0,
                max_tokens=60
            )
            expanded = expansion_res.choices[0].message.content.strip().lower()
            if expanded.startswith("{") or expanded.startswith("["):
                expanded = user_question.lower()
        except:
            expanded = user_question.lower()

        search_query = (
            expanded
            .replace("ô tô", "oto")
            .replace("xe hơi", "oto")
            .replace("giấy phép lái xe", "gplx")
            .replace("bằng lái", "gplx")
        )

        # FAISS search
        hits = hybrid_search(search_query, user_question)

        if not hits:
            answer = "⚠️ Tôi chỉ tư vấn về luật giao thông Việt Nam. Bạn có câu hỏi về vi phạm, mức phạt hoặc biển báo không?"
        else:
            contexts = []
            for h in hits:
                src = h["source"]
                legal = src.get("legal_basis", {})
                contexts.append(f"""
Tiêu đề: {src.get('title', '')}
Loại phương tiện: {src.get('vehicle', '')}
Mức phạt: {src.get('fine', '')}
Hình thức bổ sung: {src.get('additional_penalty_str', '')}
Nội dung: {src.get('content', '')}
Căn cứ: {legal.get('point','')} {legal.get('clause','')} {legal.get('article','')} {src.get('law_reference','')}
""")
            top_contexts = "\n---\n".join(contexts)

            sys_prompt = f"""
Bạn là chuyên gia tư vấn luật giao thông Việt Nam, giải thích rõ ràng như một người bạn am hiểu pháp luật.

QUY TẮC:
- Dựa trên dữ liệu cung cấp, không bịa mức phạt hoặc điều khoản.
- Nếu dữ liệu thiếu, dùng kiến thức luật giao thông Việt Nam hiện hành để bổ sung hợp lý.
- Luôn phân biệt các trường hợp khác nhau nếu có.
- Xuống dòng rõ ràng, dễ đọc trên chat.
- Không dùng markdown (#, **, ---).

CẤU TRÚC TRẢ LỜI:

Trả lời nhanh:
[1-2 câu tóm tắt]

Chi tiết mức phạt:
- [Loại xe / trường hợp]: [mức phạt]

Hình thức bổ sung (nếu có):
- ...

Phân biệt các trường hợp (nếu có):
- [Tình huống A]: ...
- [Tình huống B]: ...

Căn cứ pháp lý:
- Điểm ... Khoản ... Điều ... Nghị định ...

Bạn có muốn biết thêm:
- [gợi ý 1]
- [gợi ý 2]
- [gợi ý 3]

DỮ LIỆU:
{top_contexts}
"""

            messages_for_gpt = [{"role": "system", "content": sys_prompt}]
            for m in recent_messages:
                messages_for_gpt.append({"role": m["role"], "content": m["content"]})
            messages_for_gpt.append({"role": "user", "content": user_question})

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_for_gpt,
                temperature=0
            )
            answer = completion.choices[0].message.content

    # Hiển thị streaming
    resp_placeholder = st.empty()
    full_resp = ""
    for char in answer:
        full_resp += char
        resp_placeholder.markdown(
            f'<div class="chat-bot">{full_resp.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.01)

    # Lưu lịch sử
    st.session_state.threads[st.session_state.current_thread_id]["messages"].append(
        {"role": "user", "content": user_question}
    )
    st.session_state.threads[st.session_state.current_thread_id]["messages"].append(
        {"role": "assistant", "content": answer}
    )

    if len(st.session_state.threads[st.session_state.current_thread_id]["messages"]) <= 2:
        st.session_state.threads[st.session_state.current_thread_id]["title"] = generate_chat_title(user_question)
        st.rerun()