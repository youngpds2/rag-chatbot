import streamlit as st
import time
import os
import uuid
import io
import json
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# =========================
# KHỞI TẠO
# =========================

from dotenv import load_dotenv
load_dotenv()

try:
    api_key = st.secrets["OPENAI_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY", "")
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

client = OpenAI(api_key=api_key)
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    supabase = None

st.set_page_config(
    page_title="Traffic Law RAG Chatbot",
    page_icon="🚦",
    layout="wide"
)

if "user" not in st.session_state:
    st.session_state.user = None
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# =========================
# HÀM BỔ TRỢ
# =========================

# --- AUTH ---
def login(email, password):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return res.user, None
    except Exception as e:
        return None, str(e)

def register(email, password):
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        return res.user, None
    except Exception as e:
        return None, str(e)

def logout():
    supabase.auth.sign_out()
    st.session_state.user = None
    st.session_state.threads = {}
    st.session_state.current_thread_id = None
    st.rerun()

# --- SUPABASE HISTORY ---
def load_threads_from_db(user_id):
    """Load toàn bộ conversations + messages của user từ Supabase."""
    threads = {}
    try:
        convs = supabase.table("conversations").select("*").eq("user_id", user_id).order("created_at").execute()
        for conv in convs.data:
            msgs_res = supabase.table("messages").select("*").eq("conversation_id", conv["id"]).order("created_at").execute()
            threads[conv["id"]] = {
                "title": conv["title"],
                "messages": [{"role": m["role"], "content": m["content"]} for m in msgs_res.data]
            }
    except Exception as e:
        st.error(f"Lỗi load lịch sử: {e}")
    st.sidebar.caption(f"📦 Loaded {len(threads)} threads")
    return threads

def save_new_thread(user_id, thread_id, title):
    try:
        supabase.table("conversations").insert({"id": thread_id, "user_id": user_id, "title": title}).execute()
    except Exception as e:
        pass

def update_thread_title(thread_id, title):
    try:
        supabase.table("conversations").update({"title": title}).eq("id", thread_id).execute()
    except Exception as e:
        pass

def save_message(thread_id, role, content):
    try:
        res = supabase.table("messages").insert({"conversation_id": thread_id, "role": role, "content": content}).execute()
        return True
    except Exception as e:
        st.sidebar.error(f"Lỗi lưu message: {e}")
        return False

def delete_thread_from_db(thread_id):
    try:
        supabase.table("conversations").delete().eq("id", thread_id).execute()
    except Exception as e:
        pass

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
        delete_thread_from_db(tid)
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

# =========================
# KIỂM TRA ĐĂNG NHẬP
# =========================

if st.session_state.user is None:
    st.markdown("""
    <style>
    html, body, [class*="css"] { background-color: #0b1120; color: white; font-family: 'Inter', sans-serif; }
    div[data-testid="stTabs"] button { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h2 style='text-align:center;margin-bottom:24px;'>🚦 Traffic Law AI</h2>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])

        with tab1:
            email = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input("Mật khẩu", type="password", key="login_pass")
            if st.button("Đăng nhập", use_container_width=True, type="primary", key="btn_login"):
                if email and password:
                    user, err = login(email, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.threads = load_threads_from_db(user.id)
                        st.rerun()
                    else:
                        st.error(f"❌ Email hoặc mật khẩu không đúng")
                else:
                    st.warning("Vui lòng nhập đầy đủ thông tin")

        with tab2:
            reg_email = st.text_input("Email", key="reg_email", placeholder="you@example.com")
            reg_pass = st.text_input("Mật khẩu", type="password", key="reg_pass", placeholder="Ít nhất 6 ký tự")
            reg_pass2 = st.text_input("Nhập lại mật khẩu", type="password", key="reg_pass2")
            if st.button("Đăng ký", use_container_width=True, type="primary", key="btn_reg"):
                if reg_email and reg_pass and reg_pass2:
                    if reg_pass != reg_pass2:
                        st.error("❌ Mật khẩu không khớp")
                    elif len(reg_pass) < 6:
                        st.error("❌ Mật khẩu phải ít nhất 6 ký tự")
                    else:
                        user, err = register(reg_email, reg_pass)
                        if user:
                            st.success("✅ Đăng ký thành công! Vui lòng đăng nhập.")
                        else:
                            st.error(f"❌ {err}")
                else:
                    st.warning("Vui lòng nhập đầy đủ thông tin")
    st.stop()

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
    margin-bottom: 4px;
    color: white;
    line-height: 1.7;
    font-size: 15px;
    border-left: 4px solid #22c55e;
    position: relative;
}
.chat-bot-wrapper {
    position: relative;
    margin-bottom: 12px;
}
.copy-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: #334155;
    border: none;
    border-radius: 6px;
    color: #94a3b8;
    font-size: 12px;
    padding: 4px 8px;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.2s;
    z-index: 10;
}
.chat-bot-wrapper:hover .copy-btn {
    opacity: 1;
}
.copy-btn:hover {
    background: #475569;
    color: white;
}
.copy-btn.copied {
    color: #22c55e;
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

/* Nút X xóa đoạn chat */
section[data-testid="stSidebar"] .stButton button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    line-height: 1 !important;
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
    right: 45px !important;
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
    right: 13px !important;
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
    padding-right: 0px !important;
    margin-right: 0px !important;
}

div[data-testid="stChatInputContainer"] > div {
    padding-right: 0px !important;
}

textarea[data-testid="stChatInputTextArea"] {
    padding-right: 0px !important;
}

/* ===== RESPONSIVE ===== */
 
/* Desktop base styles - đảm bảo kích thước đúng */
.main-title { font-size: 42px !important; }
.sub-title { font-size: 16px !important; }
.chat-user { max-width: 75% !important; font-size: 15px !important; }
.chat-bot { font-size: 15px !important; }
 
/* Mobile */
@media screen and (max-width: 768px) {
    .main-title { font-size: 24px !important; margin-bottom: 2px !important; }
    .sub-title { font-size: 13px !important; margin-bottom: 12px !important; }
    .chat-user { max-width: 92% !important; font-size: 13px !important; padding: 10px 12px !important; }
    .chat-bot { font-size: 13px !important; padding: 10px 12px !important; }
    .main .block-container { padding: 0.5rem 0.5rem 100px 0.5rem !important; }
    section[data-testid="stSidebar"] { min-width: 160px !important; max-width: 180px !important; }
    div[data-testid="stBottom"] { padding: 8px 3rem 12px 0.5rem !important; }
    div[data-testid="stAudioInput"] { 
        bottom: 72px !important; 
        right: 10px !important;
        width: 44px !important;
        height: 44px !important;
    }
    div[data-testid="stAudioInput"] button {
        width: 44px !important;
        height: 44px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("📜 Lịch sử")
    # Hiện email user
    if st.session_state.user:
        st.caption(f"👤 {st.session_state.user.email}")
    if st.button("+ Đoạn chat mới", use_container_width=True):
        st.session_state.current_thread_id = None
        st.session_state.suggestions = []
        st.rerun()
    st.divider()
    for tid, tdata in list(st.session_state.threads.items()):
        cols = st.columns([0.82, 0.18])
        with cols[0]:
            if st.button(tdata['title'], key=f"select_{tid}", use_container_width=True):
                st.session_state.current_thread_id = tid
                st.session_state.suggestions = []
                st.rerun()
        with cols[1]:
            if st.button("✕", key=f"del_{tid}"):
                confirm_delete_dialog(tid)
    st.divider()
    if st.button("🚪 Đăng xuất", use_container_width=True):
        logout()

# =========================
# MAIN UI
# =========================

st.markdown("""
<div class="main-title">🚦 Traffic Law AI</div>
<div class="sub-title">Trợ lý AI tư vấn luật giao thông Việt Nam</div>
""", unsafe_allow_html=True)

if st.session_state.current_thread_id in st.session_state.threads:
    msgs = st.session_state.threads[st.session_state.current_thread_id]["messages"]
    for i, msg in enumerate(msgs):
        role_class = "chat-user" if msg["role"] == "user" else "chat-bot"
        # Cắt phần gợi ý khỏi nội dung hiển thị
        display_content = msg["content"]
        if msg["role"] == "assistant":
            for marker in ["Bạn có muốn biết thêm:", "Bạn có muốn biết thêm :", "bạn có muốn biết thêm:"]:
                if marker.lower() in display_content.lower():
                    idx = display_content.lower().find(marker.lower())
                    display_content = display_content[:idx].strip()
                    break
        if msg["role"] == "assistant":
            st.markdown(
                f'<div class="{role_class}">{display_content.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )
            # Nút copy dùng st.components để có quyền clipboard
            import streamlit.components.v1 as components
            escaped = display_content.replace("`", "'").replace("\\", "\\\\").replace("\n", "\\n")
            components.html(f"""
<button onclick="navigator.clipboard.writeText(`{escaped}`).then(()=>{{
    this.innerHTML='✓ Đã copy';
    this.style.color='#22c55e';
    setTimeout(()=>{{this.innerHTML='📋 Copy';this.style.color='#94a3b8'}},2000)
}})" style="
    background:#1e293b;border:1px solid #334155;border-radius:6px;
    color:#94a3b8;font-size:12px;padding:4px 10px;cursor:pointer;
    font-family:Inter,sans-serif;margin-bottom:8px;
">📋 Copy</button>
""", height=36)
        else:
            st.markdown(
                f'<div class="{role_class}">{display_content.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )
    # Hiện nút gợi ý sau tin nhắn cuối
    if st.session_state.suggestions:
        st.markdown("<div style='margin-top:6px; color:#94a3b8; font-size:12px;'>💡 Bạn muốn biết thêm:</div>", unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.suggestions))
        for i, sug in enumerate(st.session_state.suggestions):
            with cols[i]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.suggested_question = sug
                    st.session_state.suggestions = []
                    st.rerun()

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

if st.session_state.suggested_question:
    user_question = st.session_state.suggested_question
    st.session_state.suggested_question = None
else:
    user_question = voice_question if voice_question else typed_question

# =========================
# XỬ LÝ CÂU HỎI
# =========================

if user_question:
    if st.session_state.current_thread_id is None:
        new_id = str(uuid.uuid4())
        st.session_state.threads[new_id] = {"title": "Cuộc trò chuyện mới...", "messages": []}
        st.session_state.current_thread_id = new_id
        if st.session_state.user:
            save_new_thread(st.session_state.user.id, new_id, "Cuộc trò chuyện mới...")

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
- Điểm ... Khoản ... Điều ... [Tên đầy đủ Nghị định, ví dụ: Nghị định 168/2024/NĐ-CP]
- Ghi rõ số nghị định, không viết tắt hay để trống.

📌 Nguồn: [Tên Nghị định đầy đủ] — hiệu lực từ [ngày hiệu lực nếu biết]

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

    # Parse gợi ý và cắt khỏi answer hiển thị
    suggestions = []
    display_answer = answer
    for marker in ["Bạn có muốn biết thêm:", "Bạn có muốn biết thêm :", "bạn có muốn biết thêm:"]:
        if marker.lower() in answer.lower():
            idx = answer.lower().find(marker.lower())
            display_answer = answer[:idx].strip()
            for line in answer[idx:].split("\n"):
                line = line.strip().lstrip("-").strip()
                if line and marker.lower() not in line.lower() and len(line) > 5:
                    suggestions.append(line)
            break

    # Hiển thị streaming (không có phần gợi ý)
    resp_placeholder = st.empty()
    full_resp = ""
    for char in display_answer:
        full_resp += char
        resp_placeholder.markdown(
            f'<div class="chat-bot">{full_resp.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.01)

    # Scroll to bottom
    st.markdown("""
<script>
window.parent.document.querySelector('[data-testid="stAppViewContainer"]').scrollTo(
    0,
    window.parent.document.querySelector('[data-testid="stAppViewContainer"]').scrollHeight
);
</script>
""", unsafe_allow_html=True)

    # Lưu suggestions vào session_state để hiện sau rerun
    st.session_state.suggestions = suggestions[:3]

    # Lưu lịch sử local
    st.session_state.threads[st.session_state.current_thread_id]["messages"].append(
        {"role": "user", "content": user_question}
    )
    st.session_state.threads[st.session_state.current_thread_id]["messages"].append(
        {"role": "assistant", "content": answer}
    )

    # Lưu vào Supabase
    if st.session_state.user:
        r1 = save_message(st.session_state.current_thread_id, "user", user_question)
        r2 = save_message(st.session_state.current_thread_id, "assistant", answer)

    if len(st.session_state.threads[st.session_state.current_thread_id]["messages"]) <= 2:
        title = generate_chat_title(user_question)
        st.session_state.threads[st.session_state.current_thread_id]["title"] = title
        if st.session_state.user:
            update_thread_title(st.session_state.current_thread_id, title)
    st.rerun()