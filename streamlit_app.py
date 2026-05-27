import streamlit as st
import time
import os
import uuid
import io
from dotenv import load_dotenv
from openai import OpenAI
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# =========================
# KHỞI TẠO & CẤU HÌNH
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="Traffic Law RAG Chatbot",
    page_icon="🚦",
    layout="wide"
)

# Khởi tạo Session State
if "threads" not in st.session_state:
    st.session_state.threads = {}  
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None

# --- HÀM BỔ TRỢ ---
def generate_chat_title(question):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Tóm tắt yêu cầu thành tiêu đề dưới 5 từ."},
                      {"role": "user", "content": question}],
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
# MODERN UI
# =========================

st.markdown("""
<style>

/* ===== GLOBAL ===== */

html, body, [class*="css"] {
    background-color: #0b1120;
    color: white;
    font-family: 'Inter', sans-serif;
}

/* ===== MAIN TITLE ===== */

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

/* ===== CHAT BUBBLE ===== */

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


/* ===== SIDEBAR ===== */

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

/* ===== BUTTON ===== */

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

/* ===== CHAT INPUT ===== */

.stChatInput input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid #334155 !important;
}

/* ===== SCROLLBAR ===== */

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }

/* ===== STICKY CHAT INPUT ===== */

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

/* Đưa st.audio_input ra góc phải thanh chat, chỉ giữ button */
div[data-testid="stAudioInput"] {
    position: fixed !important;
    bottom: 70px !important;
    right: 80px !important;
    width: 40px !important;
    height: 40px !important;
    overflow: hidden !important;
    z-index: 1002 !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Ẩn label, waveform, timer — chỉ thấy nút */
div[data-testid="stAudioInput"] > label { display: none !important; }
div[data-testid="stAudioInput"] > div { overflow: hidden !important; height: 40px !important; }

/* Nút mic style */
div[data-testid="stAudioInput"] button {
    width: 40px !important;
    height: 40px !important;
    border-radius: 50% !important;
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    color: #94a3b8 !important;
    font-size: 17px !important;
    padding: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    position: relative !important;
    z-index: 1003 !important;
}
div[data-testid="stAudioInput"] button:hover {
    border-color: #3b82f6 !important;
    color: #3b82f6 !important;
    background: #1e3a5f !important;
}

</style>
""", unsafe_allow_html=True)


# =========================
# MODELS & DB
# =========================
es = Elasticsearch("http://localhost:9200")

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "keepitreal/vietnamese-sbert"
    )

model = load_model()

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
# GIAO DIỆN CHÍNH
# =========================

st.markdown("""
<div class="main-title">
🚦 Traffic Law AI
</div>

<div class="sub-title">
Trợ lý AI tư vấn luật giao thông Việt Nam
</div>
""", unsafe_allow_html=True)


# Hiển thị lịch sử chat
if st.session_state.current_thread_id in st.session_state.threads:
    for msg in st.session_state.threads[st.session_state.current_thread_id]["messages"]:
        role_class = "chat-user" if msg["role"] == "user" else "chat-bot"    
        st.markdown(
            f"""
            <div class="{role_class}">
            {msg["content"].replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True
        )


        
# =========================
# VOICE INPUT (WHISPER)
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

# Ẩn st.audio_input nhưng vẫn cần để nhận data
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

# Ô chat input chính
typed_question = st.chat_input("Hỏi về mức phạt, biển báo...")

# st.audio_input ẩn bằng CSS — nút mic thật (JS) sẽ trigger nó
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

if user_question:
    if st.session_state.current_thread_id is None:
        new_id = str(uuid.uuid4())
        st.session_state.threads[new_id] = {"title": "Cuộc trò chuyện mới...", "messages": []}
        st.session_state.current_thread_id = new_id

    st.markdown(f'<div class="chat-user">{user_question}</div>', unsafe_allow_html=True)

    with st.spinner("🔍 Đang tra cứu luật..."):
        # =========================
        # QUERY EXPANSION
        # =========================

        # Lấy lịch sử gần nhất để có context
        recent_messages = st.session_state.threads[
            st.session_state.current_thread_id
        ]["messages"][-4:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent_messages
        )

        expansion_prompt = f"""Bạn là chuyên gia luật giao thông Việt Nam.
Nhiệm vụ: Viết lại câu hỏi thành query tìm kiếm tối ưu cho Elasticsearch.

Lịch sử hội thoại (nếu có):
{history_text}

Câu hỏi người dùng: {user_question}

YÊU CẦU:
- Chỉ trả về query tìm kiếm, không giải thích
- Bổ sung từ khóa pháp lý liên quan (mức phạt, điều khoản, loại xe...)
- Đồng nghĩa: "bằng lái/GPLX", "ô tô/xe hơi/oto", "xe máy/mô tô"
- Tối đa 20 từ
- Viết thường, không dấu câu"""

        try:
            expansion_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0,
                max_tokens=60
            )
            expanded_query = expansion_res.choices[0].message.content.strip().lower()
        except:
            expanded_query = user_question.lower()

        # Normalize thêm
        search_query = (
            expanded_query
            .replace("ô tô", "oto")
            .replace("xe hơi", "oto")
            .replace("giấy phép lái xe", "gplx")
            .replace("bằng lái", "gplx")
        )

        # =========================
        # EMBEDDING
        # =========================
        messages = st.session_state.threads[
            st.session_state.current_thread_id
        ]["messages"]
            
        vehicle_filter = None

        q = search_query.lower()

        if (
            "xe máy" in q
            or "mô tô" in q
            or "motor" in q
        ):
            vehicle_filter = "Xe máy"

        elif (
            "ô tô" in q
            or "oto" in q
            or "xe hơi" in q
            or "xe ô tô" in q
        ):
            vehicle_filter = "Ô tô"

        elif "xe tải" in q:
            vehicle_filter = "Xe tải"    
            
        query_vector = model.encode(
            search_query
        ).tolist()

        # =========================
        # SEARCH ELASTICSEARCH (3 tầng fallback)
        # =========================
        hits = []
        try:
            # Tầng 1: multi_match nhanh với query expansion
            r1 = es.search(index="traffic_law", size=5, query={
                "multi_match": {
                    "query": search_query,
                    "fields": ["title^4", "keywords^5", "content^2"],
                    "type": "best_fields"
                }
            })
            hits = r1["hits"]["hits"]
            st.caption(f"🔍 Tầng 1 (expanded): {len(hits)} kết quả")
        except Exception as e1:
            st.caption(f"⚠️ Tầng 1 lỗi: {e1}")

        if not hits:
            try:
                # Tầng 2: match với query gốc
                r2 = es.search(index="traffic_law", size=5, query={
                    "multi_match": {
                        "query": user_question,
                        "fields": ["title^4", "keywords^5", "content^2"]
                    }
                })
                hits = r2["hits"]["hits"]
                st.caption(f"🔍 Tầng 2 (câu gốc): {len(hits)} kết quả")
            except Exception as e2:
                st.caption(f"⚠️ Tầng 2 lỗi: {e2}")

        if not hits:
            try:
                # Tầng 3: multi_match với câu gốc
                r3 = es.search(index="traffic_law", size=5, query={
                    "multi_match": {"query": user_question, "fields": ["title^4","keywords^5","content^2"]}
                })
                hits = r3["hits"]["hits"]
                st.caption(f"🔍 Tầng 3 (multi_match gốc): {len(hits)} kết quả")
            except Exception as e3:
                st.caption(f"⚠️ Tầng 3 lỗi: {e3}")

        # =========================
        # BUILD CONTEXT
        # =========================
        if not hits:
            answer = "⚠️ Tôi chỉ tư vấn về luật giao thông Việt Nam. Bạn có câu hỏi về vi phạm, mức phạt hoặc biển báo không?"
        else:
            contexts = []
            for h in hits:
                src = h["_source"]
                context = f"""
            Tiêu đề: {src.get("title", "")}

            Loại phương tiện:
            {src.get("vehicle", "")}

            Mức phạt:
            {src.get("fine", "")}

            Nội dung:
            {src.get("content", "")}

            Căn cứ:
            Điểm {src.get("point", "")}
            Khoản {src.get("clause", "")}
            Điều {src.get("article", "")}
            {src.get("law_reference", "")}
            """
                contexts.append(context)

            top_contexts = "\n---\n".join(contexts)

            sys_prompt = f"""
Bạn là chuyên gia tư vấn luật giao thông Việt Nam, giải thích rõ ràng như một người bạn am hiểu pháp luật.

QUY TẮC:
- Dựa trên dữ liệu cung cấp, không bịa mức phạt hoặc điều khoản.
- Nếu dữ liệu thiếu, dùng kiến thức luật giao thông Việt Nam hiện hành để bổ sung hợp lý.
- Luôn phân biệt các trường hợp khác nhau nếu có (ví dụ: đè vạch vs vượt hẳn, xe máy vs ô tô).
- Xuống dòng rõ ràng, dễ đọc trên chat.
- Không dùng markdown (#, **, ---).

CẤU TRÚC TRẢ LỜI (linh hoạt theo câu hỏi):


[1-2 câu tóm tắt ngắn gọn nhất]

Chi tiết mức phạt:
- [Loại xe / trường hợp]: [mức phạt]
- ...

Hình thức bổ sung (nếu có):
- [tước bằng / tạm giữ xe / trừ điểm...]

Phân biệt các trường hợp (nếu có nhiều tình huống):
- [Tình huống A]: ...
- [Tình huống B]: ...

Căn cứ pháp lý:
- Điểm ... Khoản ... Điều ... Nghị định ...

Bạn có muốn biết thêm:
- [gợi ý câu hỏi liên quan 1]
- [gợi ý câu hỏi liên quan 2]
- [gợi ý câu hỏi liên quan 3]

DỮ LIỆU:
{top_contexts}
"""

            # =========================
            # CONTEXT HỘI THOẠI
            # =========================
            conversation_context = ""
            recent_msgs = messages[-4:]
            history = []

            for msg in recent_msgs:
                role = msg["role"]
                content = msg["content"]
                history.append(
                    f"{role}: {content}"
                )

            conversation_context = "\n".join(history)

            user_prompt = f"""
            Lịch sử hội thoại:
            {conversation_context}

            Câu hỏi hiện tại:
            {user_question}
            """

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0
            )

            answer = completion.choices[0].message.content

   
    # Hiển thị câu trả lời (Streaming giả lập)
    resp_placeholder = st.empty()
    full_resp = ""

    for char in answer:
        full_resp += char
        resp_placeholder.markdown(
            f"""
            <div class="chat-bot">
            {full_resp.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.01)

    # Lưu vào lịch sử
    st.session_state.threads[st.session_state.current_thread_id]["messages"].append({"role": "user", "content": user_question})
    st.session_state.threads[st.session_state.current_thread_id]["messages"].append({"role": "assistant", "content": answer})
    
    # Cập nhật tiêu đề nếu là tin nhắn đầu
    if len(st.session_state.threads[st.session_state.current_thread_id]["messages"]) <= 2:
        st.session_state.threads[st.session_state.current_thread_id]["title"] = generate_chat_title(user_question)
        st.rerun()