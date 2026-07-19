"""
📚 多策略 RAG 文件問答系統 —— 單機版 (Local-First / BYOK)
不使用 Supabase Auth，使用者自行輸入 Groq API Key 即可運作。
Supabase（選填）支援兩種金鑰來源：UI 手動輸入（優先） 或 secrets.toml（備援），
用於保存雲端問答紀錄。
AI 模型可在側邊欄自由切換，避免 Groq 停用特定模型時整套系統失效。
"""

import os
import io
import tempfile
import streamlit as st
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import docx

from supabase_client import get_supabase_client, save_qa_record, load_qa_history

# ------------------------------------------------------------------
# 基本頁面設定
# ------------------------------------------------------------------
st.set_page_config(
    page_title="多策略 RAG 文件問答系統",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
.sec-label {
    font-weight: 600;
    font-size: 0.95rem;
    margin-top: 0.5rem;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 多策略 RAG 文件問答系統")
st.caption("單機版・自帶金鑰 (BYOK)・本機向量資料庫 ChromaDB・選填雲端問答紀錄 (Supabase)・可切換 AI 模型")

# ------------------------------------------------------------------
# Session State 初始化（Groq 金鑰與文件相關，不依賴 Supabase）
# ------------------------------------------------------------------
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

# ------------------------------------------------------------------
# 側邊欄（必須先建立好 _sb_client，qa_history 初始化才能用到它）
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sec-label">🔑 Step 0：設定 Groq API Key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "輸入你的 Groq API Key",
        type="password",
        placeholder="gsk_xxxxxxxxxxxxxxxx",
        help="前往 https://console.groq.com/keys 免費申請",
    )

    if st.button("套用金鑰", use_container_width=True):
        if api_key_input.strip():
            st.session_state.groq_api_key = api_key_input.strip()
            os.environ["GROQ_API_KEY"] = api_key_input.strip()
            st.success("✓ API Key 已套用")
        else:
            st.error("請先輸入金鑰")

    if st.session_state.groq_api_key:
        st.success("✓ 目前已套用金鑰")
    else:
        st.warning("⚠ 尚未設定金鑰，無法進行問答")

    st.divider()

    # ------------------------------------------------------------
    # Supabase 連線設定（選填）：UI 輸入優先，其次退回 secrets.toml
    # ------------------------------------------------------------
    st.markdown('<div class="sec-label">☁️ Supabase 連線設定（選填）</div>', unsafe_allow_html=True)

    with st.expander("點此輸入 Supabase 金鑰", expanded=False):
        sb_url_input = st.text_input(
            "Project URL",
            placeholder="https://xxxxxxxx.supabase.co",
            key="sb_url_input",
        )
        sb_key_input = st.text_input(
            "anon public key",
            type="password",
            placeholder="eyJhbGciOiJIUzI1NiIs...",
            key="sb_key_input",
        )
        st.caption("留空則不啟用雲端紀錄，或系統會嘗試讀取部署者預設的 secrets.toml")

    _sb_client = get_supabase_client(sb_url_input, sb_key_input)

    if _sb_client is not None:
        st.success("✓ Supabase 已連線，問答紀錄將自動雲端保存")
        SUPABASE_ENABLED = True
    else:
        st.caption("⚠ 尚未設定 Supabase，紀錄僅保存在本次瀏覽器 Session")
        SUPABASE_ENABLED = False

    st.divider()

    st.markdown('<div class="sec-label">📄 Step 1：上傳文件</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "支援 PDF / DOCX",
        type=["pdf", "docx"],
    )

    chunk_size = st.slider("切片大小 (字元數)", 200, 1500, 500, step=50)
    chunk_overlap = st.slider("切片重疊 (字元數)", 0, 300, 50, step=10)

    st.divider()

    st.markdown('<div class="sec-label">🧠 Step 2：選擇檢索策略</div>', unsafe_allow_html=True)
    strategy = st.selectbox(
        "RAG 策略",
        [
            "1. 基礎相似度檢索 (Similarity Search)",
            "2. MMR 多樣性檢索 (Max Marginal Relevance)",
            "3. 混合檢索 (Hybrid: BM25 + Vector)",
            "4. 多查詢檢索 (Multi-Query)",
            "5. 上下文壓縮檢索 (Contextual Compression)",
            "6. 父子文件檢索 (Parent-Child Chunking)",
            "7. 自我查詢檢索 (Self-Query with Metadata)",
            "8. 重排序檢索 (Rerank with Cross-Encoder)",
        ],
    )

    top_k = st.slider("檢索片段數 (Top-K)", 1, 10, 4)

    st.divider()

    st.markdown('<div class="sec-label">🤖 Step 3：選擇 AI 模型</div>', unsafe_allow_html=True)
    groq_model = st.selectbox(
        "生成回答使用的模型",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.6-27b",
        ],
        index=0,
        help="Groq 平台的模型偶爾會退役停用，若某個模型突然無法使用，換一個選項即可，不需要改程式碼",
    )

# ------------------------------------------------------------------
# qa_history 初始化：此時 _sb_client 已建立完成，可安全使用
# ------------------------------------------------------------------
if "qa_history" not in st.session_state:
    cloud_history = load_qa_history(_sb_client, limit=20)
    if cloud_history:
        st.session_state.qa_history = [
            {
                "question": h.get("question", ""),
                "answer": h.get("answer", ""),
                "strategy": h.get("strategy", ""),
            }
            for h in reversed(cloud_history)
        ]
    else:
        st.session_state.qa_history = []

# ------------------------------------------------------------------
# 工具函式：文字擷取
# ------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break
    return [c for c in chunks if c.strip()]


# ------------------------------------------------------------------
# 工具函式：ChromaDB 初始化
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_chroma_client():
    tmp_dir = os.path.join(tempfile.gettempdir(), "rag_chroma_db")
    os.makedirs(tmp_dir, exist_ok=True)
    return chromadb.PersistentClient(path=tmp_dir)


@st.cache_resource(show_spinner=False)
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def build_collection(chunks: list[str]):
    client = get_chroma_client()
    ef = get_embedding_function()

    # 每次上傳新文件時，重建一個乾淨的 collection
    try:
        client.delete_collection("rag_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="rag_docs",
        embedding_function=ef,
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return collection


# ------------------------------------------------------------------
# 工具函式：檢索策略實作
# ------------------------------------------------------------------
def retrieve_similarity(collection, query: str, k: int):
    result = collection.query(query_texts=[query], n_results=k)
    return result["documents"][0]


def retrieve_mmr(collection, query: str, k: int):
    result = collection.query(query_texts=[query], n_results=min(k * 3, 20))
    candidates = result["documents"][0]
    selected = []
    for doc in candidates:
        if len(selected) >= k:
            break
        if all(doc[:50] != s[:50] for s in selected):
            selected.append(doc)
    return selected[:k]


def retrieve_hybrid(collection, query: str, k: int):
    result = collection.query(query_texts=[query], n_results=min(k * 2, 20))
    candidates = result["documents"][0]
    keywords = set(query.lower().split())

    def score(doc):
        doc_words = set(doc.lower().split())
        overlap = len(keywords & doc_words)
        return overlap

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[:k]


def retrieve_multi_query(collection, query: str, k: int, client: Groq, model: str):
    variations_prompt = f"請針對以下問題，產生 2 個不同角度但語意相近的改寫問題，每行一個，不要編號：\n{query}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": variations_prompt}],
            temperature=0.5,
        )
        variations = [q.strip() for q in resp.choices[0].message.content.split("\n") if q.strip()]
    except Exception:
        variations = []

    all_queries = [query] + variations[:2]
    seen = set()
    merged = []
    for q in all_queries:
        result = collection.query(query_texts=[q], n_results=k)
        for doc in result["documents"][0]:
            if doc not in seen:
                seen.add(doc)
                merged.append(doc)
    return merged[:k]


def retrieve_contextual_compression(collection, query: str, k: int, client: Groq, model: str):
    raw_docs = retrieve_similarity(collection, query, k)
    compressed = []
    for doc in raw_docs:
        prompt = f"請將以下段落中，與問題「{query}」直接相關的句子摘錄出來，其他無關內容省略：\n\n{doc}"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            compressed.append(resp.choices[0].message.content.strip())
        except Exception:
            compressed.append(doc)
    return compressed


def retrieve_parent_child(collection, query: str, k: int):
    result = collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
    docs = result["documents"][0]
    return docs


def retrieve_self_query(collection, query: str, k: int):
    return retrieve_similarity(collection, query, k)


def retrieve_rerank(collection, query: str, k: int):
    result = collection.query(query_texts=[query], n_results=min(k * 3, 20))
    candidates = result["documents"][0]
    query_words = set(query.lower().split())

    def relevance_score(doc):
        doc_words = set(doc.lower().split())
        return len(query_words & doc_words) / (len(doc_words) + 1)

    ranked = sorted(candidates, key=relevance_score, reverse=True)
    return ranked[:k]


def run_retrieval(strategy_name: str, collection, query: str, k: int, client: Groq, model: str):
    if strategy_name.startswith("1."):
        return retrieve_similarity(collection, query, k)
    elif strategy_name.startswith("2."):
        return retrieve_mmr(collection, query, k)
    elif strategy_name.startswith("3."):
        return retrieve_hybrid(collection, query, k)
    elif strategy_name.startswith("4."):
        return retrieve_multi_query(collection, query, k, client, model)
    elif strategy_name.startswith("5."):
        return retrieve_contextual_compression(collection, query, k, client, model)
    elif strategy_name.startswith("6."):
        return retrieve_parent_child(collection, query, k)
    elif strategy_name.startswith("7."):
        return retrieve_self_query(collection, query, k)
    elif strategy_name.startswith("8."):
        return retrieve_rerank(collection, query, k)
    else:
        return retrieve_similarity(collection, query, k)


# ------------------------------------------------------------------
# 主流程：文件上傳與向量化
# ------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📥 文件處理")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        if uploaded_file.name.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_bytes)
        else:
            raw_text = extract_text_from_docx(file_bytes)

        if not raw_text.strip():
            st.error("⚠ 無法從此文件擷取任何文字，請確認文件內容是否為純圖片掃描檔")
        else:
            chunks = chunk_text(raw_text, chunk_size, chunk_overlap)

            if st.button("🔄 建立向量索引", type="primary"):
                with st.spinner("正在切片與向量化，請稍候..."):
                    st.session_state.collection = build_collection(chunks)
                    st.session_state.chunks_count = len(chunks)
                st.success(f"✓ 完成！共切成 {len(chunks)} 個片段，已建立本機向量索引")

    if st.session_state.chunks_count > 0:
        st.info(f"📊 目前索引中共有 **{st.session_state.chunks_count}** 個文件片段")

with col2:
    st.subheader("📈 系統狀態")
    st.metric("已載入片段數", st.session_state.chunks_count)
    st.metric("問答紀錄筆數", len(st.session_state.qa_history))
    st.caption(f"🤖 使用模型：`{groq_model}`")
    if SUPABASE_ENABLED:
        st.caption("☁️ 紀錄同步：已啟用")
    else:
        st.caption("💾 紀錄同步：僅本機 Session")

st.divider()

# ------------------------------------------------------------------
# 主流程：問答介面
# ------------------------------------------------------------------
st.subheader("💬 開始問答")

if not st.session_state.groq_api_key:
    st.warning("⚠ 請先在左側 Step 0 設定 Groq API Key")
elif st.session_state.collection is None:
    st.warning("⚠ 請先上傳文件並建立向量索引")
else:
    question = st.text_input("輸入你的問題", placeholder="例如：這份文件的重點是什麼？")

    if st.button("🔍 送出問題", type="primary") and question.strip():
        client = Groq(api_key=st.session_state.groq_api_key)

        with st.spinner(f"使用「{strategy}」策略檢索中..."):
            retrieved_chunks = run_retrieval(
                strategy, st.session_state.collection, question, top_k, client, groq_model
            )

        context = "\n\n---\n\n".join(retrieved_chunks)

        prompt = f"""你是一個嚴謹的文件問答助手，請僅根據以下提供的文件內容回答問題。
如果文件內容無法回答此問題，請明確說明「文件中未提及相關內容」，不要編造答案。

文件內容：
{context}

問題：{question}

請用繁體中文回答，並盡量條理清晰。"""

        with st.spinner(f"正在使用 {groq_model} 生成回答..."):
            try:
                response = client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"⚠ 呼叫 Groq API 時發生錯誤：{e}\n\n💡 若錯誤訊息提到模型已停用或找不到，請到左側 Step 3 切換成其他模型再試一次。"

        st.markdown("### 📝 回答")
        st.write(answer)

        with st.expander("📚 查看檢索到的原始片段"):
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(f"**片段 {i}：**")
                st.text(chunk)
                st.divider()

        # 寫入本機 Session 歷史
        st.session_state.qa_history.append({
            "question": question,
            "answer": answer,
            "strategy": strategy,
        })

        # 若 Supabase 已連線，同步寫入雲端
        if SUPABASE_ENABLED:
            saved = save_qa_record(_sb_client, question, answer, strategy)
            if saved:
                st.toast("☁️ 已同步至雲端紀錄", icon="✅")
            else:
                st.toast("⚠ 雲端同步失敗，僅保存於本機", icon="⚠️")

    if st.session_state.qa_history:
        st.divider()
        st.subheader("🕘 問答紀錄" + ("（含雲端歷史）" if SUPABASE_ENABLED else "（本次 Session）"))
        for item in reversed(st.session_state.qa_history):
            with st.expander(f"❓ {item['question']}　（策略：{item['strategy']}）"):
                st.write(item["answer"])
