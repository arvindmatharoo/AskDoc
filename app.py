# =============================================================================
#  app.py  —  AskDoc: Conversational HR Policy Assistant
#  UI   : Dark Glassmorphism · Violet/Purple Accent · Claude.ai-style chat
#  Stack: LangChain · LLaMA 3.3 70B · Groq · Chroma · HuggingFace Embeddings
# =============================================================================

import os
import json
import uuid
import datetime
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


# ─── Constants ────────────────────────────────────────────────────────────────

PDF_PATH = "TechNova_HR_Policy.pdf"
DB_PATH  = "./chroma_db"

# Module-level dict: keyed by session ID, holds display messages.
# Lives in the Python process — survives page refreshes (F5) because
# the session ID travels in the URL (?sid=...), but is wiped clean
# when the Streamlit server restarts. No files, no database.
_SESSION_CACHE: dict = {}

STARTER_QS = [
    ("", "What is the leave policy?"),
    ("", "How many sick leaves do employees get per year?"),
    ("", "What are the official work hours at TechNova?"),
    ("", "What does the code of conduct say?"),
]


# ─── 1. Page Config  (must be first Streamlit call) ──────────────────────────

st.set_page_config(
    page_title="AskDoc",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()


# ─── 2. Global CSS — Dark Glassmorphism + Violet Accent ──────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset ── */
*, *::before, *::after {
    box-sizing: border-box;
    font-family: 'Inter', sans-serif !important;
}

/* ── App background ── */
html, body, .stApp {
    background-color: #0D0D1A !important;
    color: #CBD5E1 !important;
}
.stApp {
    background: #0D0D1A !important;
    background-image:
        radial-gradient(ellipse at 15% 40%, rgba(124,58,237,0.17) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 10%, rgba(109,40,217,0.11) 0%, transparent 50%),
        radial-gradient(ellipse at 55% 90%, rgba(139,92,246,0.07) 0%, transparent 45%) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer,
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.35); border-radius: 4px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {

    background:#0D0D1A !important;

    background-image:
        radial-gradient(circle at top left,
        rgba(124,58,237,.18),
        transparent 55%),

        radial-gradient(circle at bottom right,
        rgba(109,40,217,.12),
        transparent 45%) !important;

    backdrop-filter:blur(28px);

    -webkit-backdrop-filter:blur(28px);

    border-right:1px solid rgba(124,58,237,.15);

    box-shadow:6px 0 32px rgba(0,0,0,.45);

}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
    padding-top: 1.4rem !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.055) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin: 6px 0 !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    transition: border-color 0.2s;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(124,58,237,0.22) !important;
}

/* Hide default avatars — labels replace them */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"],
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
img[alt="user avatar"],
img[alt="assistant avatar"] {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
    padding: 0 !important;
}
[data-testid="stChatMessage"] > div { gap: 0 !important; }

/* ── Chat input container ── */
[data-testid="stBottom"] {
    background: rgba(9,9,18,0.88) !important;
    backdrop-filter: blur(20px) !important;
    border-top: 1px solid rgba(124,58,237,0.12) !important;
    padding-bottom: 48px !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(124,58,237,0.28) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    caret-color: #A78BFA !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(124,58,237,0.62) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.18) !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(203,213,225,0.28) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: rgba(124,58,237,0.10) !important;
    border: 1px solid rgba(124,58,237,0.32) !important;
    color: #C4B5FD !important;
    border-radius: 9px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: rgba(124,58,237,0.22) !important;
    border-color: rgba(124,58,237,0.6) !important;
    color: #EDE9FE !important;
    box-shadow: 0 0 16px rgba(124,58,237,0.22) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Primary / form-submit buttons */
.stButton > button[kind="primary"],
.stButton > button[kind="primaryFormSubmit"],
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    border-color: transparent !important;
    color: #F5F3FF !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.35) !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primaryFormSubmit"]:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #8B5CF6, #7C3AED) !important;
    box-shadow: 0 6px 24px rgba(124,58,237,0.50) !important;
    transform: translateY(-2px) !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(124,58,237,0.28) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    transition: all 0.18s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(124,58,237,0.65) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.18) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: rgba(203,213,225,0.30) !important;
}

/* ── HR ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(124,58,237,0.13) !important;
    margin: 14px 0 !important;
}

/* ── Form ── */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Alert ── */
[data-testid="stAlert"] {
    background: rgba(124,58,237,0.07) !important;
    border: 1px solid rgba(124,58,237,0.22) !important;
    border-radius: 10px !important;
    color: #C4B5FD !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7C3AED !important; }

/* ── Toast ── */
[data-testid="stToastContainer"] { z-index: 9999 !important; }

/* ── Main block padding ── */
.main .block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 940px !important;
}

/* ══════════════ CUSTOM COMPONENTS ══════════════ */

/* Sidebar logo */
.sb-logo {
    display: flex; align-items: center; gap: 10px; margin-bottom: 5px;
}
.sb-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #7C3AED, #A78BFA);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 18px rgba(124,58,237,0.45);
    flex-shrink: 0;
}
.sb-title {
    font-size: 21px; font-weight: 700; letter-spacing: -0.5px;
    color: #F8FAFC !important;
}
.sb-title em { font-style: normal; color: #A78BFA !important; }

/* Online badge */
.badge-online {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(16,185,129,0.09);
    border: 1px solid rgba(16,185,129,0.22);
    color: #6EE7B7 !important;
    padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 500; margin-bottom: 18px;
}
.dot-live {
    width: 6px; height: 6px; border-radius: 50%;
    background: #10B981; box-shadow: 0 0 6px #10B981;
    display: inline-block;
}

/* Sidebar section labels */
.sec-lbl {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.10em;
    color: rgba(203,213,225,0.28) !important;
    font-weight: 600; margin-top: 20px; margin-bottom: 8px;
}

/* Info card */
.info-card {
    background: rgba(124,58,237,0.055);
    border: 1px solid rgba(124,58,237,0.14);
    border-radius: 10px; padding: 12px 14px;
}
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; font-size: 12.5px;
}
.info-row:not(:last-child) { border-bottom: 1px solid rgba(255,255,255,0.04); }
.ik { color: rgba(203,213,225,0.42) !important; font-size: 11.5px; }
.iv {
    color: #C4B5FD !important; font-weight: 500;
    font-size: 12px; text-align: right; word-break: break-word; max-width: 55%;
}

/* Welcome card */
.wlc-card {
    background: rgba(255,255,255,0.032);
    backdrop-filter: blur(30px); -webkit-backdrop-filter: blur(30px);
    border: 1px solid rgba(124,58,237,0.18); border-radius: 24px;
    padding: 46px 52px;
    box-shadow:
        0 12px 42px rgba(0,0,0,0.5),
        0 0 80px rgba(124,58,237,0.06),
        inset 0 1px 0 rgba(255,255,255,0.06);
    text-align: center; margin-bottom: 20px;
}
.wlc-badge {
    display: inline-block;
    background: rgba(124,58,237,0.14);
    border: 1px solid rgba(124,58,237,0.28);
    color: #C4B5FD !important; padding: 4px 14px; border-radius: 20px;
    font-size: 11.5px; font-weight: 500; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 22px;
}
.wlc-title {
    font-size: 37px !important; font-weight: 700 !important;
    color: #F8FAFC !important; letter-spacing: -0.5px !important;
    line-height: 1.15 !important; margin-bottom: 10px !important;
}
.wlc-sub {
    color: rgba(203,213,225,0.50) !important;
    font-size: 14.5px !important; margin-bottom: 0 !important;
    line-height: 1.65 !important;
}

/* Chat header */
.chat-hdr {
    padding: 18px 0 14px;
    border-bottom: 1px solid rgba(124,58,237,0.10);
    margin-bottom: 6px;
}
.chat-hdr-greet {
    font-size: 12.5px; color: rgba(167,139,250,0.75) !important;
    font-weight: 500; margin-bottom: 3px;
}
.chat-hdr-title {
    font-size: 22px; font-weight: 700;
    color: #F8FAFC !important; letter-spacing: -0.3px;
}

/* Message labels */
.msg-lbl {
    font-size: 10.5px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 7px;
}
.lbl-you   { color: #A78BFA !important; }
.lbl-askdoc { color: rgba(167,139,250,0.55) !important; }

/* Typing indicator */
.typing-wrap {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 20px;
    background: rgba(255,255,255,0.022);
    border: 1px solid rgba(124,58,237,0.10);
    border-radius: 14px; margin: 7px 0;
    animation: fadeUp 0.3s ease;
}
.typing-txt {
    font-size: 13px; color: rgba(167,139,250,0.70) !important; font-style: italic;
}
.t-dots { display: flex; gap: 4px; align-items: center; }
.t-dots span {
    width: 5px; height: 5px; border-radius: 50%;
    background: #7C3AED; opacity: 0.3;
    animation: tdot 1.3s infinite ease-in-out;
}
.t-dots span:nth-child(2) { animation-delay: 0.18s; }
.t-dots span:nth-child(3) { animation-delay: 0.36s; }
@keyframes tdot {
    0%,80%,100% { opacity:0.2; transform:scale(0.85); }
    40%          { opacity:1;   transform:scale(1.15); }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(6px); }
    to   { opacity:1; transform:translateY(0);   }
}

/* Empty state */
.empty-wrap { text-align: center; padding: 32px 0 22px; }
.empty-icon {
    font-size: 44px; margin-bottom: 14px; display: block;
    filter: drop-shadow(0 0 14px rgba(124,58,237,0.5));
}
.empty-title {
    font-size: 19px; font-weight: 600;
    color: #E2E8F0 !important; margin-bottom: 6px;
}
.empty-sub {
    font-size: 13.5px;
    color: rgba(203,213,225,0.38) !important; margin-bottom: 28px;
}
.starters-lbl {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.10em;
    color: rgba(167,139,250,0.45) !important;
    margin-bottom: 12px; font-weight: 600;
}

/* Footer */
.footer {
    position: fixed; bottom: 0; left: 0; right: 0; height: 38px;
    background: rgba(9,9,18,0.95);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border-top: 1px solid rgba(124,58,237,0.10);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 22px; z-index: 200; font-size: 11.5px;
}
.ft-tech { color: rgba(203,213,225,0.25) !important; }
.ft-tech em { font-style: normal; color: rgba(167,139,250,0.50) !important; }
.ft-name { color: rgba(203,213,225,0.25) !important; }
.ft-name em { font-style: normal; color: rgba(167,139,250,0.50) !important; }
/* Hide Streamlit's default sidebar controls */
button[kind="header"] {
    display: none !important;
}

[data-testid="collapsedControl"] {
    display: none !important;
}

[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
            </style>
""", unsafe_allow_html=True)


# ─── 3. Backend  (cached across reruns) ──────────────────────────────────────

@st.cache_resource(show_spinner="⚙️  Initialising AskDoc backend…")
def initialize_backend():
    """
    Build the full RAG pipeline exactly as in chatbot_og.ipynb.
    Loads Chroma from disk if it already exists, otherwise creates it.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # ── Vector DB ──
    if not os.path.exists(DB_PATH):
        loader   = PyPDFLoader(PDF_PATH)
        raw_docs = loader.load()
        docs     = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=150
        ).split_documents(raw_docs)
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_PATH,
            collection_name="hr_policy",
        )
        n_chunks = len(docs)
    else:
        db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name="hr_policy",
        )
        try:
            n_chunks = db._collection.count()
        except Exception:
            n_chunks = "N/A"

    # ── Retriever (MMR) ──
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 30, "lambda_mult": 0.7},
    )

    # ── History-aware retriever prompt ──
    ctx_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might reference "
         "context in the chat history, formulate a standalone question that can be "
         "understood without the chat history. Do NOT answer the question — just "
         "reformulate it if needed, otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    hist_retriever = create_history_aware_retriever(llm, retriever, ctx_prompt)

    # ── QA prompt ──
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI assistant that answers questions ONLY from the uploaded "
         "HR Policy document.\n\n"
         "Rules:\n"
         "1. Answer ONLY using the retrieved context.\n"
         "2. Never use outside knowledge.\n"
         "3. If the answer is not in the context, say: "
         "\"I don't know based on the uploaded document.\"\n"
         "4. Combine multiple relevant passages into one coherent answer.\n"
         "5. Do not produce contradictory statements.\n"
         "6. Use bullet points whenever appropriate.\n"
         "7. Keep answers clear and concise.\n\n"
         "Retrieved Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(hist_retriever, doc_chain)

    return rag_chain, n_chunks


# ─── 4. Session State ─────────────────────────────────────────────────────────

def _init():
    defaults = {
        "user_name":      None,          # set after welcome screen
        "messages":       [],            # list of {"role", "content"} dicts
        "session_id":     None,  # passed to RunnableWithMessageHistory
        "store":          {},            # in-memory chat history store
        "pending_q":      None,          # starter question waiting to be processed
        "sidebar_visible":   True,          # sidebar visibility toggle
        "history_loaded": False,         # guard: load from cache only once per session
        "chat_sessions" : {},
        "current_chat" : None,
        "chat_counter" : 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─── 5. Helpers ───────────────────────────────────────────────────────────────

def get_greeting() -> tuple[str, str]:
    h = datetime.datetime.now().hour
    if  5 <= h < 12: return "Good Morning",   "☀️"
    if 12 <= h < 17: return "Good Afternoon", "🌤️"
    if 17 <= h < 23: return "Good Evening",   "🌙"
    return "Hello Night Owl", "🦉"


def get_hist(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) per-session chat history stored in session_state."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


def build_chain(rag_chain) -> RunnableWithMessageHistory:
    """Wrap the cached RAG chain with per-session history on every call."""
    return RunnableWithMessageHistory(
        rag_chain,
        get_hist,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# ── Session-cache helpers (chat history persistence) ─────────────────────────

def _get_sid() -> str:
    """
    Return a stable session ID stored as a URL query param (?sid=...).
    The URL is preserved across F5 page refreshes, so the same ID is
    recovered and the module-level cache can be consulted again.
    Falls back to session_state if query params are unavailable.
    """
    try:
        params = st.query_params
        if "sid" in params:
            return params["sid"]
        new_sid = uuid.uuid4().hex[:12]
        st.query_params["sid"] = new_sid
        return new_sid
    except Exception:
        if "_sid" not in st.session_state:
            st.session_state["_sid"] = uuid.uuid4().hex[:12]
        return st.session_state["_sid"]


def _cache_save(sid: str, messages: list) -> None:
    """Write current display messages into the module-level cache."""
    _SESSION_CACHE[sid] = list(messages)


def _cache_load(sid: str) -> list:
    """Read display messages from cache; returns [] if nothing stored."""
    return list(_SESSION_CACHE.get(sid, []))


def _rebuild_langchain_history(messages: list) -> InMemoryChatMessageHistory:
    """
    Reconstruct an InMemoryChatMessageHistory from the saved display
    messages so that the RAG chain has correct context after a refresh.
    """
    history = InMemoryChatMessageHistory()
    for msg in messages:
        if msg["role"] == "user":
            history.add_message(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.add_message(AIMessage(content=msg["content"]))
    return history


def copy_btn(content: str, uid: str) -> None:
    """
    Render a glassmorphism copy button inside a components.html iframe.
    Uses json.dumps for bulletproof JS string escaping of any content.
    """
    payload = json.dumps(content)   # handles newlines, quotes, backslashes
    components.html(f"""
    <style>
      body {{ margin:0; padding:0; background:transparent; }}
      .cb {{
        background: rgba(124,58,237,0.08);
        border: 1px solid rgba(124,58,237,0.22);
        color: rgba(167,139,250,0.6);
        padding: 3px 10px; border-radius: 6px;
        cursor: pointer; font-size: 11px;
        font-family: 'Inter', sans-serif;
        transition: all .18s;
        display: inline-flex; align-items: center; gap: 4px;
      }}
      .cb:hover {{
        background: rgba(124,58,237,0.20);
        color: #C4B5FD;
        border-color: rgba(124,58,237,0.5);
      }}
                            
    </style>
    <button class="cb" id="cb-{uid}">📋 Copy</button>
    <script>
      var text = {payload};
      document.getElementById("cb-{uid}").onclick = function() {{
        navigator.clipboard.writeText(text).then(function() {{
          document.getElementById("cb-{uid}").innerHTML = "✅ Copied";
          setTimeout(function() {{
            document.getElementById("cb-{uid}").innerHTML = "📋 Copy";
          }}, 1500);
        }});
      }};
    </script>
    """, height=32)


def _footer():
    st.markdown("""
    <div class="footer">
      <span class="ft-tech">
        Powered by <em>LLaMA 3.3 70B</em> · <em>Groq</em> ·
        <em>LangChain</em> · <em>Chroma</em>
      </span>
      <span class="ft-name">Built by <em>Arvind Matharoo</em></span>
    </div>
    """, unsafe_allow_html=True)


# ─── 6. Sidebar ───────────────────────────────────────────────────────────────

def render_sidebar(n_chunks):
    chunk_str = f"{n_chunks:,}" if isinstance(n_chunks, int) else str(n_chunks)
    n_msgs    = len(st.session_state.messages)
    n_qs      = sum(1 for m in st.session_state.messages if m["role"] == "user")

    with st.sidebar:
        # Logo + status
        st.markdown(f"""
        <div class="sb-logo">
          <div class="sb-icon">📄</div>
          <div class="sb-title">Ask<em>Doc</em></div>
        </div>
        <div class="badge-online">
          <span class="dot-live"></span> Online
        </div>

        <!-- Document info -->
        <div class="sec-lbl">Document</div>
        <div class="info-card">
          <div class="info-row">
            <span class="ik">📂 File</span>
            <span class="iv">{PDF_PATH}</span>
          </div>
          <div class="info-row">
            <span class="ik">🧩 Chunks</span>
            <span class="iv">{chunk_str}</span>
          </div>
        </div>

        <!-- Model stack -->
        <div class="sec-lbl">Model Stack</div>
        <div class="info-card">
          <div class="info-row">
            <span class="ik">🤖 LLM</span>
            <span class="iv">LLaMA 3.3 70B</span>
          </div>
          <div class="info-row">
            <span class="ik">⚡ Provider</span>
            <span class="iv">Groq</span>
          </div>
          <div class="info-row">
            <span class="ik">🔍 Retriever</span>
            <span class="iv">MMR · k=4</span>
          </div>
          <div class="info-row">
            <span class="ik">🧠 Embeddings</span>
            <span class="iv">BGE Small EN</span>
          </div>
          <div class="info-row">
            <span class="ik">💾 Vector DB</span>
            <span class="iv">Chroma (local)</span>
          </div>
        </div>

        <!-- Session info -->
        <div class="sec-lbl">Session</div>
        <div class="info-card">
          <div class="info-row">
            <span class="ik">👤 User</span>
            <span class="iv">{st.session_state.user_name}</span>
          </div>
          <div class="info-row">
            <span class="ik">💬 Messages</span>
            <span class="iv">{n_msgs}</span>
          </div>
          <div class="info-row">
            <span class="ik">❓ Questions</span>
            <span class="iv">{n_qs}</span>
          </div>
        </div>
        <div style="height:14px"></div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 💬 Chats")

        for chat_id, chat in st.session_state.chat_sessions.items():

            title = chat["title"]

            if st.button(title, key=chat_id, use_container_width=True):

                st.session_state.current_chat = chat_id

                st.session_state.messages = chat["messages"]

                st.session_state.session_id = chat_id

                st.session_state.store = {
                    chat_id: _rebuild_langchain_history(chat["messages"])
                }

                st.rerun()
        if st.button("➕ New Chat", use_container_width=True):

            # Save current messages
            current = st.session_state.current_chat

            st.session_state.chat_sessions[current]["messages"] = (
                st.session_state.messages
            )

            # Create new chat
            new_chat = uuid.uuid4().hex

            st.session_state.chat_sessions[new_chat] = {
                "title": f"Chat {st.session_state.chat_counter}",
                "messages": [],
            }

            st.session_state.chat_counter += 1 

            st.session_state.current_chat = new_chat

            st.session_state.messages = []

            st.session_state.store = {}

            st.session_state.pending_q = None

            st.session_state.session_id = new_chat

            st.rerun()
        if st.button("🗑️ Clear Chat", use_container_width=True):

            sid = _get_sid()

            _cache_save(sid, [])

            st.session_state.messages = []

            st.session_state.store = {}

            st.rerun()

        if st.button("🚪 Sign Out", use_container_width=True):

            sid = _get_sid()

            # Clear current cache
            _cache_save(sid, [])

            # Clear current conversation
            st.session_state.messages = []
            st.session_state.store = {}
            st.session_state.pending_q = None
            st.session_state.history_loaded = False

            # Clear all saved chats
            st.session_state.chat_sessions = {}
            st.session_state.current_chat = None
            st.session_state.chat_counter += 1

            # Sign out
            st.session_state.user_name = None

            st.rerun()


# ─── 7. Welcome Screen ────────────────────────────────────────────────────────

def render_welcome():
    greet, emoji = get_greeting()

    # ── Welcome-specific CSS overrides ──
    # Injected here so they only apply on the welcome screen, not the chat page.
    # Key goals:
    #   1. Kill all scrolling (overflow: hidden everywhere)
    #   2. Strip Streamlit's default block-container padding so nothing pushes
    #      the card below the visible fold
    #   3. Turn the block-container into a full-viewport flex column so the
    #      st.columns(...) block lands exactly at vertical centre
    st.markdown("""
    <style>
    html, body { overflow: hidden !important; height: 100vh !important; }
    .stApp      { overflow: hidden !important; height: 100vh !important; }
    section.main { overflow: hidden !important; height: 100vh !important; }

    /* Make the Streamlit content wrapper a centred flex column */
    .main .block-container {
        padding-top:    0 !important;
        padding-bottom: 0 !important;
        margin-top:     0 !important;
        max-width:      100% !important;
        width:          100%  !important;
        height:         calc(100vh - 38px) !important;   /* 38px = footer */
        display:        flex !important;
        flex-direction: column !important;
        align-items:    center !important;
        justify-content: center !important;
    }

    /* The horizontal columns block must also stretch to full width */
    [data-testid="stHorizontalBlock"] {
        width: 100% !important;
    }

    /* Remove any ghost top-margin Streamlit adds to the first child */
    .main .block-container > div:first-child {
        margin-top: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.8, 1])
    with mid:
        # Glass card — pure HTML, no Streamlit widgets inside
        st.markdown(f"""
        <div class="wlc-card">
          <div class="wlc-badge">📄 AskDoc</div>
          <div class="wlc-title">{greet} {emoji}</div>
          <div class="wlc-sub">
            Your intelligent document assistant.<br>
            Ask anything about the HR policy — instantly.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # st.form gives free Enter-to-submit behaviour
        with st.form("welcome_form", clear_on_submit=False):
            name = st.text_input(
                "",
                placeholder="Enter your name to get started…",
                label_visibility="collapsed",
            )
            ok = st.form_submit_button(
                "Get Started →",
                use_container_width=True,
                type="primary",
            )

        if ok:
            if name.strip():
                st.session_state.user_name = name.strip()
                chat_id = uuid.uuid4().hex

                st.session_state.current_chat = chat_id

                st.session_state.chat_sessions[chat_id] = {
                    "title": f"Chat {st.session_state.chat_counter}",
                    "messages": [],
                }

                st.session_state.chat_counter += 1
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")

    _footer()


# ─── 8. Chat Interface ────────────────────────────────────────────────────────

def render_chat(rag_chain):
    greet, emoji = get_greeting()
    name         = st.session_state.user_name
    sid          = _get_sid()
    st.session_state.session_id = sid
    toggle_col, header_col = st.columns([1, 12])

    with toggle_col:
        if st.button("☰", key="toggle_sidebar"):
            st.session_state.sidebar_visible = (
                not st.session_state.sidebar_visible
            )
            st.rerun()
    
    # ── Restore chat history (runs once per browser session) ──
    # On a fresh load or F5 refresh, session_state.messages is empty but
    # the module-level cache may still have the messages keyed by sid.
    if not st.session_state.history_loaded:
        saved = _cache_load(sid)
        if saved:
            st.session_state.messages = saved
            # Rebuild LangChain memory so follow-up questions still work
            st.session_state.store[st.session_state.session_id] = \
                _rebuild_langchain_history(saved)
        st.session_state.history_loaded = True

    # ── Header + sidebar toggle button ──
    with header_col:
        st.markdown(f"""
        <div class="chat-hdr">
          <div class="chat-hdr-greet">{greet}, {name} {emoji}</div>
          <div class="chat-hdr-title">What can I help you with today?</div>
        </div>
        """, unsafe_allow_html=True)
    

    # ── Empty state + starter questions ──
    if not st.session_state.messages and not st.session_state.pending_q:
        st.markdown("""
        <div class="empty-wrap">
          <span class="empty-icon">📄</span>
          <div class="empty-title">AskDoc is ready</div>
          <div class="empty-sub">
            Ask me anything about the TechNova HR Policy document
          </div>
          <div class="starters-lbl">Try asking</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        for i, (icon, q) in enumerate(STARTER_QS):
            with (c1 if i % 2 == 0 else c2):
                if st.button(f"{icon}  {q}", key=f"sq_{i}", use_container_width=True):
                    st.session_state.pending_q = q
                    st.rerun()

    # ── Historical messages ──
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(
                    '<div class="msg-lbl lbl-you">You</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="msg-lbl lbl-askdoc">AskDoc</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                copy_btn(msg["content"], uid=f"h{idx}")

    # ── Resolve current input (typed or from starter button) ──
    typed   = st.chat_input("Ask anything about the HR policy…")
    user_in = st.session_state.pending_q or typed
    if st.session_state.pending_q:
        st.session_state.pending_q = None

    # ── Process ──
    if user_in:
        st.session_state.messages.append({"role": "user", "content": user_in})
        current = st.session_state.current_chat

        if st.session_state.chat_sessions[current]["title"] == "New Chat":

            title = user_in.strip()

            if len(title) > 35:
                title = title[:35] + "..."

            st.session_state.chat_sessions[current]["title"] = title
        current = st.session_state.current_chat

        st.session_state.chat_sessions[current]["messages"] = (st.session_state.messages)
        with st.chat_message("user"):
            st.markdown(
                '<div class="msg-lbl lbl-you">You</div>',
                unsafe_allow_html=True,
            )
            _cache_save(sid, st.session_state.messages)
            st.markdown(user_in)

        with st.chat_message("assistant"):
            st.markdown(
                '<div class="msg-lbl lbl-askdoc">AskDoc</div>',
                unsafe_allow_html=True,
            )
            placeholder = st.empty()
            placeholder.markdown("""
            <div class="typing-wrap">
              <div class="t-dots">
                <span></span><span></span><span></span>
              </div>
              <span class="typing-txt">AskDoc is typing…</span>
            </div>
            """, unsafe_allow_html=True)

            try:
                chain = build_chain(rag_chain)
                ans   = chain.invoke(
                    {"input": user_in},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )["answer"]
            except Exception as exc:
                ans = f"⚠️ Something went wrong: {exc}"

            placeholder.empty()
            st.markdown(ans)
            copy_btn(ans, uid="live")

        # Persist to both session_state and module-level cache
        st.session_state.messages.append({"role": "assistant", "content": ans})
        _cache_save(sid, st.session_state.messages)

    _footer()


# ─── 9. Entry Point ───────────────────────────────────────────────────────────

def main():
    # Guard: need either the PDF or an existing Chroma DB
    if not os.path.exists(DB_PATH) and not os.path.exists(PDF_PATH):
        st.error(
            f"❌ Neither **{PDF_PATH}** nor the Chroma DB (**{DB_PATH}**) were found.\n\n"
            "Please place `TechNova_HR_Policy.pdf` in the same directory as `app.py` "
            "and re-run the app."
        )
        st.stop()

    rag_chain, n_chunks = initialize_backend()

    if not st.session_state.user_name:
        render_welcome()
    else:
        
        if st.session_state.sidebar_visible: 
            render_sidebar(n_chunks)
        render_chat(rag_chain)


if __name__ == "__main__":
    main()