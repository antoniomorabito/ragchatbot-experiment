import os
import streamlit as st
from dotenv import load_dotenv

from src.myragchatbot.core.query_engine import QueryEngine
from src.myragchatbot.rerankers.cohere_reranker import CohereReranker

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot with Internet Search + Prompt Switching + Reranking")

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- LLM, Embedding & Prompt Config ---
llm_choice = st.selectbox("🧠 Choose LLM backend:", ["openai", "ollama"])
embedding_choice = st.selectbox("📌 Choose Embedding backend:", ["openai", "ollama"])
prompt_choice = st.selectbox("📄 Prompt style:", ["default", "story", "qa", "summary"])
use_internet = st.checkbox("🌍 Use Internet Search (Tavily)?", value=True)
use_reranker = st.checkbox("🏗️ Use Cohere Reranker?", value=True)

# --- RAG Config ---
top_k = st.slider("🔍 Top K Documents to Retrieve", 1, 10, 5)
rerank_threshold = st.slider("🎯 Rerank Threshold", 0.0, 1.0, 0.4, step=0.05)

# --- Initialize Query Engine ---
if (
    "engine" not in st.session_state
    or st.session_state.get("llm_choice") != llm_choice
    or st.session_state.get("embedding_choice") != embedding_choice
):
    st.session_state["llm_choice"] = llm_choice
    st.session_state["embedding_choice"] = embedding_choice
    st.session_state["engine"] = QueryEngine(
        llm_backend=llm_choice,
        embedding_backend=embedding_choice
    )

query_engine: QueryEngine = st.session_state["engine"]

# --- Upload File ---
uploaded_file = st.file_uploader("📤 Upload PDF or TXT file", type=["pdf", "txt"])
if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    query_engine.load_and_index_file(file_path)
    st.success(f"✅ `{uploaded_file.name}` uploaded and indexed.")

# --- Ask Question ---
st.markdown("---")
query = st.text_input("💬 Ask a question (e.g., 'What happened to the main character?')")

if query:
    with st.spinner("🔍 Searching, reranking, and thinking..."):
        # Run core RAG
        answer, docs = query_engine.answer_query(
            question=query,
            top_k=top_k,
            use_internet=use_internet,
            prompt_type=prompt_choice
        )

        # Reranking
        if use_reranker:
            reranker = CohereReranker(threshold=rerank_threshold)
            reranked = reranker.rerank(query, docs)
        else:
            reranked = [{"document": doc, "relevance_score": 1.0} for doc in docs]

        # Show Answer
        st.markdown("### 💡 Answer")
        st.markdown(answer)

        # Show Sources
        st.markdown("### 📚 Reranked Documents")
        for i, item in enumerate(reranked, 1):
            score = item["relevance_score"]
            doc = item["document"]
            with st.expander(f"Doc {i} - Score: {score:.3f}"):
                st.markdown(doc.page_content[:1000])
