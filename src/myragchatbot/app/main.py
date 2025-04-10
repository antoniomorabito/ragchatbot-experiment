import os
import streamlit as st
from dotenv import load_dotenv

from src.myragchatbot.core.query_engine import QueryEngine
from src.myragchatbot.rerankers.cohere_reranker import CohereReranker
from src.myragchatbot.loaders.pdf_loader import PDFLoader
from src.myragchatbot.loaders.text_loader import TextFileLoader
from src.myragchatbot.evaluation.rerank_evaluator import evaluate_reranking
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot with Internet Search, MMR & Cohere Reranker")

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- State Init ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- LLM, Embedding & Prompt Config ---
llm_choice = st.selectbox("LLM backend:", ["openai", "ollama"], index=1)
embedding_choice = st.selectbox("Embedding backend:", ["openai", "ollama"], index=1)
prompt_choice = st.selectbox("Prompt style:", ["default", "story", "qa", "summary"])
use_internet = st.checkbox("Use Internet Search (Tavily)?", value=True)
use_reranker = st.checkbox("Use Cohere Reranker?", value=False)
use_mmr = st.checkbox("Use MMR Retriever?", value=False)
temperature = st.slider("Model Temperature", 0.0, 1.0, 0.0, step=0.1)

# --- RAG Config ---
top_k = st.slider("Top K Documents", 1, 10, 5)
rerank_threshold = st.slider("Rerank Threshold", 0.0, 1.0, 0.4, step=0.05)

# --- Init Engine ---
if (
    "engine" not in st.session_state
    or st.session_state.get("llm_choice") != llm_choice
    or st.session_state.get("embedding_choice") != embedding_choice
    or st.session_state.get("temperature") != temperature
):
    with st.spinner("Initializing engine and loading documents..."):
        st.session_state["llm_choice"] = llm_choice
        st.session_state["embedding_choice"] = embedding_choice
        st.session_state["temperature"] = temperature
        st.session_state["engine"] = QueryEngine(
            llm_backend=llm_choice,
            embedding_backend=embedding_choice,
            temperature=temperature,
        )
        st.success("Query Engine initialized.")

query_engine: QueryEngine = st.session_state["engine"]

# --- Upload File ---
uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])
if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Index ke vectorstore
    query_engine.load_and_index_file(file_path)
    st.success(f"`{uploaded_file.name}` uploaded and indexed.")

    #Show chunk summary hanya untuk file yang baru saja diupload
    if uploaded_file.name.lower().endswith(".pdf"):
        loader = PDFLoader(file_path)
    elif uploaded_file.name.lower().endswith(".txt"):
        loader = TextFileLoader(file_path)
    else:
        loader = None

    if loader:
        docs = loader.load_documents()
        splits = query_engine.processor.split_documents(docs)

        with st.expander("View Chunking Summary"):
            st.markdown(f"**File uploaded:** `{uploaded_file.name}`")
            st.markdown(f"**Total chunks created:** {len(splits)}")

            for i, chunk in enumerate(splits):
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page_number", "?")
                preview = chunk.page_content[:300].strip().replace("\n", " ")
                st.markdown(f"**Chunk {i+1}** | Page: {page} | Source: `{source}`")
                st.code(preview, language="markdown")

# --- Clear Chat ---
if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []

# --- Display Chat History ---
st.markdown("### Chat")
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
query = st.chat_input("Ask a question about the story...")

if query:
    st.session_state["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant thinking
    with st.chat_message("assistant"):
        with st.spinner("Searching, reranking, and thinking..."):
            answer, docs, debug_context = query_engine.answer_query(
                question=query,
                top_k=top_k,
                use_internet=use_internet,
                prompt_type=prompt_choice,
                use_mmr=use_mmr,
            )
            st.markdown(answer)

    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    # Reranked Docs
    if use_reranker:
        reranker = CohereReranker(threshold=rerank_threshold)
        reranked = reranker.rerank(query, docs)
    else:
        reranked = [{"document": doc, "relevance_score": 1.0} for doc in docs]

    st.markdown("###  Reranked Documents")
    for i, item in enumerate(reranked, 1):
        score = item["relevance_score"]
        doc = item["document"]
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number", "?")

        with st.expander(f"Doc {i} - Score: {score:.3f} | Page: {page} | File: {source}"):
            st.markdown(doc.page_content[:1000])
    
    
    
    k_eval = min(top_k, len(reranked))  
    metrics = evaluate_reranking(reranked, k=k_eval)

   
    st.markdown("### Reranking Evaluation")
    st.metric("Precision@k", f"{metrics['precision@k']:.2f}")
    st.metric("Recall@k", f"{metrics['recall@k']:.2f}")
    st.metric("MAP", f"{metrics['MAP']:.2f}")

    #Tambahan: penjelasan metrik
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Precision@k**: Proporsi dokumen di top-k yang memang relevan.
        - **Recall@k**: Seberapa banyak dokumen relevan yang berhasil ditemukan dari total dokumen relevan.
        - **MAP (Mean Average Precision)**: Rata-rata precision dari posisi di mana dokumen relevan muncul.
        """)

    # Internet Context
    if use_internet:
        st.markdown("###  Internet Search Context")
        for i, c in enumerate(debug_context[-top_k:], 1):
            st.markdown(f"**[{i}]** {c}")
