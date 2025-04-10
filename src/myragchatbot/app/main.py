import os
import streamlit as st
from dotenv import load_dotenv
from time import time

from src.myragchatbot.core.query_engine import QueryEngine
from src.myragchatbot.rerankers.cohere_reranker import CohereReranker
from src.myragchatbot.loaders.pdf_loader import PDFLoader
from src.myragchatbot.loaders.text_loader import TextFileLoader
from src.myragchatbot.evaluation.rerank_evaluator import evaluate_reranking

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("AI Story QA | RAG with Internet, Reranker & Model Test")

UPLOAD_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- STATE INIT ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "engine" not in st.session_state:
    st.session_state["engine"] = None

# --- CONFIG UI ---
llm_choices = st.multiselect("Compare LLMs:", ["openai", "ollama", "gemini", "mistral"], default=["openai", "ollama"])
embedding_choice = st.selectbox("Embedding backend:", ["openai", "ollama", "huggingface"], index=1)
prompt_choice = st.selectbox("Prompt style:", ["default", "story", "qa", "summary"])
use_internet = st.checkbox("Use Internet Search (Tavily)?", value=True)
use_reranker = st.checkbox("Use Cohere Reranker?", value=False)
use_mmr = st.checkbox("Use MMR Retriever?", value=False)
temperature = st.slider("Model Temperature", 0.0, 1.0, 0.0, step=0.1)
top_k = st.slider("Top K Documents", 1, 10, 5)
rerank_threshold = st.slider("Rerank Threshold", 0.0, 1.0, 0.4, step=0.05)

# --- File Upload + Indexing ---
query_engine = None
vectorstore_path = os.path.join(VECTORSTORE_DIR, embedding_choice, "chroma.sqlite3")

if st.button("Generate Vectorstore for Selected Embedding"):
    with st.spinner("Creating vectorstore..."):
        query_engine = QueryEngine(
            llm_backend=llm_choices[0],  # default llm untuk indexing
            embedding_backend=embedding_choice,
            temperature=temperature,
        )
        st.session_state["engine"] = query_engine
        st.success(f"Vectorstore created using `{embedding_choice}`.")

if st.session_state["engine"] is None and os.path.exists(vectorstore_path):
    with st.spinner("Loading existing vectorstore..."):
        query_engine = QueryEngine(
            llm_backend=llm_choices[0],
            embedding_backend=embedding_choice,
            temperature=temperature,
        )
        st.session_state["engine"] = query_engine
        st.success("Existing vectorstore loaded.")

query_engine = st.session_state["engine"]

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
if uploaded_file and query_engine:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    query_engine.load_and_index_file(file_path)
    st.success(f"`{uploaded_file.name}` uploaded and indexed.")

    if uploaded_file.name.endswith(".pdf"):
        loader = PDFLoader(file_path)
    else:
        loader = TextFileLoader(file_path)

    docs = loader.load_documents()
    splits = query_engine.processor.split_documents(docs)

    with st.expander("Chunking Summary"):
        st.markdown(f"**Total Chunks:** {len(splits)}")
        for i, chunk in enumerate(splits[:5]):
            st.markdown(f"**Chunk {i+1}** — Page: {chunk.metadata.get('page_number', '?')}")
            st.code(chunk.page_content[:300])

if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []

st.markdown("###Chat")
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the story...")
if query and query_engine:
    st.session_state["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Multi-LLM comparison
    st.markdown("### LLM Comparison")
    for llm in llm_choices:
        with st.chat_message("assistant"):
            with st.spinner(f"Answering with {llm}..."):
                engine = QueryEngine(
                    llm_backend=llm,
                    embedding_backend=embedding_choice,
                    temperature=temperature,
                )

                start_time = time()
                answer, docs, debug_context = engine.answer_query(
                    question=query,
                    top_k=top_k,
                    use_internet=use_internet,
                    prompt_type=prompt_choice,
                    use_mmr=use_mmr,
                )
                duration = time() - start_time

                st.markdown(f"**{llm.upper()}** _({duration:.2f}s)_: {answer}")

                if use_reranker:
                    reranker = CohereReranker(threshold=rerank_threshold)
                    reranked = reranker.rerank(query, docs)
                else:
                    reranked = [{"document": doc, "relevance_score": 1.0} for doc in docs]

                for r in reranked:
                    r["is_relevant"] = r["relevance_score"] >= rerank_threshold

                with st.expander(f"{llm.upper()} - Reranked Documents"):
                    for i, item in enumerate(reranked, 1):
                        score = item["relevance_score"]
                        doc = item["document"]
                        page = doc.metadata.get("page_number", "?")
                        st.markdown(f"**Doc {i}** — Score: {score:.2f} — Page: {page}")
                        st.code(doc.page_content[:500])

                k_eval = min(top_k, len(reranked))
                metrics = evaluate_reranking(reranked, k=k_eval)
                st.markdown("**Evaluation**")
                st.metric("Precision@k", f"{metrics['precision@k']:.2f}")
                st.metric("Recall@k", f"{metrics['recall@k']:.2f}")
                st.metric("MAP", f"{metrics['MAP']:.2f}")

                if use_internet:
                    with st.expander("Internet Context"):
                        for i, c in enumerate(debug_context[-top_k:], 1):
                            st.markdown(f"**[{i}]** {c}")

elif not query_engine:
    st.warning("Please generate or load a vectorstore first.")