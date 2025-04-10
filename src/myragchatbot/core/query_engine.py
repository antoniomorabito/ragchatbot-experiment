import os
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from src.myragchatbot.embeddings.embedder_factory import get_embedder
from src.myragchatbot.loaders.pdf_loader import PDFLoader
from src.myragchatbot.loaders.text_loader import TextFileLoader
from src.myragchatbot.llm_backends.openai_llm import OpenAIChat
from src.myragchatbot.internet_search.tavily_search import run_tavily_search
from src.myragchatbot.core.prompt_template import (
    default_rag_prompt,
    story_prompt,
    summary_prompt,
    qa_prompt
)

class QueryEngine:
    def __init__(
        self,
        llm_backend: str = "openai",
        embedding_backend: str = "openai",
        persist_dir: str = "vectorstore",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Embeddings via factory
        self.embedder = get_embedder(embedding_backend)

        # Vector DB
        self.vectorstore = Chroma(
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
            collection_name="docs"
        )

        # LLM backend
        if llm_backend == "openai":
            self.llm = OpenAIChat(temperature=0)
        elif llm_backend == "ollama":
            from src.myragchatbot.llm_backends.ollama_llm import OllamaChat
            self.llm = OllamaChat(model="llama3.2:latest", temperature=0)
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}")

    def load_and_index_file(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            docs = PDFLoader(file_path).load_documents()
        elif ext == ".txt":
            docs = TextFileLoader(file_path).load_documents()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = self.splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)

    def answer_query(
        self,
        question: str,
        top_k: int = 5,
        use_internet: bool = False,
        prompt_type: str = "default"
    ) -> Tuple[str, List[Document]]:
        
        docs = self.vectorstore.similarity_search(question, k=top_k)
        context_chunks = [doc.page_content for doc in docs]

        if use_internet:
            try:
                tavily_results = run_tavily_search(question)
                context_chunks.extend(tavily_results)
            except Exception:
                context_chunks.append("⚠️ Tavily search failed.")

        context = "\n\n".join(context_chunks)

        # Choose prompt
        if prompt_type == "story":
            prompt = story_prompt.format(context=context, question=question)
        elif prompt_type == "qa":
            prompt = qa_prompt.format(context=context, question=question)
        elif prompt_type == "summary":
            prompt = summary_prompt.format(context=context)
        else:
            prompt = default_rag_prompt.format(context=context, question=question)

        response = self.llm.invoke(prompt)
        return response, docs
