import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from src.myragchatbot.embeddings.embedder_factory import get_embedder
from src.myragchatbot.loaders.pdf_loader import PDFLoader
from src.myragchatbot.loaders.text_loader import TextFileLoader
from src.myragchatbot.loaders.document_processor import DocumentProcessor
from src.myragchatbot.llm_backends.openai_llm import OpenAIChat
from src.myragchatbot.llm_backends.ollama_llm import OllamaChat
from src.myragchatbot.internet_search.tavily_search import run_tavily_search
from src.myragchatbot.core.prompt_template import (
    default_rag_prompt, story_prompt, summary_prompt, qa_prompt
)

class QueryEngine:
    def __init__(
        self,
        llm_backend: str = "openai",
        embedding_backend: str = "openai",
        persist_dir: str = "vectorstore",
        data_dir: str = "data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedder = get_embedder(embedding_backend)

        self.vectorstore = Chroma(
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
            collection_name="docs"
        )

        if llm_backend == "openai":
            self.llm = OpenAIChat(temperature=0)
        elif llm_backend == "ollama":
            self.llm = OllamaChat(model="llama3.2:latest", temperature=0)
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}")

        # Load dokumen otomatis pertama kali dijalankan
        if self.vectorstore._collection.count() == 0:
            self.load_initial_documents(data_dir)

    def load_initial_documents(self, data_dir: str):
        documents = self.processor.load_documents(data_dir)
        if documents:
            splits = self.processor.split_documents(documents)
            self.vectorstore.add_documents(splits)

    def load_and_index_file(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PDFLoader(file_path)
        elif ext == ".txt":
            loader = TextFileLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        documents = loader.load_documents()
        if documents:
            splits = self.processor.split_documents(documents)
            self.vectorstore.add_documents(splits)

    def answer_query(
        self,
        question: str,
        top_k: int = 5,
        use_internet: bool = False,
        use_mmr=False,
        prompt_type: str = "default"
    ) -> Tuple[str, List[Document]]:
        
        if use_mmr:
                docs = self.vectorstore.max_marginal_relevance_search(question, k=top_k, fetch_k=top_k * 2)
        else:
                docs = self.vectorstore.similarity_search(question, k=top_k)

        context_chunks = [doc.page_content for doc in docs]

        if use_internet:
            try:
                tavily_results = run_tavily_search(question)
                context_chunks.extend(tavily_results)
                print("[DEBUG] Tavily returned results:")
                for res in tavily_results:
                    print(res)
            except Exception:
                context_chunks.append("Tavily search failed.")

        context = "\n\n".join(context_chunks)

        if prompt_type == "story":
            prompt = story_prompt.format(context=context, question=question)
        elif prompt_type == "qa":
            prompt = qa_prompt.format(context=context, question=question)
        elif prompt_type == "summary":
            prompt = summary_prompt.format(context=context)
        else:
            prompt = default_rag_prompt.format(context=context, question=question)

        response = self.llm.invoke(prompt)
        return response, docs,context_chunks
