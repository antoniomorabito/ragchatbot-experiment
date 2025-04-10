import os
from typing import List, Tuple, Dict
from pathlib import Path

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from src.myragchatbot.loaders.pdf_loader import PDFLoader
from src.myragchatbot.loaders.text_loader import TextFileLoader
from src.myragchatbot.llm_backends.openai_llm import OpenAIChat
from src.myragchatbot.core.prompt_template import default_rag_prompt

from langchain_openai import OpenAIEmbeddings


class QueryEngine:
    def __init__(self, persist_dir="vectorstore", chunk_size=1000, chunk_overlap=200):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        self.embedder = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
            collection_name="docs"
        )

        self.llm = OpenAIChat(temperature=0)  # Default to OpenAI, will be dynamic later

    def load_and_index_file(self, file_path: str):
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            docs = PDFLoader("data").load_documents()
        elif ext == ".txt":
            docs = TextFileLoader("data").load_documents()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = self.splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)

    def answer_query(self, question: str, top_k: int = 5) -> Tuple[str, List[Document]]:
        docs = self.vectorstore.similarity_search(question, k=top_k)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = default_rag_prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt)

        return response, docs
