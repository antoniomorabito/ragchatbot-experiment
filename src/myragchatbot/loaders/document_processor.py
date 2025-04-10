import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import streamlit as st


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def load_documents(self, data_directory: str) -> List[Document]:
        """Load PDF and TXT documents from a directory."""
        try:
            if not os.path.exists(data_directory):
                st.error(f"Directory does not exist: {data_directory}")
                return []

            documents = []
            for filename in os.listdir(data_directory):
                file_path = os.path.join(data_directory, filename)
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load_and_split()
                    for i, doc in enumerate(docs):
                        doc.metadata["source"] = filename
                        doc.metadata["page_number"] = i + 1
                    documents.extend(docs)

                elif filename.lower().endswith(".txt"):
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = filename
                        doc.metadata.setdefault("page_number", 1)  # Default page for TXT
                    documents.extend(docs)

            st.success(f"Loaded {len(documents)} documents from `{data_directory}`")

            for i, doc in enumerate(documents[:5]):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page_number", "?")
                preview = doc.page_content[:150].strip().replace("\n", " ")
                st.markdown(f"**Doc {i+1}** | Page: {page} | Source: `{source}`")
                st.code(preview, language="markdown")

            return documents

        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            splits = self.text_splitter.split_documents(documents)
            st.success(f"Split into {len(splits)} chunks (chunk size: {self.chunk_size}, overlap: {self.chunk_overlap})")

            for i, chunk in enumerate(splits[:5]):
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page_number", "?")
                content = chunk.page_content[:200].strip().replace("\n", " ")
                st.markdown(f"**Chunk {i+1}** | Page: {page} | Source: `{source}`")
                st.code(content, language="markdown")

            return splits
        except Exception as e:
            st.error(f"Error splitting documents: {str(e)}")
            return documents