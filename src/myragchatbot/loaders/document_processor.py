import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
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
        """Load documents from a directory."""
        try:
            if not os.path.exists(data_directory):
                st.error(f"Directory does not exist: {data_directory}")
                return []

            loader = DirectoryLoader(
                data_directory,
                glob="**/*.*",
                show_progress=True,
            )

            documents = loader.load()
            st.success(f"Loaded {len(documents)} documents from `{data_directory}`")

            # Optional detail preview
            for i, doc in enumerate(documents):
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

            # Optional: show first few chunk previews
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
