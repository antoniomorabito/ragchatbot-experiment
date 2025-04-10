import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
import streamlit as st


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
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
            st.info(f"Loaded {len(documents)} documents from {data_directory}")
            return documents

        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            splits = self.text_splitter.split_documents(documents)
            st.info(f"Split documents into {len(splits)} chunks")
            return splits
        except Exception as e:
            st.error(f"Error splitting documents: {str(e)}")
            return documents
