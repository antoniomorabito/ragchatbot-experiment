# src/myragchatbot/loaders/pdf_loader.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, path: str = "data"):
        self.path = path

    def load_documents(self) -> List[Document]:
        """Load PDF files from a directory or a single PDF file."""
        if os.path.isdir(self.path):
            loader = DirectoryLoader(
                self.path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
            )
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            loader = PyPDFLoader(self.path)
        else:
            raise FileNotFoundError(f"Path not found or invalid PDF: {self.path}")

        documents = loader.load()
        if not documents:
            raise ValueError("No PDF documents found!")

        return documents
