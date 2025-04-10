# src/myragchatbot/loaders/pdf_loader.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, path: str = "data"):
        self.path = path

    def load_documents(self) -> List[Document]:
        """Load PDF files from a directory or a single PDF file, with page metadata."""
        documents = []

        if os.path.isdir(self.path):
            # Load semua PDF dari folder
            for root, _, files in os.walk(self.path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        full_path = os.path.join(root, file)
                        documents.extend(self._load_single_pdf(full_path))
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            documents.extend(self._load_single_pdf(self.path))
        else:
            raise FileNotFoundError(f"Path not found or invalid PDF: {self.path}")

        if not documents:
            raise ValueError("No PDF documents found!")

        return documents

    def _load_single_pdf(self, file_path: str) -> List[Document]:
        """Load and annotate pages from a single PDF file."""
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        for i, doc in enumerate(pages):
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["page_number"] = i + 1  # page numbers start at 1

        return pages
