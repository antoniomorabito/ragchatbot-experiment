import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document


class PDFLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_documents(self) -> List[Document]:
        """Load PDF files from the directory into Document chunks"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )

        documents = loader.load()
        if not documents:
            raise ValueError("No PDF documents found!")

        return documents
