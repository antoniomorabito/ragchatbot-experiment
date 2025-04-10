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
            for root, _, files in os.walk(self.path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        full_path = os.path.join(root, file)
                        try:
                            documents.extend(self._load_single_pdf(full_path))
                        except Exception as e:
                            print(f"[ERROR] Failed to load {file}: {e}")
        elif os.path.isfile(self.path) and self.path.lower().endswith(".pdf"):
            try:
                documents.extend(self._load_single_pdf(self.path))
            except Exception as e:
                print(f"[ERROR] Failed to load file {self.path}: {e}")
        else:
            print(f"[WARNING] No valid PDF found at path: {self.path}")

        if not documents:
            raise ValueError(f"No PDF documents could be parsed from: {self.path}")

        return documents

    def _load_single_pdf(self, file_path: str) -> List[Document]:
        print(f"[DEBUG] Trying to load: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        print(f"[DEBUG] Loaded {len(pages)} pages from {file_path}")

        for i, doc in enumerate(pages):
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["page_number"] = i + 1

        return pages

