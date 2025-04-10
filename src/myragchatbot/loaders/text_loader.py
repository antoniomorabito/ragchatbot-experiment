import os
from typing import List
from langchain_core.documents import Document

class TextFileLoader:
    def __init__(self, directory: str):
        self.directory = directory

    def load_documents(self) -> List[Document]:
        documents = []
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        for filename in os.listdir(self.directory):
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(self.directory, filename)
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    #  Masukkan metadata: source + halaman
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page_number": 1  # default 1 for .txt
                        }
                    )
                    documents.append(doc)

        return documents
