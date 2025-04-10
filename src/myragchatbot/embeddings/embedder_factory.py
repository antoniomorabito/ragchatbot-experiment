import os
from typing import Literal
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings

def get_embedder(
    backend: Literal["openai", "ollama", "huggingface"] = "openai"
):
    if backend == "openai":
        return OpenAIEmbeddings(openai_api_key=os.environ["OPEN_API_KEY"])
    elif backend == "ollama":
        return OllamaEmbeddings(model="nomic-embed-text")
    elif backend == "huggingface":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")
