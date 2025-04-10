from langchain.prompts import PromptTemplate

# Default RAG prompt template

default_rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant tasked with answering questions based on provided context.
Use only the given context to formulate a response. If the answer is not in the context, respond with "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)
