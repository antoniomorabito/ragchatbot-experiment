from langchain.prompts import PromptTemplate

# 1. Default RAG Prompt
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

# 2. Story-Focused Prompt (Fiction/Non-Fiction)
story_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a literary assistant helping someone understand a story (fiction or non-fiction).
Analyze the story structure, character motivations, events, and themes.
Base your answer strictly on the given context. If you cannot find an answer, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)

# 3. Summarization Prompt
summary_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
You are a summarization expert. Read the following context and write a concise summary that captures the key points.

Context:
{context}

Summary:
"""
)

# 4. Q&A Prompt with Citations
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a research assistant answering questions based strictly on the context below.
If possible, cite specific phrases from the context in your response.
If the answer is not found, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)
