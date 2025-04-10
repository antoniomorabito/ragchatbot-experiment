from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiChat:
    def __init__(self, temperature: float = 0.0):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
