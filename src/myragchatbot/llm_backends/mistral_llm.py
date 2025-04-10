from langchain_community.chat_models import ChatOllama

class MistralChat:
    def __init__(self, model: str = "mistral:latest", temperature: float = 0.0):
        self.llm = ChatOllama(model=model, temperature=temperature)

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
