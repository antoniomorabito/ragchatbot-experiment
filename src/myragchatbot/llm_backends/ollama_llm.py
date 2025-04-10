from langchain_community.llms import Ollama

class OllamaChat:
    def __init__(self, model="llama3", temperature=0.0):
        self.llm = Ollama(model=model, temperature=temperature)

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
