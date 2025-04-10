import ollama
from typing import Generator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class OllamaChat(BaseChatModel):
    def __init__(self, model: str = "llama3.2:latest", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def _convert_prompt(self, prompt: str) -> list:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

    def invoke(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=self._convert_prompt(prompt),
            temperature=self.temperature
        )
        return response["message"]["content"]

    def stream(self, prompt: str, callbacks=None) -> Generator[AIMessage, None, None]:
        response_stream = ollama.chat(
            model=self.model,
            messages=self._convert_prompt(prompt),
            temperature=self.temperature,
            stream=True
        )

        full_content = ""
        for chunk in response_stream:
            token = chunk.get("message", {}).get("content", "")
            full_content += token
            yield AIMessage(content=token)  # streaming per token

        # Optional: return final ChatResult
        # yield ChatResult(generations=[ChatGeneration(message=AIMessage(content=full_content))])

    @property
    def _llm_type(self) -> str:
        return "ollama-chat"
