from typing import List, Generator, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import ollama

class OllamaChat(BaseChatModel):
    model: str = "llama3"
    temperature: float = 0.0

    def _convert_prompt(self, messages: List[HumanMessage]) -> list:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": messages[-1].content}
        ]

    def _generate(
        self,
        messages: List[HumanMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        converted = self._convert_prompt(messages)
        response = ollama.chat(
            model=self.model,
            messages=converted,
            temperature=self.temperature
        )
        content = response["message"]["content"]
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def invoke(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=self._convert_prompt([HumanMessage(content=prompt)]),
            temperature=self.temperature
        )
        return response["message"]["content"]

    def stream(self, prompt: str, callbacks=None) -> Generator[AIMessage, None, None]:
        response_stream = ollama.chat(
            model=self.model,
            messages=self._convert_prompt([HumanMessage(content=prompt)]),
            temperature=self.temperature,
            stream=True
        )
        for chunk in response_stream:
            token = chunk.get("message", {}).get("content", "")
            yield AIMessage(content=token)

    @property
    def _llm_type(self) -> str:
        return "ollama-chat"
