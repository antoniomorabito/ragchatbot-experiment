
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class GeminiChat(BaseChatModel):
    def __init__(self, temperature=0.0):
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)

    def _generate(
        self,
        messages: List[HumanMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        response = self.model.invoke(messages)
        return ChatResult(generations=[ChatGeneration(message=response)])

    def invoke(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    def _llm_type(self) -> str:
        return "gemini-chat"
