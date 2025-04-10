import google.generativeai as genai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class GeminiChat(BaseChatModel):
    def __init__(self, model="models/gemini-pro", temperature=0.0):
        self.model = model
        self.temperature = temperature
        self.client = genai.GenerativeModel(model)

    def invoke(self, prompt: str) -> str:
        response = self.client.generate_content(prompt)
        return response.text
