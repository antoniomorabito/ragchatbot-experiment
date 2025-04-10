import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


class OpenAIChat:
    def __init__(self, temperature=0.0, model="gpt-4o"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.environ["OPEN_API_KEY"]  
        )

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
