from langfuse.openai import openai 
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from src.prompt import PROMPT
import os 
openai.api_key = os.getenv("OPENAI_API_KEY")


class UnicornChatbot:

    def __init__(self, retriever):
        self.retriever = retriever

    def ask(self, query):

        docs = self.retriever.invoke(query)

        context = "\n".join([d.page_content for d in docs])

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert on Indian Unicorn Startups. Use context only."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:{query}"
                }
            ],
        )

        return response.choices[0].message.content