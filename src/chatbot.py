from langfuse.openai import openai 
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from src.prompt import PROMPT_TEMPLATE
from src.reranker import Reranker

from src.cache import get_cached, set_cache
import os 
openai.api_key = os.getenv("OPENAI_API_KEY")


class UnicornChatbot:

    def __init__(self, retriever):
        self.retriever = retriever
        self.reranker = Reranker()

    def ask(self, query):
        cached = get_cached(query)

        if cached:
            return cached
        docs = self.retriever.invoke(query)


        # -------- Reranking --------
        docs = self.reranker.rerank(query, docs)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer =  response.choices[0].message.content
        set_cache(query, answer)

        return answer if answer else "I couldn't find the answer in the dataset"