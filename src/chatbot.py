from langfuse.openai import openai 
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from src.prompt import PROMPT_TEMPLATE
from src.reranker1 import Reranker
from src.query_analyzer import analyze_query
from src.query_router import route_query 

import json

from src.cache import get_cached, set_cache
import os 
openai.api_key = os.getenv("OPENAI_API_KEY")


class UnicornChatbot:

    def __init__(self, retriever):
        self.retriever = retriever
        self.reranker1 = Reranker()
        self.history = [] #conversational memory

    def ask(self, query):

        cache_key = query
        if self.history:
            cache_key = f"{self.history[-1]['user']} {query}"

        cached = get_cached(cache_key)
        if cached:
            return cached

        # Query rewriting
        context_query = query
        if self.history:
            recent_context = " ".join([h["user"] for h in self.history[-2:]])
            context_query = f"{recent_context} {query}"

        # --- 3. Route + Analyze the REWRITTEN query ---
        query_type = route_query(context_query)  # ROUTE after rewriting
        
        # If router says CLARIFICATION, get specific question from analyzer
        if query_type == "CLARIFICATION":
            is_ambiguous, clarification = analyze_query(context_query)
            if clarification:
                return clarification

        # --- 4. Retrieval Strategy based on query type ---
        if query_type == "FILTER":
            # Broader search for list/filter questions
            docs = self.retriever.invoke(context_query, search_kwargs={"k": 10})
        else:
            # Standard precision-focused retrieval
            docs = self.retriever.invoke(context_query)

        # Reranking
        docs = self.reranker1.rerank(context_query, docs)

        context = "\n\n".join([d.page_content for d in docs])

        history_text = "\n".join(
            [f"User: {h['user']}\nBot: {h['bot']}" for h in self.history[-5:]]
        )

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=query,
            history=history_text
        )

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        set_cache(cache_key, answer)

        self.history.append({"user": query, "bot": answer})

        if len(self.history) > 10:
            self.history.pop(0)

        return answer if answer else "I couldn't find the answer in the dataset"