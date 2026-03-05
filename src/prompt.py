from langchain_classic.prompts import PromptTemplate

template = """
You are an expert on Indian Unicorn Startups.

Rules:
1. Use the provided context only.
2. If unsure, say you don't know.
3. Ask clarification if the question is vague.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "chat_history", "question"]
)