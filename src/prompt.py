from langchain_classic.prompts import PromptTemplate


PROMPT_TEMPLATE = """
You are an expert on Indian Unicorn Startups.

Rules:
- Only answer using the provided context
- If the answer is not in the context, say you do not know
- Be concise and factual
-Use the conversation history to understand follow-up questions. 

Conversation History: 
{history}

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "chat_history", "question"]
)