from langfuse.openai import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def route_query(query: str):

    prompt = f"""
You are a query router for a chatbot that answers questions about Indian unicorn startups.

Classify the user query into one of the following categories:

FACTUAL - asking about a specific company
FILTER - asking to list companies by sector or category
CLARIFICATION - vague question that needs a follow-up question

Examples:

Query: What does Razorpay do?
FACTUAL

Query: List fintech unicorn startups
FILTER

Query: Which company is best to collaborate with?
CLARIFICATION

Query: Which startup is the best?
CLARIFICATION

User Query:
{query}

Return only one word: FACTUAL, FILTER, or CLARIFICATION.
"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()