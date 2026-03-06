from langfuse.openai import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def analyze_query(query: str):

    prompt = f"""
You are a query analyzer for a company dataset chatbot.

Determine if the user's query is clear enough to answer using company data.

Return JSON with:
- "status": "clear" or "ambiguous"
- "clarification": question to ask if ambiguous

Examples:

Query: Which fintech startups exist?
{{"status":"clear"}}

Query: Which company is best?
{{"status":"ambiguous","clarification":"Best according to what metric? revenue, valuation, or growth?"}}

Query: Which company should I collaborate with?
{{"status":"ambiguous","clarification":"What type of collaboration are you looking for (technology, payments, logistics)?"}}

User Query:
{query}

Return JSON only.
"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    output = response.choices[0].message.content.strip()

    if output.startswith("AMBIGUOUS"): 
        clarification = output.replace("AMBIGUOUS:", "").strip()
        return True, clarification
    
    return False, None 