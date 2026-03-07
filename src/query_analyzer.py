from langfuse.openai import openai
import os
import json 
import re 
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
    try: 

        response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You output only valid JSON. No explanations, no markdown."},
        {"role": "user", "content": prompt}], 
        temperature = 0 # Deterministic output for parsing
    )

        output = response.choices[0].message.content.strip()
        
        # Extract JSON if wrapped in markdown or extra text
        json_match = re.search(r'\{[\s\S]*\}', output)
        if json_match:
            output = json_match.group()
        
        result = json.loads(output)
        
        is_ambiguous = result.get("status") == "ambiguous"
        clarification = result.get("clarification") if is_ambiguous else None
        
        return is_ambiguous, clarification

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        # Fallback: if JSON parsing fails, treat as clear to avoid blocking user
        print(f"[WARN] Query analysis failed: {e}")
        return False, None
    
    except Exception as e:
        # Catch-all for API/network errors
        print(f"[ERROR] analyze_query exception: {e}")
        return False, None