import re
from langfuse.openai import openai


class Reranker:

    def rerank(self, query, docs):

        scored_docs = []

        for doc in docs:

            prompt = f"""
You are a relevance scoring system.

Score how relevant the DOCUMENT is to the QUESTION from 1 to 10.

Return ONLY the number.

QUESTION:
{query}

DOCUMENT:
{doc.page_content}
"""

            res = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            output = res.choices[0].message.content.strip()

            # Extract number safely
            match = re.search(r"\d+", output)

            score = int(match.group()) if match else 0

            scored_docs.append((score, doc))

        scored_docs.sort(reverse=True, key=lambda x: x[0])

        return [d[1] for d in scored_docs[:3]]