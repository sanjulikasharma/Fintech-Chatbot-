import os
from dotenv import load_dotenv
from langfuse.openai import openai

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.chat.completions.create(
    name="test-chat",
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a very accurate calculator. You output only the result of the calculation."
        },
        {"role": "user", "content": "1 + 1 = "}
    ],
    temperature=0,
    metadata={"someMetadataKey": "someValue"},
)

print(completion.choices[0].message.content)