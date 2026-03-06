import logging
from langfuse import get_client

from src.data_engine import DataEngine
from src.retriever import get_retriever
from src.chatbot import UnicornChatbot

from dotenv import load_dotenv
import os
load_dotenv()

api = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

langfuse = get_client()


def sanitize_input(text: str):
    return text.strip()[:500]


def main():

    print("Initializing Startup Insight Bot...")

    engine = DataEngine("data/startups.csv")
    retriever = get_retriever(engine.vector_db)
    bot = UnicornChatbot(retriever)

    print("Ready! Type 'exit' to quit.")

    while True:

        user_input = input("\nYou: ").strip()

        # normalize once
        command = user_input.lower()

        if command in ["exit", "quit"]:
            print("Goodbye!")
            break

        query = sanitize_input(user_input)

        logging.info(f"User query: {query}")

        try:

            with langfuse.start_as_current_observation(
                as_type="span",
                name="chatbot-request"
            ) as span:

                span.update(input={"query": query})

                answer = bot.ask(query)

                span.update(output={"response": answer})

                print("Bot:", answer)

                logging.info(f"Bot response: {answer}")

        except Exception as e:

            logging.error(f"Error: {str(e)}")
            print("ERROR:", e)


if __name__ == "__main__":
    main()