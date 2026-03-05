import pandas as pd
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter 


load_dotenv()
api = os.getenv("OPENAI_API_KEY")

class DataEngine:

    def __init__(self, file_path):
        self.file_path = file_path
        self.vector_db = self._prepare_data()

    def _prepare_data(self):

        df = pd.read_csv(self.file_path, encoding="latin-1")

        documents = []

        for _, row in df.iterrows():

            content = f"""
Company: {row['Company']}
Sector: {row['primary_sector']}
Location: {row['location']}
Founded: {row['founded_year']}

Description:
{row['company_background']}
"""

            documents.append(Document(page_content=content))

        # -------- Chunking --------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        split_docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        vector_db = FAISS.from_documents(split_docs, embeddings)

        return vector_db