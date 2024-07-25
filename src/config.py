import os

from langchain_cohere import ChatCohere, CohereRerank
from langchain_fireworks import ChatFireworks
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pymongo import MongoClient

CONNECTION_STRING = os.getenv('MONGODB_CONNECTION_STRING')
DATABASE_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME') + '.full.2048.128__v4'
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')
PDF_CHAR_SPLITTER_CHUNK_SIZE = 2048
PDF_CHAR_SPLITTER_CHUNK_OVERLAP = 128
TEMPERATURE = 0
TOP_K = 4

client = MongoClient(CONNECTION_STRING)
COLLECTION = client[DATABASE_NAME][COLLECTION_NAME]
HISTORY_COLLECTION_NAME = "message_store"

llm_map = {
    "chatgpt": ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    "gemini": ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0),
    "cohere": ChatCohere(model="command-r-plus", temperature=0),
    "ollamma": ChatOllama(model="llama3", temperature=0),
    "groq": ChatGroq(model_name="llama3-70b-8192", temperature=0),
    "fireworks": ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct", temperature=0)
}

llm_label_map = {
    # "chatgpt": "OpenAI (gpt-3.5-turbo)",
    "gemini": 'Gemini (gemini-1.5-pro)',
    "cohere": 'Cohere (command-r-plus)',
    "ollamma": "Ollama (llama3)",
    "groq": 'Groq (llama3-70b-8192)',
    "fireworks": "Fireworks (llama-v3-70b-instruct)"
}

embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

rerank = CohereRerank(
    model="rerank-english-v3.0",
    top_n=TOP_K
)

default_model = llm_map["cohere"],
