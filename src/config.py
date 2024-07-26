import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereRerank
from langchain_fireworks import ChatFireworks
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from llama_parse import LlamaParse
from pymongo import MongoClient

load_dotenv()

CONNECTION_STRING = os.getenv('MONGODB_CONNECTION_STRING')
DATABASE_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME') + '.full.2048.128__v4'
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')

PDF_CHAR_SPLITTER_CHUNK_SIZE = 2048
PDF_CHAR_SPLITTER_CHUNK_OVERLAP = 128
TEMPERATURE = 0
TOP_K = 4

client = MongoClient(CONNECTION_STRING)
collection = client[DATABASE_NAME][COLLECTION_NAME]
HISTORY_COLLECTION_NAME = "message_store"

pdf_parser = LlamaParse(
    api_key=os.getenv('LLAMA_PARSE'),
    result_type="markdown",
    # parsing_instruction=instruction,
    max_timeout=5000,
)

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


# ========================AZURE stuffs================================

AZURE_ENDPOINT: str = os.getenv('AZURE_ENDPOINT')
AZURE_OPENAI_API_KEY: str = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_VERSION: str = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_EMBEDDING_DEPLOYMENT_NAME: str = os.getenv(
    'AZURE_EMBEDDING_DEPLOYMENT_NAME'
)
AZURE_SEARCH_ENDPOINT: str = os.getenv('AZURE_SEARCH_ENDPOINT')
AZURE_SEARCH_KEY: str = os.getenv('AZURE_SEARCH_KEY')

azure_embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

# no need to precreate in Azure
AZURE_SEARCH_INDEX_NAME: str = os.getenv('AZURE_SEARCH_INDEX_NAME')
