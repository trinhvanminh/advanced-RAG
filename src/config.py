from langchain_core.language_models.chat_models import BaseChatModel
from typing import TypedDict
from langchain_core.vectorstores import VectorStore
import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereRerank
from langchain_community.vectorstores.azure_cosmos_db import \
    AzureCosmosDBVectorSearch
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_fireworks import ChatFireworks
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from llama_parse import LlamaParse
from pymongo import MongoClient

load_dotenv(override=True)


pdf_parser = LlamaParse(
    api_key=os.getenv('LLAMA_PARSE'),
    result_type="markdown",
    # parsing_instruction=instruction,
    max_timeout=5000,
)

PDF_CHAR_SPLITTER_CHUNK_SIZE = 2048
PDF_CHAR_SPLITTER_CHUNK_OVERLAP = 128

# mongodb vector store
CONNECTION_STRING = os.getenv('MONGODB_CONNECTION_STRING')
DATABASE_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = (os.getenv('COLLECTION_NAME') +
                   f'.{PDF_CHAR_SPLITTER_CHUNK_SIZE}.{PDF_CHAR_SPLITTER_CHUNK_OVERLAP}')

VECTOR_SEARCH_INDEX_NAME = os.getenv('VECTOR_SEARCH_INDEX_NAME')

client = MongoClient(CONNECTION_STRING)
collection = client[DATABASE_NAME][COLLECTION_NAME]
HISTORY_COLLECTION_NAME = "message_store"

# model config
TEMPERATURE = 0
TOP_K = 4


embedding_models: dict[str, Embeddings] = {
    "google": GoogleGenerativeAIEmbeddings(model='models/text-embedding-004'),
    "azure": AzureOpenAIEmbeddings(
        azure_deployment=os.getenv(
            'AZURE_EMBEDDING_DEPLOYMENT_NAME'
        ),
        openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    )
}


class LLMOption(TypedDict):
    label: str
    llm: BaseChatModel


llm_options: dict[str, LLMOption] = {
    "azure-openai": {
        "label": "Azure OpenAI (mortgage-gpt-4o)",
        "llm": AzureChatOpenAI(azure_deployment="mortgage-gpt-4", api_version="2024-05-01-preview", temperature=TEMPERATURE),
    },
    "chatgpt": {
        "label": "OpenAI (gpt-4o)",
        "llm": ChatOpenAI(model="gpt-3.5-turbo", temperature=TEMPERATURE),
    },
    "gemini": {
        "label": 'Gemini (gemini-1.5-pro)',
        "llm": ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=TEMPERATURE),
    },
    "cohere": {
        "label": 'Cohere (command-r-plus)',
        "llm": ChatCohere(model="command-r-plus", temperature=TEMPERATURE),
    },
    "ollamma": {
        "label": "Ollama (llama3)",
        "llm": ChatOllama(model="llama3", temperature=TEMPERATURE),
    },
    "groq": {
        "label": 'Groq (llama3-70b-8192)',
        "llm": ChatGroq(model_name="llama3-70b-8192", temperature=TEMPERATURE),
    },
    "fireworks": {
        "label": "Fireworks (llama-v3-70b-instruct)",
        "llm": ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct", temperature=TEMPERATURE),
    },
}

vector_stores: dict[str, VectorStore] = {
    # "azure-search": AzureSearch(
    #     azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    #     azure_search_key=os.getenv('AZURE_SEARCH_KEY'),
    #     embedding_function=embedding_models["azure"].embed_query,
    #     index_name=os.getenv('VECTOR_SEARCH_INDEX_NAME'),
    # ),
    "mongo-atlas": MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_models["google"],
        index_name=VECTOR_SEARCH_INDEX_NAME,
    ),
    "azure-cosmos": AzureCosmosDBVectorSearch(
        collection=collection,
        embedding=embedding_models["azure"],
        index_name=VECTOR_SEARCH_INDEX_NAME,
    )
}

rerank = CohereRerank(
    model="rerank-english-v3.0",
    top_n=TOP_K
)
