from typing import TypedDict

from langchain_cohere import ChatCohere, CohereRerank
from langchain_community.vectorstores.azure_cosmos_db import \
    AzureCosmosDBVectorSearch
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_fireworks import ChatFireworks
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from llama_parse import LlamaParse
from pymongo import MongoClient

import src.constants as c

pdf_parser = LlamaParse(
    api_key=c.LLAMA_PARSE,
    result_type="markdown",
    # parsing_instruction=instruction,
    max_timeout=5000,
)

client = MongoClient(c.CONNECTION_STRING)
collection = client[c.DATABASE_NAME][c.COLLECTION_NAME]

embedding_models: dict[str, Embeddings] = {
    "google": GoogleGenerativeAIEmbeddings(model='models/text-embedding-004'),
    "azure": AzureOpenAIEmbeddings(
        azure_deployment=c.AZURE_EMBEDDING_DEPLOYMENT_NAME,
        openai_api_version=c.AZURE_OPENAI_API_VERSION,
        azure_endpoint=c.AZURE_OPENAI_ENDPOINT,
        api_key=c.AZURE_OPENAI_API_KEY,
    )
}


class LLMOption(TypedDict):
    label: str
    llm: BaseChatModel


llm_options: dict[str, LLMOption] = {
    "azure-openai": {
        "label": "Azure OpenAI (mortgage-gpt-4o)",
        "llm": AzureChatOpenAI(azure_deployment="mortgage-gpt-4", api_version="2024-05-01-preview", temperature=c.TEMPERATURE),
    },
    "chatgpt": {
        "label": "OpenAI (gpt-4o)",
        "llm": ChatOpenAI(model="gpt-3.5-turbo", temperature=c.TEMPERATURE),
        "disabled": True
    },
    "gemini": {
        "label": 'Gemini (gemini-1.5-pro)',
        "llm": ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=c.TEMPERATURE),
    },
    "cohere": {
        "label": 'Cohere (command-r-plus)',
        "llm": ChatCohere(model="command-r-plus", temperature=c.TEMPERATURE),
    },
    "ollamma": {
        "label": "Ollama (llama3)",
        "llm": ChatOllama(model="llama3", temperature=c.TEMPERATURE),
        "disabled": True
    },
    "groq": {
        "label": 'Groq (llama3-70b-8192)',
        "llm": ChatGroq(model_name="llama3-70b-8192", temperature=c.TEMPERATURE),
    },
    "fireworks": {
        "label": "Fireworks (llama-v3-70b-instruct)",
        "llm": ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct", temperature=c.TEMPERATURE),
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
        index_name=c.VECTOR_SEARCH_INDEX_NAME,
    ),
    "azure-cosmos": AzureCosmosDBVectorSearch(
        collection=collection,
        embedding=embedding_models["azure"],
        index_name=c.VECTOR_SEARCH_INDEX_NAME,
    )
}

rerank = CohereRerank(
    model="rerank-english-v3.0",
    top_n=c.TOP_K
)
