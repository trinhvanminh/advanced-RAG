
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores.azure_cosmos_db import \
    AzureCosmosDBVectorSearch
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI

import src.config as cfg
from src.csv_store import CSVStore

load_dotenv()


def get_docs_retriever():
    embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv(
            'AZURE_EMBEDDING_DEPLOYMENT_NAME'
        ),
        openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    )

    vector_store: AzureCosmosDBVectorSearch = AzureCosmosDBVectorSearch(
        collection=cfg.collection,
        embedding=embedding,
        index_name=cfg.VECTOR_SEARCH_INDEX_NAME,
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=cfg.rerank,
        base_retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                        "fetch_k": 20,
            },
        ),
    )

    return retriever


def get_csv_retriever():
    llm = AzureChatOpenAI(
        azure_deployment="mortgage-gpt-4",
        api_version="2024-05-01-preview",
        temperature=cfg.TEMPERATURE
    )
    csv_store = CSVStore(llm=llm, directory_path='./data/preprocessed/csv/')
    csv_retriever = csv_store.get_retriever()

    return csv_retriever
