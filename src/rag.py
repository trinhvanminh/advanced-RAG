import os

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores.azure_cosmos_db import \
    AzureCosmosDBVectorSearch
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings

import src.config as cfg


class RAG:
    def __init__(
        self,
        model,
        rerank,
        vector_store: VectorStore | None = None
    ):
        self.rerank = rerank
        self.model = model

        # init vector store
        if vector_store is None:
            embedding = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv(
                    'AZURE_EMBEDDING_DEPLOYMENT_NAME'
                ),
                openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            )

            self.vector_store: AzureCosmosDBVectorSearch = AzureCosmosDBVectorSearch(
                collection=cfg.collection,
                embedding=embedding,
                index_name=cfg.VECTOR_SEARCH_INDEX_NAME,
            )

        else:
            self.vector_store = vector_store

        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.rerank,
            base_retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "fetch_k": 20,
                },
            ),
        )
