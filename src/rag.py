from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores.azure_cosmos_db import \
    AzureCosmosDBVectorSearch
from langchain_core.vectorstores import VectorStore

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
            self.vector_store: AzureCosmosDBVectorSearch = cfg.vector_stores['azure-cosmos']
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
