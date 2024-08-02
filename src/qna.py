import time

from langchain.chains.combine_documents.stuff import \
    create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores.azure_cosmos_db import \
    AzureCosmosDBVectorSearch
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pymongo import MongoClient

import src.config as cfg
import src.prompts as prompts


class QnA:
    def __init__(
        self,
        model,
        rerank,
        vector_store: VectorStore | None = None
    ):

        self.model = model
        self.rerank = rerank

        # init vector store
        if vector_store is None:
            self.vector_store: AzureCosmosDBVectorSearch = cfg.vector_stores['azure-cosmos']
        else:
            self.vector_store = vector_store

        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.rerank,
            base_retriever=self.vector_store.as_retriever(
                search_type="similarity",
                # k=cfg.TOP_K,  # for Azure
                search_kwargs={
                    "fetch_k": 20,
                    # "k": cfg.TOP_K,
                    # "score_threshold": 0.5,
                },
            ),
        )

    @property
    def csv_retriever(self):
        from src.csv_store import CSVStore

        directory_path = './data/preprocessed/csv'
        csv_store = CSVStore(
            llm=self.model,
            directory_path=directory_path
        )

        return csv_store.get_retriever()

    def get_collection(self, collection_name: str = cfg.COLLECTION_NAME):
        client = MongoClient(cfg.CONNECTION_STRING)
        collection = client[cfg.DATABASE_NAME][collection_name]

        return collection

    def get_session_history(self, session_id: str) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=cfg.CONNECTION_STRING,
            database_name=cfg.DATABASE_NAME,
            collection_name=cfg.HISTORY_COLLECTION_NAME or 'message_store',
        )

    def ask_question(self, query: str, session_id: str):
        start_time = time.time()

        # TODO: dynamic switch using [RunnableBranch](https://python.langchain.com/v0.1/docs/expression_language/how_to/routing/#using-a-runnablebranch)
        history_aware_retriever = create_history_aware_retriever(
            self.model,
            self.retriever,
            # self.csv_retriever,
            prompts.contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            self.model,
            prompts.qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = conversational_rag_chain.invoke(
            input={"input": query},
            config={
                "configurable": {
                    "session_id": session_id,
                }
            },
        )

        exec_time = time.time() - start_time

        print("exec_time", exec_time)

        return response
