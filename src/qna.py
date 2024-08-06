import time

from langchain.chains.combine_documents.stuff import \
    create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pymongo import MongoClient

import src.config as cfg
import src.prompts as prompts


class QnA:
    def __init__(
        self,
        model,
        retriever: RetrieverLike
    ):

        self.model = model
        self.retriever = retriever

    @staticmethod
    def get_collection(collection_name: str = cfg.COLLECTION_NAME):
        client = MongoClient(cfg.CONNECTION_STRING)
        collection = client[cfg.DATABASE_NAME][collection_name]

        return collection

    @staticmethod
    def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=cfg.CONNECTION_STRING,
            database_name=cfg.DATABASE_NAME,
            collection_name=cfg.HISTORY_COLLECTION_NAME or 'message_store',
        )

    def ask_question(self, query: str, session_id: str):
        start_time = time.time()

        history_aware_retriever = create_history_aware_retriever(
            llm=self.model,
            retriever=self.retriever,
            prompt=prompts.contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            llm=self.model,
            prompt=prompts.qa_prompt
        )

        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain
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
