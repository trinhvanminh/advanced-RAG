import time
from typing import Generator, List, Optional, TypedDict, Union

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains.combine_documents.stuff import \
    create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableConfig

from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pymongo import MongoClient

import src.constants as c
import src.prompts as prompts


class QnAResponse(TypedDict):
    """Class for QnAResponse.

    Example:

        .. code-block:: python
            from langchain_core.messages import HumanMessage, AIMessage
            from langchain_core.documents import Document

            # chat_history = output_messages_key
            # answer = output_messages_key
            {
                "chat_history": [HumanMessage(content="Could you..."), AIMessage(content="Yeah. I Can"), ...]
                "context":  [
                                Document(
                                    metadata={
                                        "source": "https://example.com",
                                        "_id": null,
                                        "vectorContent": [...],
                                        "relevance_score": 0.9998969
                                    }
                                    page_content="Hello, world!"
                                ),
                                ....
                            ]
                "input": "Hello "
                "answer": "Hi there!"
            }
    """
    chat_history: List[BaseMessage]
    context: list[Document]
    answer: str
    input: str


class QnA:
    def __init__(
        self,
        model,
        retriever: RetrieverLike,
        data_retriever: Optional[RetrieverLike] = None
    ):

        self.model = model
        self.retriever = retriever

        if data_retriever is None:
            self.data_retriever = retriever
        else:
            self.data_retriever = data_retriever

    @staticmethod
    def get_collection(collection_name: str = c.COLLECTION_NAME):
        client = MongoClient(c.CONNECTION_STRING)
        collection = client[c.DATABASE_NAME][collection_name]

        return collection

    @staticmethod
    def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=c.CONNECTION_STRING,
            database_name=c.DATABASE_NAME,
            collection_name=c.HISTORY_COLLECTION_NAME or 'message_store',
        )

    def _retriever_router(self, data) -> RetrieverLike:
        standalone_input: str = data.get('input', '')
        print('standalone_input', standalone_input)

        question_intent: str = data.get('question_intent', '')
        print('question_intent', question_intent)

        question_intent = question_intent.lower()

        retriever = RunnableLambda(lambda _: [])

        if question_intent == 'docs':
            retriever = self.retriever
        elif question_intent == 'data':
            retriever = self.data_retriever
        elif question_intent == 'combination':
            docs_tool = create_retriever_tool(
                self.retriever,
                "docs_retriever",
                "Useful for when you need to query the banks/lenders documentation. Input should be a question formatted as a string.",
            )

            data_tool = create_retriever_tool(
                self.data_retriever,
                "data_retriever",
                "Useful for when you need to have access to banks/lenders attributes data. Input should be a question.",
            )

            tools = [docs_tool, data_tool]

            agent = create_tool_calling_agent(
                self.model,
                tools,
                prompts.tool_calling_agent_prompt
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )

            # TODO: move this agent out, cause it already included the answer,
            # no need to pass it as retriever
            def to_document(data):
                # print('to_document', data)
                return [Document(page_content=data['output'])]

            # the input is a standalone question after formulated
            return {"input": RunnablePassthrough()} | agent_executor | RunnableLambda(to_document)

        final_chain = (lambda x: x['input']) | retriever
        return final_chain

    def get_history_aware_retriever_based_on_question_intent(self, data):
        question_intent_chain = (
            prompts.question_intent_prompt
            | self.model
            | StrOutputParser()
        )

        # the input is a standalone question after formulated
        retriever = (
            {
                "input": RunnablePassthrough(),
                "question_intent": {"input": RunnablePassthrough()} | question_intent_chain,
            }
            | RunnableLambda(self._retriever_router)
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=self.model,
            retriever=retriever,
            prompt=prompts.contextualize_q_prompt
        )

        return history_aware_retriever

    def ask_question(self, query: str, config: RunnableConfig, stream: bool = True) -> Union[QnAResponse, Generator[QnAResponse, None, None]]:
        start_time = time.time()

        question_answer_chain = create_stuff_documents_chain(
            llm=self.model,
            prompt=prompts.qa_prompt
        )

        history_aware_retriever = RunnableLambda(
            self.get_history_aware_retriever_based_on_question_intent
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

        if stream:
            response: Generator[QnAResponse, None, None] = conversational_rag_chain.stream(
                input={"input": query},
                config=config,
            )
        else:
            response: QnAResponse = conversational_rag_chain.invoke(
                input={"input": query},
                config=config,
            )

        exec_time = time.time() - start_time

        print("exec_time", exec_time)

        return response
