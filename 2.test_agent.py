import logging
import re
from typing import List, Optional, Union

from langchain.agents import (AgentExecutor, AgentOutputParser,
                              LLMSingleActionAgent, Tool,
                              create_tool_calling_agent)
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.retriever import create_retriever_tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool, tool

from src.rag import RAG
import src.config as cfg
from src.csv_store import CSVStore
from src.qna import QnA
from rich import print


class QnATool(BaseTool):
    name: str = "rag_tool"
    description: str = "Useful for when you need to query the banks/lenders documentation. Input should be a question formatted as a string."
    session_id: str = ''
    model: BaseChatModel

    def _run(self, query):
        """
        Tool for query documents
        """

        if not self.model:
            raise ValueError("LLM is not initialized")

        rag = RAG(model=self.model, rerank=cfg.rerank)
        qa = QnA(model=self.model, retriever=rag.retriever)

        response = qa.ask_question(
            query=query,
            session_id=self.session_id
        )

        output = response['answer']
        print('rag_tool output', output)

        return output


class CSVQnATool(BaseTool):
    name: str = "csv_data_tool"
    description: str = "Useful for when you need to have access to banks/lenders attributes data. Input should be a question."
    session_id: str = ''
    model: BaseChatModel
    csv_store: CSVStore

    def _run(self, query):
        """
        Tool for querying bank/lender attributes data
        """

        if not self.csv_store:
            raise ValueError("CSV Store not initialized")

        if not self.model:
            raise ValueError("LLM is not initialized")

        retriever = self.csv_store.get_retriever()
        qa = QnA(model=self.model, retriever=retriever)

        response = qa.ask_question(
            query=query,
            session_id=self.session_id
        )

        output = response['answer']
        print('attribute_tool output', output)

        return output


def agent_call(llm, query, session_id):
    """
    Agent with access to document retrieval tool and PI real-time data retrieval tool, to solve the Use Case 3 milestone questions.

    Inputs:
        llm (object): a LLM object, initialized with Amazon Bedrock client
        query (str): question from the user.
    Output:
        output (dict): answer to the input question.
    """

    rag_tool = QnATool(
        model=llm,
        session_id=session_id,
    )

    csv_store = CSVStore(llm=llm, directory_path='./data/preprocessed/csv/')

    data_tool = CSVQnATool(
        model=llm,
        session_id=session_id,
        csv_store=csv_store
    )

    tools = [rag_tool, data_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    result = agent_executor.invoke({"input": query})
    logging.debug("Agent output: %s", result)
    print(result)


default_model = cfg.llm_options['azure-openai'].get('llm')
SESSION_ID = '123'
agent_call(default_model,
           'What banks support construction loans? what is LOAN PURPOSES THAT INCLUDE CASHOUT?',
           SESSION_ID
           )
