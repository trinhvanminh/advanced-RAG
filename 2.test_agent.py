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
from src.csv_retriever import CSVRetriever
from src.qna import QnA
from rich import print


def agent_call(llm, query):
    """
    Agent with access to document retrieval tool and PI real-time data retrieval tool, to solve the Use Case 3 milestone questions.

    Inputs:
        llm (object): a LLM object, initialized with Amazon Bedrock client
        query (str): question from the user.
    Output:
        output (dict): answer to the input question.
    """

    rag = RAG(model=default_model, rerank=cfg.rerank)

    csv_retriever = CSVRetriever(
        llm=default_model,
        directory_path='./data/preprocessed/csv/'
    )

    docs_tool = create_retriever_tool(
        rag.retriever,
        "docs_retriever",
        "Useful for when you need to query the banks/lenders documentation. Input should be a question formatted as a string.",
    )

    data_tool = create_retriever_tool(
        csv_retriever,
        "data_retriever",
        "Useful for when you need to have access to banks/lenders attributes data. Input should be a question.",
    )

    tools = [docs_tool, data_tool]

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
        handle_parsing_errors=True
    )

    result = agent_executor.invoke({"input": query})
    print(result)
    # logging.debug("Agent output: %s", result)
    # for chunk in result:
    #     print(chunk)


default_model = cfg.llm_options['azure-openai'].get('llm')

# Which banks offer construction loans? What are loan purposes that include cash-out?
agent_call(default_model,
           'What banks support construction loans? what is LOAN PURPOSES THAT INCLUDE CASHOUT?'
           )
