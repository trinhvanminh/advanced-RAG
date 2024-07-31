from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
import json
import os
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from rich import print

from src.config import llm_options


class FileNames(BaseModel):
    """list of JSON file name"""
    file_names: List[str]


# Define the directory path
directory_path = './data/preprocessed/csv'

# Get all file names in the directory
file_names = [f for f in os.listdir(directory_path) if f.endswith('.csv')]


parser = PydanticOutputParser(pydantic_object=FileNames)

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', """You are provided with a list of JSON files containing information about various financial products and services.
        Which files would be most helpful to answer this question. Answer the user query. Wrap the output between ```json and ```\n{format_instructions}",
        List of JSON files:

        {file_names}

        """),
        ("human", "{input}")
    ]
).partial(format_instructions=parser.get_format_instructions())

llm = llm_options['azure-openai'].get('llm')

chain = prompt | llm | parser

# response: FileNames = chain.invoke({
#     "input": 'give me the best lender with lowest briding loans fees and lowest cashback amount',
#     "file_names": file_names
# })

# response.file_names
file_paths = [
    f"{directory_path}/{fn}" for fn in ['bridging_loans_data.csv']]

agent = create_csv_agent(
    llm,
    file_paths,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True,
    agent_executor_kwargs={
        "handle_parsing_errors": True
    }
)

res = agent.run(
    "give me the list of lenders investment data")

print(res)
