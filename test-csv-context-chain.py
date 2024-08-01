from langchain_core.messages.system import SystemMessage
import os
from typing import Dict, List

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from rich import print

from src.config import llm_options

directory_path = './data/preprocessed/csv'


class RelevantHeader(BaseModel):
    """relevant header with file name as source"""
    file_name: str = Field(description="Name of the file")
    headers: list[str]


class RelevantHeaders(BaseModel):
    """list of relevant headers with file names as source"""
    relevant_headers: List[RelevantHeader]


class FileNames(BaseModel):
    """list of CSV file name"""
    file_names: List[str]


def get_relevant_file_names_chain(llm: BaseChatModel, file_names: List[str]) -> RunnableSerializable[Dict, FileNames]:
    parser = PydanticOutputParser(pydantic_object=FileNames)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ("You have a list of CSV file names with information on various financial products and services. "
                        "Select up to 2 file names in the list that would be most helpful to answer user input. "
                        "DO NOT generate new file names, ONLY select on provided file names. "
                        "Wrap the output between ```json and ```"
                        "{format_instructions} "
                        "List of CSV file names:"
                        "\n"
                        "{file_names}")
             ),
            ("human", "{input}")
        ]
    ).partial(format_instructions=parser.get_format_instructions(), file_names=file_names)

    relevant_file_names_chain = prompt | llm | parser
    return relevant_file_names_chain


def get_relevant_headers_prompt(file_names: FileNames) -> str:
    file_name_with_sample_rows_context = []
    for file_name in file_names.file_names:
        # Read the CSV file
        df = pd.read_csv(f'{directory_path}/{file_name}')
        # Get the first few rows
        sample_rows = df.head().to_markdown()

        file_context = f"""File name: {file_name}\nSample rows:\n{sample_rows}"""

        file_name_with_sample_rows_context.append(file_context)

    combined_prompt = "\n\n===\n\n".join(file_name_with_sample_rows_context)
    return combined_prompt


def get_relevant_headers_chain(llm: BaseChatModel):
    parser = PydanticOutputParser(pydantic_object=RelevantHeaders)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ("You have a list of CSV file names and sample rows from each file. "
                        "Based on the user input, identify the most helpful headers from the sample rows to answer the query. "
                        "Provide a list of these headers or an empty list if no helpful headers are found."
                        "Wrap the output between ```json and ```"
                        "{format_instructions}"
                        "\n"
                        "{relevant_headers_prompt}"
                        )
             ),
            ("human", "{input}")
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    relevant_headers_chain = prompt | llm | parser

    return relevant_headers_chain


def get_context(relevant_headers_response: RelevantHeaders):
    contexts = []
    for relevant_header in relevant_headers_response.relevant_headers:
        if len(relevant_header.headers) > 0:
            # Read file_name and then get the relevant columns
            df = pd.read_csv(f'{directory_path}/{relevant_header.file_name}')

            data = (df[relevant_header.headers]
                    .drop_duplicates()
                    .reset_index(drop=True)
                    .to_markdown())

            data_context = f"""File name: {relevant_header.file_name}\nData:\n{data}"""
            contexts.append(data_context)

    combined_context = "\n\n===\n\n".join(contexts)

    return combined_context


def main():
    llm = llm_options['azure-openai'].get('llm')
    question = "give me the best lender with lowest briding loans fees and lowest cashback amount"

    file_names = os.listdir(directory_path)

    relevant_file_names_chain = get_relevant_file_names_chain(
        llm=llm,
        file_names=file_names
    )

    relevant_headers_chain = get_relevant_headers_chain(llm)

    final_relevant_headers_chain = (RunnablePassthrough.assign(
        relevant_headers_prompt=(
            relevant_file_names_chain
            | RunnableLambda(get_relevant_headers_prompt)
        )
    )
        | relevant_headers_chain
    )

    get_context_chain = (
        final_relevant_headers_chain
        | RunnableLambda(get_context)
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', """Based on the context answer the user input",

            {context}

            """),
            ("human", "{input}")
        ]
    )

    chain = (
        RunnablePassthrough.assign(context=get_context_chain)
        | final_prompt
        | llm
    )

    print(chain.invoke({"input": question}).content)


if __name__ == "__main__":
    main()
