import os
from typing import Dict, List, TypedDict

import pandas as pd
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import RetrieverOutput
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable


FILE_SELECTION_PROMPT = (
    "You have a list of CSV file names with information on various financial products and services. "
    "Select up to 2 file names in the list that would be most helpful to answer user input. "
    "DO NOT generate new file names, ONLY select on provided file names. "
    "Wrap the output between ```json and ```"
    "{format_instructions} "
    "\nList of CSV file names:"
    "\n"
    "{file_names}"
)

HEADER_SELECTION_PROMPT = (
    "You have a list of CSV file names and sample rows from each file. "
    "Based on the user input, identify the most helpful headers from the sample rows to answer the query. "
    "Provide a list of these headers or an empty list if no helpful headers are found."
    "Wrap the output between ```json and ```"
    "{format_instructions}"
    "\n"
    "{relevant_headers_prompt}"
)

FINAL_QUERY_PROMPT = (
    "Based on the context answer the user input",
    "{context}"
)


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


class InputAndRelevantHeadersPrompt(TypedDict):
    input: str
    relevant_headers_prompt: str


# TODO: make it inherrit from BaseRetriever and override these methods
# - stream
# - _get_relevant_documents
#   .venv\Lib\site-packages\langchain_core\vectorstores\base.py:1244
#   .venv/Lib/site-packages/langchain/retrievers/contextual_compression.py:29
class CSVStore:
    def __init__(self, llm: BaseChatModel, directory_path: str):
        self.llm = llm
        self.directory_path = directory_path

    def get_file_selection_chain(self, file_names: List[str]) -> RunnableSerializable[Dict, FileNames]:
        parser = PydanticOutputParser(pydantic_object=FileNames)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FILE_SELECTION_PROMPT),
                ("human", "{input}")
            ]
        ).partial(format_instructions=parser.get_format_instructions(), file_names=file_names)

        file_selection_chain = prompt | self.llm | parser
        return file_selection_chain

    def get_relevant_headers_prompt(self, file_names: FileNames) -> str:
        file_name_with_sample_rows_context = []
        for file_name in file_names.file_names:
            df = pd.read_csv(f'{self.directory_path}/{file_name}')
            sample_rows = df.head().to_markdown(index=False)

            file_context = f"""## File name: `{file_name}`\n### Sample rows:\n{sample_rows}"""

            file_name_with_sample_rows_context.append(file_context)

        combined_prompt = "\n\n===\n\n".join(
            file_name_with_sample_rows_context)

        return combined_prompt

    def get_header_selection_chain(self, _data: InputAndRelevantHeadersPrompt) -> RunnableSerializable[Dict, RelevantHeaders]:
        parser = PydanticOutputParser(pydantic_object=RelevantHeaders)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", HEADER_SELECTION_PROMPT),
                ("human", "{input}")
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        header_selection_chain = prompt | self.llm | parser

        return header_selection_chain

    def document_retriever(self, relevant_headers_response: RelevantHeaders) -> RetrieverOutput:
        documents: RetrieverOutput = []
        for relevant_header in relevant_headers_response.relevant_headers:
            if len(relevant_header.headers) > 0:
                file_name = relevant_header.file_name
                df = pd.read_csv(f'{self.directory_path}/{file_name}')

                data = (df[relevant_header.headers]
                        .drop_duplicates()
                        .to_markdown(index=False))

                document = Document(
                    page_content=data,
                    metadata={"source": file_name}
                )

                documents.append(document)

        return documents

    def as_retriever(self):
        file_names = os.listdir(self.directory_path)

        file_selection_chain = self.get_file_selection_chain(
            file_names=file_names
        )

        data: InputAndRelevantHeadersPrompt = {
            "input": RunnablePassthrough(),
            "relevant_headers_prompt": (
                file_selection_chain
                | RunnableLambda(self.get_relevant_headers_prompt)
            )
        }

        combined_header_selection_chain = (
            data
            | RunnableLambda(self.get_header_selection_chain)
        )

        document_retriever_chain = (
            combined_header_selection_chain
            | RunnableLambda(self.document_retriever)
        )

        return document_retriever_chain

# from rich import print
# import config as cfg
# llm = cfg.llm_options['azure-openai'].get('llm')

# retriever = CSVStore(
#     llm=llm, directory_path='./data/preprocessed/csv/').as_retriever()


# # print(retriever.invoke("bridging loan"))
# for chunk in retriever.stream("bridging_loans_data"):
#     print('chunk', chunk)
# #
