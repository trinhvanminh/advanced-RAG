from langchain_core.runnables import RunnablePassthrough
import os
from typing import Dict, List

import pandas as pd
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import RetrieverOutput
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from rich import print

from src.config import llm_options


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


class CSVStore:
    def __init__(self, llm: BaseChatModel, directory_path: str):
        self.llm = llm
        self.directory_path = directory_path

    def get_file_selection_chain(self, file_names: List[str]) -> RunnableSerializable[Dict, FileNames]:
        parser = PydanticOutputParser(pydantic_object=FileNames)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ("You have a list of CSV file names with information on various financial products and services. "
                            "Select up to 2 file names in the list that would be most helpful to answer user input. "
                            "DO NOT generate new file names, ONLY select on provided file names. "
                            "Wrap the output between ```json and ```"
                            "{format_instructions} "
                            "\nList of CSV file names:"
                            "\n"
                            "{file_names}")
                 ),
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
            file_name_with_sample_rows_context
        )

        return combined_prompt

    def get_header_selection_chain(self, _) -> RunnableSerializable[Dict, RelevantHeaders]:
        # second parameter is input and relevant_headers_prompt from the previous output

        parser = PydanticOutputParser(pydantic_object=RelevantHeaders)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ("You have a list of CSV file names (usually 2 file names) and sample rows from each file. "
                            "Each file name start with: ## File name: <file_name>. "
                            "Based on the user input, identify the most helpful headers from the sample rows to answer the query. "
                            "Provide a list of these headers or an empty list if no helpful headers are found."
                            "Wrap the output between ```json and ```"
                            "{format_instructions}"
                            "\n"
                            "\n"
                            "Here is the list of CSV file names and sample rows:\n"
                            "{relevant_headers_prompt}"
                            )
                 ),
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
                        # .reset_index(drop=True)
                        .to_markdown(index=False))

                document = Document(
                    page_content=data,
                    metadata={"source": file_name}
                )

                documents.append(document)

        return documents

    def get_retriever(self):
        file_names = os.listdir(self.directory_path)

        file_selection_chain = self.get_file_selection_chain(
            file_names=file_names
        )

        header_prompt = (
            file_selection_chain
            | RunnableLambda(self.get_relevant_headers_prompt)
        )

        combined_header_selection_chain = (
            {
                "input": RunnablePassthrough(),
                "relevant_headers_prompt": header_prompt
            }
            | RunnableLambda(self.get_header_selection_chain)
        )

        document_retriever_chain = (
            combined_header_selection_chain
            | RunnableLambda(self.document_retriever)
        )

        print(document_retriever_chain.invoke(
            "give me the best lender with lowest bridging loans fees and lowest cashback amount"))

        return document_retriever_chain

    # def main(self):

    #     document_retriever = self.get_retriever()

    #     final_query_prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ('system', """Based on the context answer the user input",

    #             {context}

    #             """),
    #             ("human", "{input}")
    #         ]
    #     )

    #     chain = (
    #         RunnablePassthrough.assign(
    #             context=document_retriever.with_config(
    #                 run_name="retrieve_documents")
    #         )
    #         | final_query_prompt
    #         | self.llm
    #     )

    #     print(chain.invoke({"input": question}).content)


if __name__ == "__main__":
    directory_path = './data/preprocessed/csv'
    llm = llm_options['azure-openai'].get('llm')
    question = "give me the best lender with lowest bridging loans fees and lowest cashback amount"

    csv_store = CSVStore(
        llm=llm,
        directory_path=directory_path
    )

    retriever = csv_store.get_retriever()

    # response = retriever.invoke({"input": question})
    # print(response)
