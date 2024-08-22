import os
from typing import Any, Dict, List, TypedDict

import pandas as pd
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever, RetrieverOutput
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from azure.storage.blob import BlobServiceClient

import src.prompts as prompts


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


class CSVRetriever(BaseRetriever):
    llm: BaseChatModel
    directory_path: str
    connection_string: str = ''

    @property
    def storage_options(self) -> dict | None:
        storage_options = {
            "connection_string": self.connection_string
        } if self.connection_string else None

        return storage_options

    def file_path(self, file_name):
        prefix = 'abfs://' if self.connection_string else ''

        return f'{prefix}{self.directory_path}/{file_name}'

    def get_filenames(self) -> list[str]:
        if self.connection_string:
            blob_service_client = BlobServiceClient.from_connection_string(
                conn_str=self.connection_string)

            container_client = blob_service_client.get_container_client(
                container=self.directory_path)

            blob_list = container_client.list_blobs()
            return [blob.name for blob in blob_list]

        return os.listdir(self.directory_path)

    def get_file_selection_chain(self, file_names: List[str]) -> RunnableSerializable[Dict, FileNames]:
        parser = PydanticOutputParser(pydantic_object=FileNames)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.file_selection_prompt),
                ("human", "{input}")
            ]
        ).partial(format_instructions=parser.get_format_instructions(), file_names=file_names)

        file_selection_chain = prompt | self.llm | parser
        return file_selection_chain

    def get_relevant_headers_prompt(self, file_names: FileNames) -> str:
        file_name_with_sample_rows_context = []
        for file_name in file_names.file_names:
            df = pd.read_csv(
                self.file_path(file_name),
                storage_options=self.storage_options
            )
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
                ("system", prompts.header_selection_prompt),
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
                df = pd.read_csv(
                    self.file_path(file_name),
                    storage_options=self.storage_options
                )

                data = (df[relevant_header.headers]
                        .drop_duplicates()
                        .to_markdown(index=False))

                document = Document(
                    page_content=data,
                    metadata={"source": file_name}
                )

                documents.append(document)

        return documents

    @property
    def retriever(self):
        file_names = self.get_filenames()

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

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

         Args:
             query: string to find relevant documents for

         Returns:
             Sequence of relevant documents
         """
        docs = self.retriever.invoke(
            query,
            config={
                "callbacks": run_manager.get_child()
            },
            **kwargs
        )

        return docs
