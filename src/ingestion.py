import os

from langchain_community.document_loaders import (PyMuPDFLoader, TextLoader,
                                                  UnstructuredExcelLoader,
                                                  UnstructuredMarkdownLoader)
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

import src.config as cfg


class Ingestion:
    def __init__(self, embeddings):
        self._text_vectorstore = None
        self._embeddings = embeddings

    def create_and_add_embeddings(self, folder_path: str, **text_splitter_kwargs):
        documents = []

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if file.endswith('.pdf'):
                loader = PyMuPDFLoader(file_path)
            elif file.endswith('.md'):
                loader = UnstructuredMarkdownLoader(file_path)
            elif file.endswith('.xlsx'):
                loader = UnstructuredExcelLoader(file_path)
            elif file.endswith('.txt'):
                loader = TextLoader(file_path, 'utf8')

            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.PDF_CHAR_SPLITTER_CHUNK_SIZE,
            chunk_overlap=cfg.PDF_CHAR_SPLITTER_CHUNK_OVERLAP,
            **text_splitter_kwargs
        )

        chunked_documents = text_splitter.split_documents(documents)

        self._text_vectorstore = MongoDBAtlasVectorSearch.from_documents(
            documents=chunked_documents,
            embedding=self._embeddings,
            collection=cfg.COLLECTION,
            index_name=cfg.ATLAS_VECTOR_SEARCH_INDEX_NAME
        )
