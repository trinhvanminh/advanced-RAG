import os
from pathlib import Path

from langchain_community.document_loaders import (PyMuPDFLoader, TextLoader,
                                                  UnstructuredExcelLoader,
                                                  UnstructuredMarkdownLoader)
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

import src.config as cfg


class Ingestion:
    def __init__(self, embeddings, raw_data_folder_path: str = './data/raw/', preprocessed_folder_path: str = './data/preprocessed/'):
        self._text_vectorstore = None
        self.embeddings = embeddings
        self.pdf_parser = cfg.pdf_parser
        self.raw_data_folder_path = raw_data_folder_path
        self.preprocessed_folder_path = preprocessed_folder_path

    def preprocess_data(self):

        for file in os.listdir(self.raw_data_folder_path):
            file_path = os.path.join(self.raw_data_folder_path, file)

            if file.endswith('.pdf'):
                llama_parse_documents = self.pdf_parser.load_data(file_path)

                parsed_doc = "\n\n".join(
                    [doc.text for doc in llama_parse_documents]
                )

                output_path = Path(os.path.join(
                    self.preprocessed_folder_path, file) + '.md'
                )

                with output_path.open("w", encoding="utf-8") as f:
                    f.write(parsed_doc)

    def create_and_add_embeddings(self, **text_splitter_kwargs):
        documents = []

        for file in os.listdir(self.preprocessed_folder_path):
            file_path = os.path.join(self.preprocessed_folder_path, file)

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
            embedding=self.embeddings,
            collection=cfg.collection,
            index_name=cfg.ATLAS_VECTOR_SEARCH_INDEX_NAME
        )
