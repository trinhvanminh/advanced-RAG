import os
from rich import print
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)


getenv = st.secrets.get
# getenv = os.getenv

LLAMA_PARSE = getenv('LLAMA_PARSE')

PDF_CHAR_SPLITTER_CHUNK_SIZE = 2048
PDF_CHAR_SPLITTER_CHUNK_OVERLAP = 128

# mongodb vector store
CONNECTION_STRING = getenv('MONGODB_CONNECTION_STRING')
DATABASE_NAME = getenv('DB_NAME')
COLLECTION_NAME = (getenv('COLLECTION_NAME') +
                   f'.{PDF_CHAR_SPLITTER_CHUNK_SIZE}.{PDF_CHAR_SPLITTER_CHUNK_OVERLAP}')
VECTOR_SEARCH_INDEX_NAME = getenv('VECTOR_SEARCH_INDEX_NAME')
HISTORY_COLLECTION_NAME = "message_store"

# model config
TEMPERATURE = 0
TOP_K = 4

AZURE_EMBEDDING_DEPLOYMENT_NAME = getenv('AZURE_EMBEDDING_DEPLOYMENT_NAME')
AZURE_OPENAI_API_VERSION = getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_ENDPOINT = getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = getenv('AZURE_OPENAI_API_KEY')
