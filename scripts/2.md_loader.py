from pymongo import MongoClient
from langchain_google_genai import (GoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

folder_path = './data/'
documents = []
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file.endswith('.md'):
        loader = UnstructuredMarkdownLoader(file_path)
        loaded_document = loader.load()
        documents.extend(loaded_document)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048, chunk_overlap=128)
chunked_documents = text_splitter.split_documents(documents)
print(len(chunked_documents))


MONGODB_CONNECTION_STRING = os.getenv('MONGODB_CONNECTION_STRING')
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME') + '_v2'
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')


llm = GoogleGenerativeAI(model='gemini-1.5-pro',
                         temperature=0, top_k=64, top_p=0.95)

embedding = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents=chunked_documents,
    embedding=embedding,
    collection=MongoClient(MONGODB_CONNECTION_STRING)[
        DB_NAME][COLLECTION_NAME],
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)
