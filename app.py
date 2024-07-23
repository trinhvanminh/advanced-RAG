from langchain_cohere import ChatCohere
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag import RAG

load_dotenv()

MONGODB_CONNECTION_STRING = os.getenv('MONGODB_CONNECTION_STRING')
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME') + '.full.2048.128__v4'
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')

# default llm
llm = ChatCohere(model="command-r", temperature=0)

embedding = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

rag = RAG(
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    mongodb_uri=MONGODB_CONNECTION_STRING,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    llm=llm,
    embedding=embedding
)


# history_collection = rag.collection('history')
# results = history_collection.distinct("SessionId")

# print(results)

# print(rag.retriever.invoke("What is the maximum loan term I can get?"))

# rag.collection.drop()
# count = rag.collection().count_documents({})

# if count == 0:
#     rag.load_documents(folder_path="./data/small")

# session_id = "zzz"
# config = {"configurable": {"session_id": session_id}}

# chain = rag.conversational_rag_chain

# response = chain.invoke(
#     {"input": "What is the maximum loan term I can get?"}, config=config
# )

# print(response['answer'])

# response = chain.invoke(
#     {"input": "Am I eligible for a construction loan?"}, config=config
# )

# response['context']

# print(response['answer'])

# session_history = rag.get_session_history(session_id=session_id)
# session_history.clear()
