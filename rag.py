import os

from langchain.chains.combine_documents.stuff import \
    create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import (PyMuPDFLoader, TextLoader,
                                                  UnstructuredExcelLoader,
                                                  UnstructuredMarkdownLoader)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Answer the question and provide additional helpful information, "
    "based on the pieces of information, if applicable. Be concise."
    "\n\n"
    "{context}"
    "\n\n"
    "IMPORTANT: Responses should be properly formatted to be easily read. MARKDOWN list syntax is recommended for long answer"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


class RAG():
    def __init__(self,
                 mongodb_uri: str,
                 db_name: str,
                 collection_name: str,

                 llm,
                 embedding,

                 index_name: str = 'vector_index',
                 history_collection_name: str = 'message_store',
                 search_type: str = 'similarity',
                 search_kwargs: dict = None,
                 rerank: bool = True,
                 ):

        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.index_name = index_name

        self.embedding = embedding
        self.llm = llm
        self.history_collection_name = history_collection_name

        self.search_type = search_type
        self.search_kwargs = search_kwargs or {"k": 4}
        self.rerank = rerank

    @property
    def client(self):
        return MongoClient(self.mongodb_uri)

    def collection(self, collection_name: str = ''):
        if collection_name == '':
            collection_name = self.collection_name

        return self.client[self.db_name][collection_name]

    @property
    def vector_store(self) -> MongoDBAtlasVectorSearch:
        return MongoDBAtlasVectorSearch(collection=self.collection(), embedding=self.embedding, index_name=self.index_name)

    @property
    def retriever(self):
        retriever = self.vector_store.as_retriever(
            search_type=self.search_type, search_kwargs=self.search_kwargs)

        if self.rerank:
            compressor = CohereRerank(
                model="rerank-english-v3.0", top_n=self.search_kwargs["k"])

            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )

        return retriever

    @property
    def rag_chain(self):
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt)

        question_answer_chain = create_stuff_documents_chain(
            self.llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    @property
    def conversational_rag_chain(self):
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain

    def get_session_history(self, session_id: str) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=self.mongodb_uri,
            database_name=self.db_name,
            collection_name=self.history_collection_name,
        )

    def load_documents(self, folder_path: str = "./data", text_splitter_kwargs: dict = {}) -> MongoDBAtlasVectorSearch:
        count = self.collection().count_documents({})
        if count != 0:
            self.collection().drop()

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

        # TODO: dynamic the chunk_size (recommended increase to 2048:128)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048, chunk_overlap=128, **text_splitter_kwargs)

        chunked_documents = text_splitter.split_documents(documents)

        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=chunked_documents,
            embedding=self.embedding,
            collection=self.collection,
            index_name=self.index_name
        )

        return vector_store
