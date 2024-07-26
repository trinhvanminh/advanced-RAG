from langchain_community.vectorstores.azuresearch import AzureSearch

import src.config as cfg
from src.ingestion import Ingestion

# ingestion = Ingestion(embeddings=cfg.embeddings)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=cfg.AZURE_SEARCH_ENDPOINT,
    azure_search_key=cfg.AZURE_SEARCH_KEY,
    index_name=cfg.AZURE_SEARCH_INDEX_NAME,
    embedding_function=cfg.azure_embeddings.embed_query,
)

ingestion = Ingestion(embeddings=cfg.embeddings, vector_store=vector_store)

# ingestion.preprocess_data()

ingestion.create_and_add_embeddings()
