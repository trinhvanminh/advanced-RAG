from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch, CosmosDBSimilarityType,
    CosmosDBVectorSearchType)

from src.ingestion import Ingestion

ingestion = Ingestion()

ingestion.preprocess_data()

vector_store = ingestion.create_and_add_embeddings()

if isinstance(vector_store, AzureCosmosDBVectorSearch):
    # Read more about these variables in detail here. https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search
    vector_store.create_index(
        num_lists=100,
        dimensions=1536,
        similarity=CosmosDBSimilarityType.COS,
        kind=CosmosDBVectorSearchType.VECTOR_IVF,
        m=16,
        ef_construction=64,
        # ef_search=40,
        # score_threshold=0.1,
    )
