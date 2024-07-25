import os
from src.ingestion import Ingestion
import src.config as cfg


ingestion = Ingestion(embeddings=cfg.embeddings)

ingestion.preprocess_data()

# ingestion.create_and_add_embeddings()
