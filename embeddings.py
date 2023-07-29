import os

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings


def get_embeddings():
    """EMBEDDINGS=local uses a free sentence-transformers model instead of OpenAI.

    Note: you have to re-ingest after switching, the vector sizes differ.
    """
    if os.environ.get("EMBEDDINGS") == "local":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return OpenAIEmbeddings()
