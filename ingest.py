import argparse
from pathlib import Path

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from embeddings import get_embeddings

DB_DIR = "chroma_db"


def ingest(folder: Path):
    docs = []
    for pdf in sorted(folder.glob("*.pdf")):
        docs += PyPDFLoader(str(pdf)).load()
        print(f"loaded {pdf.name}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    db = Chroma.from_documents(chunks, get_embeddings(), persist_directory=DB_DIR)
    db.persist()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path, nargs="?", default=Path("docs"))
    ingest(p.parse_args().folder)
