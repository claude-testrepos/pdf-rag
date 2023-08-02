# Chat with your PDFs

Upload course PDFs (slides, papers, textbooks) and ask questions about them. Classic retrieval-augmented generation: chunk → embed → store in Chroma → retrieve → answer.

## Run

```
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

## CLI ingest

`python ingest.py path/to/pdfs`

## Free embeddings

`EMBEDDINGS=local` uses all-MiniLM-L6-v2 locally. Delete `chroma_db/` and re-ingest after switching.
