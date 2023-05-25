import sys

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
qa = RetrievalQA.from_chain_type(ChatOpenAI(temperature=0), retriever=db.as_retriever())
print(qa.run(" ".join(sys.argv[1:])))
