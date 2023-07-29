from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from embeddings import get_embeddings

PROMPT = PromptTemplate.from_template(
    """Answer the question using only the context below. If the answer is not in the
context, say you don't know. Keep it short.

Context:
{context}

Question: {question}
Answer:"""
)


def build_chain(k=4, model="gpt-3.5-turbo"):
    db = Chroma(persist_directory="chroma_db", embedding_function=get_embeddings())
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name=model, temperature=0),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
