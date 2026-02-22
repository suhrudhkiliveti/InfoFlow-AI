# backend/rag/qa_pipeline.py
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from .vector_store import get_vector_db
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def get_rag_chain():
    db = get_vector_db()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
