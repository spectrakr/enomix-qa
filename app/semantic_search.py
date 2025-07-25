from typing import List, Dict, Optional
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document


class SearchInput(BaseModel):
    question: str
    top_k: Optional[int] = 3
    fetch_k: Optional[int] = 15


class SearchResult(BaseModel):
    source: str
    chunk_index: Optional[int]
    page: Optional[int]
    content: str


def get_retrieved_chunks_before_llm(
    question: str,
    chroma_dir: str = "./chroma_db",
    top_k: int = 3,
    fetch_k: int = 15
) -> List[Dict]:
    vectordb = Chroma(persist_directory=chroma_dir, embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": fetch_k}
    )
    docs: List[Document] = retriever.get_relevant_documents(question)

    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content
        }
        for doc in docs
    ]