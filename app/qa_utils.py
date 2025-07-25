from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from .database import get_vectordb

def run_qa_chain(question: str) -> dict:
    vectordb = get_vectordb()

    if vectordb._collection.count() == 0:
        return {"error": "색인된 문서가 없습니다."}

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 15}
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""다음은 문서에서 추출한 정보입니다:
---------------------
{context}
---------------------
질문: {question}

위 정보를 바탕으로 사용자에게 신뢰감 있게 답변해주세요.  
답변은 핵심을 명확하게 전달해야 하며, 가능한 한 문서 내용에 기반해야 합니다.

문서에 관련된 정보가 없을 경우, 다음과 같이 정중하게 응답해주세요:
"문서에 해당 정보가 없습니다."
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    result = qa_chain({"query": question})
    answer = result["result"].strip()
    source_docs = result.get("source_documents", [])

    # 참조 문서 정보 추출
    sources = []
    for doc in source_docs:
        metadata = doc.metadata
        source = metadata.get("source", "알 수 없음")
        page = metadata.get("page", "N/A")
        sources.append({
            "source": source,
            "page": page,
            "content": doc.page_content
        })

    return {
        "question": question,
        "answer": answer if answer else "문서에 해당 정보가 없습니다.",
        "sources": sources
    }