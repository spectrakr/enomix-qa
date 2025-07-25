from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os
import re
import logging
from datetime import datetime
from .config import CHROMA_DIR, DOCS_DIR

logger = logging.getLogger(__name__)

def get_vectordb():
    embedding = OpenAIEmbeddings()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

def get_file_metadata(file_path: str):
    """Get file metadata including last modification time"""
    return {
        "source": os.path.basename(file_path),
        "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
        "file_size": os.path.getsize(file_path)
    }

def is_file_modified(file_path: str, vectordb) -> bool:
    """Check if file needs to be reindexed by comparing modification times"""
    try:
        current_metadata = get_file_metadata(file_path)
        collection = vectordb.get()
        
        for metadata in collection["metadatas"]:
            if (isinstance(metadata, dict) and 
                metadata.get("source") == current_metadata["source"]):
                # If file exists in index, check if it's been modified
                if (metadata.get("last_modified") == current_metadata["last_modified"] and
                    metadata.get("file_size") == current_metadata["file_size"]):
                    return False
                return True
        
        # File not found in index
        return True
    except Exception as e:
        logger.error(f"Error checking file modification: {str(e)}")
        return True

def index_document(file_path: str, file_type: str = "pdf", force: bool = False):
    try:
        vectordb = get_vectordb()
        
        # Skip if file is already indexed and hasn't been modified
        if not force and not is_file_modified(file_path, vectordb):
            logger.info(f"📝 Skipping unchanged file: {file_path}")
            return True

        # Remove existing documents for this file if any
        collection = vectordb.get()
        docs_to_remove = []
        for i, metadata in enumerate(collection["metadatas"]):
            if isinstance(metadata, dict) and metadata.get("source") == os.path.basename(file_path):
                docs_to_remove.append(collection["ids"][i])
        
        if docs_to_remove:
            vectordb._collection.delete(docs_to_remove)
            logger.info(f"🗑️ Removed old version of: {file_path}")

        # Load and process the document
        if file_type == "pdf":
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
        else:  # txt
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            documents = [Document(page_content=content)]

        # Add metadata to all documents
        file_metadata = get_file_metadata(file_path)
        for doc in documents:
            doc.metadata.update(file_metadata)

        splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
        docs = splitter.split_documents(documents)

        # ✅ 전처리 함수 적용
        for idx, doc in enumerate(docs):
            doc.page_content = doc.page_content  # <-- 이 부분!
            doc.metadata["chunk_index"] = idx

        # Process documents in smaller batches
        BATCH_SIZE = 100
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i:i + BATCH_SIZE]
            vectordb.add_documents(batch)
            vectordb.persist()
            logger.info(f"✅ Processed batch {i//BATCH_SIZE + 1} of {(len(docs)-1)//BATCH_SIZE + 1}")
        
        logger.info(f"✅ Document indexed successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error indexing document: {str(e)}")
        return False

def get_indexed_files():
    try:
        vectordb = get_vectordb()
        collection = vectordb.get()
        sources = set()
        
        for metadata in collection["metadatas"]:
            if isinstance(metadata, dict) and "source" in metadata:
                sources.add(metadata["source"])
                
        return list(sources)
    except Exception as e:
        logger.error(f"❌ Error getting indexed files: {str(e)}")
        return []

def remove_document(file_path: str):
    """Remove document from Chroma DB and delete the file"""
    try:
        vectordb = get_vectordb()
        filename = os.path.basename(file_path)
        
        # Remove from Chroma DB
        collection = vectordb.get()
        docs_to_remove = []
        for i, metadata in enumerate(collection["metadatas"]):
            if isinstance(metadata, dict) and metadata.get("source") == filename:
                docs_to_remove.append(collection["ids"][i])
        
        if docs_to_remove:
            vectordb._collection.delete(docs_to_remove)
            vectordb.persist()
            logger.info(f"🗑️ Removed document from Chroma DB: {filename}")
        
        # Delete the file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🗑️ Deleted file: {file_path}")
            return True
        else:
            logger.warning(f"⚠️ File not found: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error removing document: {str(e)}")
        return False

def reset_vectordb():
    """
    ✅ Chroma DB의 모든 문서를 안전하게 제거합니다.
    ✅ embedding 호출 없이, 단순히 저장된 문서 ID 기준으로 삭제합니다.
    """
    try:
        vectordb = get_vectordb()
        collection = vectordb.get()

        all_ids = collection.get("ids", [])
        if all_ids:
            BATCH_SIZE = 100  # 안전을 위해 삭제도 batch 처리 가능
            for i in range(0, len(all_ids), BATCH_SIZE):
                batch_ids = all_ids[i:i + BATCH_SIZE]
                vectordb._collection.delete(batch_ids)
            vectordb.persist()
            logger.info(f"✅ Successfully reset Chroma DB - {len(all_ids)}개 문서 삭제 완료")
        else:
            logger.info("ℹ️ Chroma DB에 삭제할 문서가 없습니다.")
        return True

    except Exception as e:
        logger.error(f"❌ Error resetting Chroma DB: {str(e)}")
        return False

def clean_chunk_text(text: str) -> str:
    text = re.sub(r"[^\w\s가-힣.,:;!?()\\[\\]<>/@&%\"'\-]", "", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # ✅ 페이지 번호 제거 (예: "Page 1", "1 / 27", "15페이지" 등)
    text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\d+페이지", "", text)
    return text.strip()