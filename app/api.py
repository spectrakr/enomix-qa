from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import requests

import os
import re
import shutil
import logging


from .config import STATIC_DIR, DOCS_DIR, LOG_FILE, CHROMA_DIR
from .database import (
    get_vectordb,
    index_document,
    get_indexed_files,
    remove_document,
    reset_vectordb
)
from .semantic_search import (
    SearchInput,
    SearchResult,
    get_retrieved_chunks_before_llm
)
from .slack import clean_mention, post_slack_reply
from .summary import summarize_slack_thread
from .slack import handle_slack_message
from .utils import format_sources

from langchain.chains import RetrievalQA
from .qa_utils import run_qa_chain
from slack_sdk.web.async_client import AsyncWebClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/admin")
async def root():
    return FileResponse(f"{STATIC_DIR}/index.html")

@app.get("/indexing")
async def root():
    return FileResponse(f"{STATIC_DIR}/text-indexing.html")

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🚀 Server starting up...")
        indexed_count = 0
        skipped_count = 0

        for filename in os.listdir(DOCS_DIR):
            if filename.endswith((".pdf", ".txt")):
                file_path = os.path.join(DOCS_DIR, filename)
                file_type = "pdf" if filename.endswith(".pdf") else "txt"

                # index_document의 결과로 성공 여부 반환
                result = index_document(file_path, file_type)

                if result is True:
                    indexed_count += 1
                elif result == "skipped":
                    skipped_count += 1
                else:
                    logger.warning(f"❓ Unknown result from indexing '{filename}': {result}")

        logger.info(f"📚 Startup indexing complete: {indexed_count} files indexed, {skipped_count} files unchanged")

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Error during startup: {str(e)}")

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(DOCS_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        if index_document(file_path, "pdf"):
            return {"message": f"'{file.filename}' indexed successfully"}
        else:
            return JSONResponse(status_code=500, content={"error": "Failed to index document"})
            
    except Exception as e:
        logger.error(f"❌ Error uploading PDF: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

def extract_questions(text: str) -> set:
    """Q1:, Q2: 형식으로 시작하는 질문 추출"""
    return set(
        m.strip()
        for m in re.findall(r"Q\d+:\s*(.+?)(?:\n|$)", text)
    )


@app.post("/upload_text/")
async def upload_text(text: str = Form(...), source: str = Form(...)):
    try:
        # 1. 입력값 검증
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "텍스트 내용이 비어있습니다."})

        safe_source = re.sub(r'[\\/]', '_', source.strip()) or "TEMP"
        txt_filename = f"{safe_source}.txt"
        txt_path = os.path.join(DOCS_DIR, txt_filename)

        logger.info(f"📝 텍스트 업로드 요청 - source: {safe_source}")

        # 2. 중복 질문 필터링
        new_questions = extract_questions(text.strip())
        existing_text = ""
        existing_questions = set()

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                existing_text = f.read()
                existing_questions = extract_questions(existing_text)

        duplicated = new_questions & existing_questions
        if duplicated:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "일부 또는 전체 질문이 이미 존재합니다.",
                    "duplicated_questions": list(duplicated)
                }
            )

        # 3. 텍스트 파일에 append 저장
        with open(txt_path, "a", encoding="utf-8") as f:
            if existing_text:
                f.write("\n")
            f.write(text.strip())

        # 4. 색인 처리 (database.py의 index_document 호출)
        if index_document(txt_path, file_type="txt", force=True):
            logger.info(f"✅ '{safe_source}' 텍스트 색인 완료")
            return {
                "message": f"'{safe_source}' 텍스트가 성공적으로 추가되고 재색인되었습니다.",
            }
        else:
            return JSONResponse(status_code=500, content={"error": "문서 색인 실패"})

    except Exception as e:
        logger.error(f"❌ upload_text 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/indexed_files/")
async def get_indexed_files_endpoint():
    try:
        files = get_indexed_files()
        # Add download URLs for each file
        files_with_urls = [
            {
                "filename": filename,
                "download_url": f"/download/{filename}"
            }
            for filename in files
        ]
        return {"indexed_files": files_with_urls}
    except Exception as e:
        logger.error(f"❌ Error getting indexed files: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        # Validate file extension
        if not filename.endswith((".pdf", ".txt")):
            return JSONResponse(
                status_code=400,
                content={"error": "Only .pdf and .txt files are supported"}
            )
            
        file_path = os.path.join(DOCS_DIR, filename)
        
        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"File '{filename}' not found"}
            )
            
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
            
    except Exception as e:
        logger.error(f"❌ Error downloading file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"💬 Question received: {question}")
        result = run_qa_chain(question)

        if "error" in result:
            return JSONResponse(status_code=400, content={"error": result["error"]})
        
        sources_text = format_sources(result["sources"])

        return {
            "question": result["question"],
            "answer": result["answer"],
            # "sources": result["sources"],
            "formatted_response": f"{result['answer']}{sources_text}"
        }

    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/slack/events")
async def slack_event_listener(
    request: Request,
    background_tasks: BackgroundTasks,
    x_slack_retry_num: str = Header(default=None),
    x_slack_retry_reason: str = Header(default=None)
):
    try:
        data = await request.json()

        # 중복 전송 방지
        if x_slack_retry_num:
            return {"status": "ok"}

        # Slack URL 인증
        if data.get("type") == "url_verification":
            return {"challenge": data.get("challenge")}

        # 실제 이벤트 처리
        if data.get("type") == "event_callback":
            event = data.get("event", {})
            event_type = event.get("type")
            user = event.get("user", "알 수 없음")

            # 앱 멘션
            if event_type == "app_mention":
                channel = event.get("channel")
                thread_ts = event.get("thread_ts", event.get("ts"))
                text = clean_mention(event.get("text", ""))
                logger.info(f"📥 채널 수신된 메시지 {user}: {text}")
                background_tasks.add_task(handle_slack_message, text, channel, thread_ts, event.get("ts"))

            # DM 메시지
            elif event_type == "message" and event.get("channel_type") == "im" and not event.get("bot_id"):
                channel = event.get("channel")
                text = event.get("text", "")
                logger.info(f"📥 앱 메세지 탭 수신된 메시지 {user} : {text}")
                background_tasks.add_task(handle_slack_message, text, channel, None, event.get("ts"))

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"❌ Error handling Slack event: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})





@app.delete("/files/{filename}")
async def delete_file(filename: str):
    try:
        # Validate file extension
        if not filename.endswith((".pdf", ".txt")):
            return JSONResponse(
                status_code=400,
                content={"error": "Only .pdf and .txt files are supported"}
            )
            
        file_path = os.path.join(DOCS_DIR, filename)
        
        if remove_document(file_path):
            return {"message": f"File '{filename}' deleted successfully"}
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"File '{filename}' not found"}
            )
            
    except Exception as e:
        logger.error(f"❌ Error deleting file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/indexed_files")
async def reindex_all_files():
    try:
        logger.info("🔄 Starting complete reindexing process...")
        
        # First, reset the Chroma DB
        if not reset_vectordb():
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to reset database"}
            )
        
        logger.info("🗑️ Database reset complete, starting reindexing...")
        indexed_count = 0
        error_count = 0
        
        # Get list of all files
        for filename in os.listdir(DOCS_DIR):
            if filename.endswith((".pdf", ".txt")):
                file_path = os.path.join(DOCS_DIR, filename)
                file_type = "pdf" if filename.endswith(".pdf") else "txt"
                
                try:
                    # No need for force=True since DB is fresh
                    if index_document(file_path, file_type):
                        indexed_count += 1
                        logger.info(f"✅ Indexed: {filename}")
                    else:
                        error_count += 1
                        logger.error(f"❌ Failed to index: {filename}")
                except Exception as e:
                    error_count += 1
                    logger.error(f"❌ Error indexing {filename}: {str(e)}")
        
        message = f"전체 재색인 완료: {indexed_count}개 성공"
        if error_count > 0:
            message += f", {error_count}개 실패"
            
        logger.info(message)
        return {"message": message}
        
    except Exception as e:
        error_msg = f"❌ Error during reindexing: {str(e)}"
        logger.error(error_msg)
        return JSONResponse(status_code=500, content={"error": error_msg})

@app.post("/index_url/")
async def upload_url(url: str = Form(...), source: str = Form(default="web")):
    try:
        logger.info(f"🌐 URL 크롤링 요청: {url}")

        # ✅ 웹 페이지 요청 및 파싱
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return JSONResponse(status_code=400, content={"error": f"Failed to fetch URL: {response.status_code}"})

        soup = BeautifulSoup(response.text, "html.parser")

        # ✅ 주요 태그 위주로 텍스트 구조화
        lines = []

        # 제목 계열 먼저
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            lines.append(f"# {header.get_text(strip=True)}")

        # 단락
        for paragraph in soup.find_all('p'):
            lines.append(paragraph.get_text(strip=True))

        # 리스트
        for li in soup.find_all('li'):
            lines.append(f"- {li.get_text(strip=True)}")

        # 기타 텍스트 누락 방지용 (기본적 body에서 추가로 가져오기)
        body_text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
        lines.append(body_text)

        # 중복 제거 및 정리
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in clean_lines:
                clean_lines.append(line)

        text = "\n".join(clean_lines)

        # ✅ 파일 저장
        safe_source = re.sub(r'[\\/]', '_', source.strip()) or "web"
        txt_filename = f"{safe_source}.txt"
        txt_path = os.path.join(DOCS_DIR, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        # ✅ 색인 처리
        if index_document(txt_path, "txt", force=True):
            return {"message": f"'{url}' 크롤링 및 색인 성공", "source": txt_filename}
        else:
            return JSONResponse(status_code=500, content={"error": "문서 색인 실패"})

    except Exception as e:
        logger.error(f"❌ upload_url 오류: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask_preview/", response_model=List[SearchResult])
def ask_preview(req: SearchInput):
    return get_retrieved_chunks_before_llm(
        question=req.question,
        chroma_dir=CHROMA_DIR,
        top_k=req.top_k,
        fetch_k=req.fetch_k
    )