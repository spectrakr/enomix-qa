# 문서 검색 및 QA 시스템

이 프로젝트는 FastAPI를 기반으로 한 문서 검색 및 질의응답(QA) 시스템입니다. PDF 문서와 텍스트 파일을 처리하고, 사용자의 질문에 대해 의미론적 검색을 통해 답변을 제공합니다.

## 주요 기능

- PDF 및 텍스트 파일 업로드 및 색인
- 의미론적 검색 기반 질의응답
- Slack 통합 지원
- URL 기반 문서 처리
- 문서 관리 (업로드, 삭제, 재색인)

## 기술 스택

- FastAPI 0.110.0
- Uvicorn 0.29.0
- LangChain 0.1.13
- OpenAI API
- ChromaDB (벡터 데이터베이스)
- Slack SDK

## 설치 방법

1. Python 3.10  버전이 필요합니다.
2. python에 필요한 패키지를 설치합니다:
   ```bash
   pip3 install fastapi requests beautifulsoup4 slack_sdk python-dotenv langchain openai pypdf chromadb langchain-openai langchain-community pymupdf python-multipart
   ```

## 환경 설정

1. `.env` 파일을 생성하고 다음 환경 변수를 설정합니다:
   ```
   OPENAI_API_KEY=your_api_key
   SLACK_BOT_TOKEN=your_slack_bot_token
   SLACK_SIGNING_SECRET=your_slack_signing_secret
   ```

## 실행 방법

1. 서버를 시작합니다:
   ```bash
   uvicorn app.api:app --reload --host 0.0.0.0 --port 7070
   ```
2. 웹 브라우저에서 `http://localhost:8000/admin`에 접속하여 관리자 인터페이스를 사용할 수 있습니다.

## API 엔드포인트

- `GET /admin`: 관리자 인터페이스
- `POST /upload_pdf/`: PDF 파일 업로드
- `POST /upload_text/`: 텍스트 파일 업로드
- `POST /ask/`: 질문하기
- `POST /slack/events`: Slack 이벤트 처리
- `GET /indexed_files/`: 색인된 파일 목록 조회
- `DELETE /files/{filename}`: 파일 삭제
- `POST /indexed_files`: 모든 파일 재색인
- `POST /index_url/`: URL 기반 문서 처리

## 프로젝트 구조

```
.
├── app/
│   ├── api.py           # FastAPI 엔드포인트
│   ├── database.py      # 벡터 데이터베이스 관리
│   ├── semantic_search.py # 의미론적 검색 구현
│   ├── slack.py         # Slack 통합
│   ├── summary.py       # 문서 요약
│   ├── utils.py         # 유틸리티 함수
│   ├── qa_utils.py      # QA 관련 유틸리티
│   └── config.py        # 설정
├── chroma_db/           # 벡터 데이터베이스 저장소
├── static/             # 정적 파일
├── logs/               # 로그 파일
├── docs/               # 문서 저장소
└── requirements.txt    # 의존성 패키지
```

## 주의사항

- OpenAI API 키가 필요합니다.
- Slack 통합을 사용하려면 Slack Bot 토큰과 Signing Secret이 필요합니다.
- 대용량 문서 처리 시 메모리 사용량에 주의하세요.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 
