import requests
import logging
from .config import SLACK_BOT_TOKEN
import re
from .qa_utils import run_qa_chain
from .utils import format_sources
from .summary import summarize_slack_thread

logger = logging.getLogger(__name__)

def clean_mention(text: str) -> str:
    return re.sub(r'<@[^>]+>', '', text).strip()

def post_slack_reply(channel: str, thread_ts: str, text: str):
    try:
        headers = {
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            "Content-Type": "application/json"
        }
        
        data = {
            "channel": channel,
            "thread_ts": thread_ts,
            "text": text
        }
        
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=data,
            verify=False  # SSL 검증 비활성화
        )
        
        if response.status_code == 200:
            logger.info("✅ Slack message sent successfully")
            return True
        else:
            logger.error(f"❌ Failed to send Slack message: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error sending Slack message: {str(e)}")
        return False 


async def handle_slack_message(text: str, channel: str, thread_ts: str, message_ts: str):
    try:
        clean_text = text.strip().lower()

        if clean_text.startswith("/요약"):
            if not thread_ts or thread_ts == message_ts:
                logger.warning("⛔ /요약은 스레드의 댓글에서만 실행 가능합니다.")
                post_slack_reply(channel, message_ts, "⚠️ '/요약'은 스레드 **댓글**에서만 사용할 수 있습니다.")
                return
            else:
                logger.info("📄 요약 명령어 수신: 스레드 요약 시작")
                # 추가 메시지가 있는 경우 추출
                additional_message = clean_text.replace("/요약", "").strip()
                logger.info(f" 추가 메시지: {additional_message}")
                summary = await summarize_slack_thread(channel, thread_ts, additional_message)
                post_slack_reply(channel, thread_ts, f"🧾 요약 결과:\n{summary}")
                return

        # 일반 질문 처리
        result = run_qa_chain(clean_text)

        if "error" in result:
            post_slack_reply(channel, thread_ts, "📂 색인된 문서가 없습니다. 먼저 문서를 업로드해주세요.")
            return

        answer = result.get("answer", "답변을 생성하지 못했습니다.")
        sources = result.get("sources", [])
        sources_text = format_sources(sources)
        final_message = f"{answer}{sources_text}"

        post_slack_reply(channel, thread_ts, final_message)

    except Exception as e:
        logger.error(f"❌ Error handling Slack message: {str(e)}")
        post_slack_reply(channel, thread_ts, "❌ 오류가 발생했습니다. 다시 시도해주세요.")