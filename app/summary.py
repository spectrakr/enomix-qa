import os
import re
import logging
from openai import AsyncOpenAI
from slack_sdk.web.async_client import AsyncWebClient

logger = logging.getLogger(__name__)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"), ssl=False)

async def summarize_slack_thread(channel: str, thread_ts: str = None, additional_message: str = None) -> str:

    # ✅ 스레드에서 실행한 경우만 요약 허용
    if not thread_ts:
        return "⚠️ 요약은 스레드 안에서만 사용할 수 있습니다."

    resp = await client.conversations_replies(channel=channel, ts=thread_ts)

    # ✅ 봇 메시지 및 명령어 제외
    messages = [
        msg["text"]
        for msg in resp["messages"]
        if not msg.get("subtype")  # 메시지 subtype이 있는 경우 제외 (예: bot_message, message_changed 등)
        and not msg.get("bot_id")  # 봇이 보낸 메시지 제외
        and not re.sub(r"<@[\w]+>", "", msg.get("text", "").strip()).strip().startswith("/요약")  # 멘션 제거 후 /요약 여부 판단
    ]

    messages = messages[-50:]

    logger.info("📋 요약 대상 메시지 목록:")
    for i, m in enumerate(messages, start=1):
        logger.info(f"{i:02d}. {m}")

    if not messages:
        return "⚠️ 요약할 메시지가 없습니다."

       # 추가 메시지가 있는 경우 메시지 목록에 추가
    if additional_message:
        messages.append("핵심 내용과 논의 요점을 요약해 주세요.\n")
        messages.append(additional_message)

    prompt = (
        "\n".join(f"- {m}" for m in messages)
    )

    logger.info(f"📝 prompt: {prompt}")

    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.5
    )

    return completion.choices[0].message.content.strip()