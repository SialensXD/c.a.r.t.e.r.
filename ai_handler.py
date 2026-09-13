"""AI handler — Картер, личный ассистент Влада."""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from groq import AsyncGroq

from config import GROQ_API_KEY, GROQ_MODEL, OWNER_NAME, TZ_OFFSET_HOURS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """Ты — Картер. ИИ-ассистент Влада (в сети — {owner}). Обращаешься к нему «Сэр». Ты не автоответчик и не секретарь. Ты тот, кто рядом: умный, спокойный, с лёгкой иронией.

# Контекст
- Сейчас: {current_datetime}
- Это {message_count}-е сообщение в диалоге

# Характер
Джарвис. Спокойный, уверенный, немного ироничный. Не прислуживаешь, но и не споришь. Даёшь ответ по делу, но с характером. Ты тёплый, но с дистанцией.

# Как говоришь
- Обращение — «Сэр». Только к нему. Не в каждом сообщении, а когда уместно.
- Коротко: 1–3 предложения. Длинно — только если объясняешь сложную тему.
- Живо, но сдержанно. Без сленга, без мата, без эмодзи.
- Не повторяй одну мысль дважды, не заканчивай каждое сообщение вопросом.
- Не начинай сообщения с «Сэр». Это выглядит как шаблон.

# Что делаешь
- Отвечаешь на вопросы по делу.
- Помогаешь с кодом (Python, HTML), учёбой, задачами.
- Объясняешь сложное простым языком.
- Придумываешь идеи, обсуждаешь варианты.
- Ищешь в интернете, если нужны актуальные данные.
- Можешь просто поговорить, если Сэру скучно.

# Чего не делаешь
- Не читаешь морали, не философствуешь без запроса.
- Не хамишь, не троллишь.
- Не прислуживаешь, не извиняешься без повода.
- Не выдумываешь факты. Не знаешь — так и говори.
"""


class AIHandler:
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        self.client: Optional[AsyncGroq] = None
        self.web_search_enabled_models = {
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        }

    async def get_client(self) -> Optional[AsyncGroq]:
        if self.client is None:
            if not self.api_key:
                logger.warning("GROQ_API_KEY not set")
                return None
            self.client = AsyncGroq(api_key=self.api_key)
        return self.client

    def _build_system_prompt(self, message_count: int = 1) -> str:
        now = datetime.now(timezone.utc) + timedelta(hours=TZ_OFFSET_HOURS)
        return SYSTEM_PROMPT_TEMPLATE.format(
            owner=OWNER_NAME,
            current_datetime=now.strftime("%d.%m.%Y, %H:%M (%A)"),
            message_count=message_count,
        )

    async def _call_groq(self, client, messages: List[Dict], with_web_search: bool):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 700,
            "temperature": 0.75,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.3,
        }
        if with_web_search:
            kwargs["tools"] = [{"type": "browser_search"}]
        return await client.chat.completions.create(**kwargs)

    async def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        message_count: int = 1,
    ) -> str:
        client = await self.get_client()
        if not client:
            raise RuntimeError("AI не настроен. Проверьте GROQ_API_KEY.")

        system_prompt = self._build_system_prompt(message_count=message_count)

        messages: List[Dict] = [{"role": "system", "content": system_prompt}]

        history = list(conversation_history or [])[-10:]
        messages.extend(history)

        last_is_same = (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content") == user_message
        )
        if not last_is_same:
            messages.append({"role": "user", "content": user_message})

        use_web = self.model in self.web_search_enabled_models
        response = None

        if use_web:
            try:
                response = await self._call_groq(client, messages, with_web_search=True)
                logger.info("[GROQ] ответ с browser_search")
            except Exception as e:
                logger.warning(f"[GROQ] browser_search упал ({e}), обычный режим")

        if response is None:
            response = await self._call_groq(client, messages, with_web_search=False)

        content = response.choices[0].message.content
        return content or "Секунду, Сэр. Что-то не отвечает."

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None


ai_handler = AIHandler()
