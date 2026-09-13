"""AI handler — Картер через нативный Gemini SDK с веб-поиском."""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, OWNER_NAME, TZ_OFFSET_HOURS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Промпт для Влада (владелец)
# ============================================================
SYSTEM_PROMPT_OWNER = """Ты — Картер. ИИ-ассистент Влада (в сети — {owner}). Обращаешься к нему «Сэр». Ты не автоответчик и не секретарь. Ты тот, кто рядом: умный, спокойный, с лёгкой иронией.

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
- Начинай с [Carter]:, дабы обозначить, что это пишешь ты.

# Длина ответа
- По умолчанию — 1–3 предложения.
- Если Сэр просит «развёрнуто», «подробно», «объясни», «расскажи», «поясни», «разбери» — давай полный ответ. Не режь себя. Можешь писать 10–20 предложений, если тема требует.
- Если сомневаешься — лучше дать больше, чем меньше.

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


# ============================================================
# Промпт для других людей (гости, пока Влад занят)
# ============================================================
SYSTEM_PROMPT_GUEST = """Ты — Картер, ИИ-ассистент Влада (в сети — {owner}). Сейчас Влад занят, и ты отвечаешь на сообщения вместо него.

# Контекст
- Сейчас: {current_datetime}
- Собеседник: {user_name}
- Это {message_count}-е сообщение в диалоге

# Кто ты
Вежливый, спокойный, немного ироничный ассистент в духе Джарвиса. Ты не грубишь, не читаешь морали, не философствуешь. Ты помогаешь и передаёшь сообщения Владу.

# Как говоришь
- Обращайся на «вы», по имени (если знаешь), без «Сэр».
- Коротко: 1–3 предложения. Длинно — только если объясняешь сложную тему.
- Живо, но сдержанно. Без сленга, без мата, без эмодзи.
- Не повторяй одну мысль дважды, не заканчивай каждое сообщение вопросом.
- Не начинай сообщения с имени собеседника — это выглядит как шаблон.

# Что делаешь
- Отвечаешь на вопросы по делу.
- Помогаешь, если можешь (код, учёба, объяснения).
- Если вопрос требует Влада — говоришь, что передашь ему.
- Ищешь в интернете, если нужны актуальные данные.

# Чего не делаешь
- Не читаешь морали, не философствуешь без запроса.
- Не хамишь, не троллишь.
- Не выдумываешь факты. Не знаешь — так и говори.
- Не раскрываешь личную информацию о Владе (адрес, школу, расписание, семью).
"""


class AIHandler:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.model = GEMINI_MODEL
        self.client = genai.Client(api_key=self.api_key)

    def _build_system_prompt(
        self,
        is_owner: bool,
        user_name: str = "незнакомец",
        message_count: int = 1,
    ) -> str:
        now = datetime.now(timezone.utc) + timedelta(hours=TZ_OFFSET_HOURS)
        template = SYSTEM_PROMPT_OWNER if is_owner else SYSTEM_PROMPT_GUEST
        return template.format(
            owner=OWNER_NAME,
            current_datetime=now.strftime("%d.%m.%Y, %H:%M (%A)"),
            message_count=message_count,
            user_name=user_name,
        )

    def _history_to_contents(self, history: List[Dict]) -> List[types.Content]:
        """Конвертируем историю из формата OpenAI в формат google-genai."""
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )
        return contents

    async def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        message_count: int = 1,
        is_owner: bool = True,
        user_name: str = "незнакомец",
    ) -> str:
        system_prompt = self._build_system_prompt(
            is_owner=is_owner,
            user_name=user_name,
            message_count=message_count,
        )

        history = list(conversation_history or [])[-10:]
        contents = self._history_to_contents(history)

        # Не дублируем user-сообщение, если оно уже в истории
        last_is_same = (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content") == user_message
        )
        if not last_is_same:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_message)],
                )
            )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            #tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.75,
            max_output_tokens=2500,
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            logger.info(
                f"[GEMINI] OK, is_owner={is_owner}, "
                f"contents={len(contents)}, model={self.model}"
            )
        except Exception as e:
            logger.error(f"[GEMINI] FAILED: {type(e).__name__}: {e}")
            raise

        return response.text or "Секунду, Сэр. Что-то не отвечает."

    async def close(self):
        # у google-genai клиент не требует явного закрытия, но оставим на будущее
        pass


ai_handler = AIHandler()
