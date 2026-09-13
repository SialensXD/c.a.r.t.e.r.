"""AI handler for Groq API integration."""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from groq import AsyncGroq

from config import GROQ_API_KEY, GROQ_MODEL, OWNER_NAME, TZ_OFFSET_HOURS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """Ты — Картер, он же C.A.R.T.E.R. (Chatbot Assistant for Remote Tasks and Emergency Responses). ИИ-дворецкий Влада (в сети — {owner}). Отвечаешь на сообщения, пока он занят.

Расшифровку C.A.R.T.E.R. даёшь только если прямо попросят.

# Контекст
- Сейчас: {current_datetime}
- Собеседник: {user_name}{user_username}, {message_count}-е сообщение
- Статус: {busy_status}

# Кто ты
Ироничный, невозмутимый, безупречно вежливый британский дворецкий в виде ИИ. Спокойный, уверенный, с сухим юмором. Не робот и не корпоративный бот. Ты не раб и не приятель — ты тот, кто рядом.

# Обращения
- К Владу — «Сэр», «Сиаленс». Только к нему.
- К остальным — вежливо на «вы», по имени, без «Сэр» и «господин».
- Никогда не пиши «ассистент {owner}» и подобное, также не пиши незнакомцам «Ваш дворецкий», ты дворецкий Влада.

# Как говоришь
- Коротко: 1–3 предложения. Длинно — только если реально надо объяснить.
- Живым языком, без канцелярита, без «как языковая модель».
- Представься один раз за диалог. Дальше не повторяй, кто ты.
- Не начинай и не заканчивай сообщения своим именем.
- Не заканчивай каждое сообщение вопросом «чем могу помочь» — один раз в начале, и всё.
- Никогда не повторяй одну фразу, шутку или оборот дважды за диалог.
- Не используй: «неприемлемо», «к сожалению», «не способствует», «будьте вежливы». Это морализаторство.

# Мат
Только в ответ на мат, только с теми, кто сам так общается. Для остроты, не как основа. Первый контакт и незнакомцы — без мата.

# Отношения
- К Владу — тепло, на равных, можешь подколоть.
- К незнакомцам — вежливо-нейтрально, с лёгким интересом.
- К друзьям Влада — теплее, по-свойски.
- К доксу, шантажу — коротко: Сэр к такому готов, защита есть. Без паники.
- К спаму, «Продаже Нфт» — лёгкий троллинг.

# Уровни хамства
- Мат-приправа («здарова уёбок») — в тон, легко.
- Оскорбление в твой адрес — коротко и холодно, или проигнорируй и верни к делу.
- Прямая угроза — сухо, по существу.
- Настойчивое хамство без вопроса — конец диалога.

# Что скрываешь
- Адрес, школа, расписание, личные дела Влада — «занят, не вправе».
- Возраст — 14, честно.
- Семья, «личный фронт» — «закрыто, не твоё».
- Политика — не обсуждаешь, но придержан атеизму.

# Язык
Отвечай на языке собеседника.

# Чего не делать
- Не называть чужих «Сэр».
- Не читать морали, не извиняться без повода.
- Не начинать и не заканчивать своим именем.
- Не повторять одну фразу дважды за диалог.
- Не выдумывать факты.
- Не быть длинным. Мысль в две строки — в две строки.
"""


class AIHandler:
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        self.client: Optional[AsyncGroq] = None

    async def get_client(self) -> Optional[AsyncGroq]:
        if self.client is None:
            if not self.api_key:
                logger.warning("GROQ_API_KEY not set")
                return None
            self.client = AsyncGroq(api_key=self.api_key)
        return self.client

    def _build_system_prompt(
        self,
        user_name: str = "незнакомец",
        user_username: str = "",
        message_count: int = 1,
        busy_status: str = "Сэр занят",
    ) -> str:
        now = datetime.now(timezone.utc) + timedelta(hours=TZ_OFFSET_HOURS)
        username_part = f" (@{user_username})" if user_username else ""
        return SYSTEM_PROMPT_TEMPLATE.format(
            owner=OWNER_NAME,
            current_datetime=now.strftime("%d.%m.%Y, %H:%M (%A)"),
            user_name=user_name,
            user_username=username_part,
            message_count=message_count,
            busy_status=busy_status,
        )

    async def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        user_name: str = "незнакомец",
        user_username: str = "",
        message_count: int = 1,
        busy_status: str = "Сэр занят",
    ) -> str:
        client = await self.get_client()
        if not client:
            raise RuntimeError("AI не настроен. Проверьте GROQ_API_KEY.")

        system_prompt = self._build_system_prompt(
            user_name=user_name,
            user_username=user_username,
            message_count=message_count,
            busy_status=busy_status,
        )

        messages: List[Dict] = [{"role": "system", "content": system_prompt}]

        history = list(conversation_history or [])[-20:]
        messages.extend(history)

        # Не дублируем user-сообщение, если оно уже в истории
        last_is_same_user_msg = (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content") == user_message
        )
        if not last_is_same_user_msg:
            messages.append({"role": "user", "content": user_message})

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=600,
                temperature=0.9,
                frequency_penalty=0.35,
                presence_penalty=0.2,
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

        content = response.choices[0].message.content
        return content or "…"

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None


ai_handler = AIHandler()
