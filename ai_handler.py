"""AI handler for Groq API integration."""

import logging
from typing import List, Dict
from groq import AsyncGroq

from config import GROQ_API_KEY, GROQ_MODEL, OWNER_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIHandler:
    def init(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        self.client = None
        
        self.system_prompt = f"""Ты - AI-ассистент по имени Картер для {OWNER_NAME}. Твоя задача - отвечать на сообщения людей, когда {OWNER_NAME} занят и не в сети.

Правила:
1. Будь вежливым и профессиональным
2. Отвечай кратко и по делу
3. Если вопрос сложный - предложи подождать ответа от {OWNER_NAME}
4. Не выдумывай информацию, которой не знаешь
5. Используй русский язык
6. Если спрашивают что-то тебе не знакомое - говори что нужно уточнить у {OWNER_NAME}
7. Тон общения - дружелюбный, но деловой, можешь редко подшучивать
"""

    async def get_client(self):
        if self.client is None:
            if not self.api_key:
                logger.warning("GROQ_API_KEY not set")
                return None
            self.client = AsyncGroq(api_key=self.api_key)
        return self.client

    async def generate_response(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        client = await self.get_client()
        if not client:
            raise RuntimeError("AI не настроен. Проверьте GROQ_API_KEY.")

        messages = [{"role": "system", "content": self.system_prompt}]

        if conversation_history:
            messages.extend(conversation_history[-10:])

        messages.append({"role": "user", "content": user_message})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )

        return response.choices[0].message.content

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None


ai_handler = AIHandler()