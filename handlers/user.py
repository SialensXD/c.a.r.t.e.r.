"""Basic user-facing commands for AI assistant."""

from aiogram import types
from aiogram.filters import Command


def register_user_handlers(dp):
    @dp.message(Command("start"))
    async def cmd_start(message: types.Message):
        """Приветственное сообщение."""
        text = (
            f"👋 <b>Привет, {message.from_user.first_name}!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Я AI-ассистент. Помогаю отвечать на сообщения, когда владелец занят.\n\n"
            f"<b>📋 Основные команды:</b>\n"
            f"/help - список команд\n"
            f"/status - статус бота\n\n"
            f"<b>💡 Как это работает:</b>\n"
            f"Когда владелец занят, я автоматически отвечу на твоё сообщение.\n"
            f"Если вопрос сложный - предложу подождать ответа от владельца.\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        await message.answer(text, parse_mode="HTML")

    @dp.message(Command("help"))
    async def cmd_help(message: types.Message):
        """Справка по командам."""
        text = (
            "📖 <b>СПРАВКА</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>🤖 Для всех:</b>\n"
            "/start - начать работу\n"
            "/help - эта справка\n"
            "/status - статус бота\n\n"
            "<b>👑 Для владельца:</b>\n"
            "/busy - включить режим занятости\n"
            "/available - выключить режим занятости\n"
            "/reset - сбросить историю всех диалогов\n"
            "/resetuser user_id - сбросить историю для пользователя\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<i>AI отвечает автоматически, когда включен режим занятости.</i>"
        )
        await message.answer(text, parse_mode="HTML")
