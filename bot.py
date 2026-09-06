print("=== СТАРТ AI-АССИСТЕНТА ===", flush=True)
import os
import sys
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config import (
    OWNER_USER_ID,
    BOT_TOKEN,
    DATABASE_URL,
    PORT,
    WEBHOOK_SECRET,
    WEBHOOK_PATH,
    webhook_url,
)
from ai_handler import ai_handler
from handlers.user import register_user_handlers

from aiohttp import web
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.types import BotCommand

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
db_pool = None

START_TIME = datetime.now(timezone.utc)
APP_ROOT = Path(__file__).resolve().parent

busy_mode = False
busy_since = None
conversation_history = {}


async def health_check(request):
    """Эндпоинт проверки здоровья бота для Render."""
    return web.Response(text="OK", status=200)


async def init_db():
    global db_pool
    if not DATABASE_URL:
        logging.info("⚠️ База данных не настроена. Работаем без БД.")
        return

    try:
        import asyncpg
        dsn = DATABASE_URL.replace("postgres://", "postgresql://")
        db_pool = await asyncpg.create_pool(
            dsn=dsn,
            statement_cache_size=0,
            min_size=1,
            max_size=5,
        )
        async with db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_ai_response TIMESTAMPTZ,
                    message_count INTEGER NOT NULL DEFAULT 0
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_history_user_id 
                ON conversation_history(user_id, created_at DESC);
            """)
        logging.info("✅ Подключение к БД успешно!")
    except Exception as e:
        logging.warning(f"⚠️ Не удалось подключиться к БД: {e}")


async def save_user_message(user_id: int, username: str, first_name: str):
    if not db_pool:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, username, first_name, message_count)
                VALUES ($1, $2, $3, 1)
                ON CONFLICT (user_id) DO UPDATE 
                SET username = EXCLUDED.username,
                    first_name = EXCLUDED.first_name,
                    message_count = users.message_count + 1
                """,
                user_id, username, first_name
            )
    except Exception as e:
        logging.error(f"Ошибка сохранения пользователя: {e}")


async def save_conversation_message(user_id: int, role: str, content: str):
    if not db_pool:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversation_history (user_id, role, content)
                VALUES ($1, $2, $3)
                """,
                user_id, role, content
            )
    except Exception as e:
        logging.error(f"Ошибка сохранения сообщения в историю: {e}")


async def load_conversation_history(user_id: int, limit: int = 20) -> list:
    if not db_pool:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content 
                FROM conversation_history 
                WHERE user_id = $1 
                ORDER BY created_at DESC 
                LIMIT $2
                """,
                user_id, limit
            )
            return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
    except Exception as e:
        logging.error(f"Ошибка загрузки истории: {e}")
        return []


async def clear_conversation_history(user_id: int):
    if not db_pool:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM conversation_history WHERE user_id = $1", user_id)
    except Exception as e:
        logging.error(f"Ошибка очистки истории: {e}")


def get_busy_keyboard():
    builder = InlineKeyboardBuilder()
    if busy_mode:
        builder.button(text="🟢 Доступен", callback_data="toggle_busy")
    else:
        builder.button(text="🔴 Занят", callback_data="toggle_busy")
    builder.adjust(1)
    return builder.as_markup()


@dp.callback_query(F.data == "toggle_busy")
async def toggle_busy_mode(callback: types.CallbackQuery):
    global busy_mode, busy_since
    if callback.from_user.id != OWNER_USER_ID:
        await callback.answer("⚠️ Эта команда только для владельца!", show_alert=True)
        return

    busy_mode = not busy_mode
    busy_since = datetime.now(timezone.utc) if busy_mode else None
    status = "занятости" if busy_mode else "доступности"
    await callback.answer(f"✅ Режим {status} включен!")

    await callback.message.edit_text(
        f"👀 <b>Привет, я из будущего, снова фиксы?</b>\n\n"
        f"Режим: {'🔴 ЗАНЯТ' if busy_mode else '🟢 ДОСТУПЕН'}\n"
        f"{'Включен: ' + busy_since.strftime('%H:%M') if busy_since else 'Отключен'}\n\n"
        f"{'Картер будет отвечать на сообщения вместо тебя.' if busy_mode else 'Ты отвечаешь на сообщения сам.'}",
        reply_markup=get_busy_keyboard(),
        parse_mode="HTML"
    )


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    uptime = datetime.now(timezone.utc) - START_TIME
    uptime -= timedelta(microseconds=uptime.microseconds)
    ping_ms = (datetime.now(timezone.utc) - message.date).total_seconds() * 1000

    text = (
        "🤖 <b>СТАТУС AI-АССИСТЕНТА</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🟢 Состояние: <b>Онлайн</b>\n\n"
        f"⏱  Аптайм: <b>{uptime}</b>\n\n"
        f"🏓 Задержка: <b>~{int(ping_ms)} мс</b>\n\n"
        f"🤖 Режим: <b>{'🔴 ЗАНЯТ' if busy_mode else '🟢 ДОСТУПЕН'}</b>\n\n"
        f"💬 Диалогов в памяти: <b>{len(conversation_history)}</b>\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if message.from_user.id == OWNER_USER_ID:
        text += "\n\n" + ("🔴 Картер отвечает вместо тебя" if busy_mode else "🟢 Ты отвечаешь сам")
        await message.answer(text, reply_markup=get_busy_keyboard(), parse_mode="HTML")
    else:
        await message.answer(text, parse_mode="HTML")


@dp.message(Command("busy"))
async def cmd_busy(message: types.Message):
    if message.from_user.id != OWNER_USER_ID:
        await message.answer("⚠️ Эта команда только для владельца!")
        return
    global busy_mode, busy_since
    busy_mode = True
    busy_since = datetime.now(timezone.utc)
    await message.answer(
        "🔴 <b>РЕЖИМ ЗАНЯТОСТИ ВКЛЮЧЕН</b>\n\nAI-ассистент будет отвечать на сообщения вместо тебя.",
        reply_markup=get_busy_keyboard(),
        parse_mode="HTML"
    )


@dp.message(Command("available"))
async def cmd_available(message: types.Message):
    if message.from_user.id != OWNER_USER_ID:
        await message.answer("⚠️ Эта команда только для владельца!")
        return
    global busy_mode, busy_since
    busy_mode = False
    busy_since = None
    await message.answer(
        "🟢 <b>РЕЖИМ ДОСТУПНОСТИ ВКЛЮЧЕН</b>\n\nТеперь ты отвечаешь на сообщения сам.",
        reply_markup=get_busy_keyboard(),
        parse_mode="HTML"
    )
@dp.message(Command("reset"))
async def cmd_reset(message: types.Message):
    if message.from_user.id != OWNER_USER_ID:
        await message.answer("⚠️ Эта команда только для владельца!")
        return
    global conversation_history
    conversation_history = {}
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM conversation_history")
        except Exception as e:
            logging.error(f"Ошибка очистки истории в БД: {e}")
    await message.answer("🗑 <b>История диалогов сброшена!</b>", parse_mode="HTML")


@dp.message(Command("resetuser"))
async def cmd_reset_user(message: types.Message):
    if message.from_user.id != OWNER_USER_ID:
        await message.answer("⚠️ Эта команда только для владельца!")
        return
    try:
        args = message.text.split()
        if len(args) < 2:
            await message.answer("Использование: /resetuser user_id")
            return
        target_user_id = int(args[1])
        if target_user_id in conversation_history:
            del conversation_history[target_user_id]
        if db_pool:
            await clear_conversation_history(target_user_id)
        await message.answer(f"🗑 История диалога для пользователя {target_user_id} сброшена!")
    except ValueError:
        await message.answer("⚠️ Неверный формат user_id. Используйте число.")

@dp.business_message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    username = message.from_user.username or ""
    first_name = message.from_user.first_name

    await save_user_message(user_id, username, first_name)

    if user_id == OWNER_USER_ID or not busy_mode:
        return

    if message.text and message.text.startswith("/"):
        return

    # Отправляем статус "печатает..."
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    if db_pool:
        history = await load_conversation_history(user_id, limit=20)
    else:
        history = conversation_history.get(user_id, [])

    history.append({"role": "user", "content": message.text})
    if len(history) > 20:
        history = history[-20:]

    try:
        ai_response = await ai_handler.generate_response(message.text, history)
        
        # Обрезаем ответ, если он превышает лимит Telegram
        if len(ai_response) > 4000:
            ai_response = ai_response[:4000] + "..."

        # Сохраняем в историю диалога и в базу данных
        history.append({"role": "assistant", "content": ai_response})
        conversation_history[user_id] = history

        await save_conversation_message(user_id, "user", message.text)
        await save_conversation_message(user_id, "assistant", ai_response)

        # Пробуем отправить с Markdown, в случае ошибки парсинга отправляем чистый текст
        try:
            await message.answer(ai_response, parse_mode="Markdown")
        except Exception:
            await message.answer(ai_response)

    except Exception as e:
        logging.error(f"Ошибка обработки сообщения: {e}")
        await message.answer(f"⚠️ ошибка: {e}")


async def on_startup(dispatcher: Dispatcher):
    await init_db()
    register_user_handlers(dispatcher)
    await bot.set_my_commands([
        BotCommand(command="start", description="Начать"),
        BotCommand(command="help", description="Помощь"),
        BotCommand(command="status", description="Статус бота"),
        BotCommand(command="busy", description="Включить режим занятости"),
        BotCommand(command="available", description="Выключить режим занятости"),
        BotCommand(command="reset", description="Сбросить историю всех диалогов"),
        BotCommand(command="resetuser", description="Сбросить историю для пользователя"),
    ])
    logging.info("✅ Бот инициализирован!")


async def on_shutdown(dispatcher: Dispatcher):
    await ai_handler.close()
    if db_pool:
        await db_pool.close()
    logging.info("👋 Бот остановлен")


async def main():
    webhook = webhook_url()

    if webhook:
        app = web.Application()
        app.router.add_get("/health", health_check)

        SimpleRequestHandler(
            dispatcher=dp,
            bot=bot,
            secret_token=WEBHOOK_SECRET,
        ).register(app, path=WEBHOOK_PATH)
        
        setup_application(app, dp, bot=bot)
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)

        await bot.set_webhook(
            url=webhook,
            secret_token=WEBHOOK_SECRET,
            drop_pending_updates=True
        )

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
        await site.start()

        logging.info(f"🚀 Webhook запущен на порту {PORT}")
        await asyncio.Event().wait()
    else:
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)
        await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("👋 Бот остановлен пользователем")
    except Exception as e:
        logging.error(f"❌ Критическая ошибка: {e}")
        raise
