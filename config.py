"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required")

# Gemini AI Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

OWNER_NAME = os.environ.get("OWNER_NAME", "Сиаленс")

# Optional: Database for conversation history
DATABASE_URL = os.environ.get("DATABASE_URL")

# Webhook configuration
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", "/webhook")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
PORT = int(os.environ.get("PORT", "10000"))

# Owner configuration
OWNER_USER_ID = int(os.environ.get("OWNER_USER_ID", "7857165309"))

# Часовой пояс
TZ_OFFSET_HOURS = int(os.environ.get("TZ_OFFSET_HOURS", "3"))


def webhook_url() -> str | None:
    """Return the public webhook URL when the service is configured for webhooks."""
    if not RENDER_EXTERNAL_URL:
        return None
    return f"{RENDER_EXTERNAL_URL}{WEBHOOK_PATH}"
