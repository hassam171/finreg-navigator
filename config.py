from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_VISION_API_KEY = os.getenv("GEMINI_VISION_API_KEY")
OPENAI_VISION_API_KEY = os.getenv("OPENAI_VISION_API_KEY")
OPENAI_VISION_API_KEY_PAID = os.getenv("OPENAI_VISION_API_KEY_PAID")