import os
from dotenv import load_dotenv
from pathlib import Path

# Грузим .env рядом с этим файлом (а не из текущей папки)
load_dotenv(Path(__file__).resolve().parent / ".env")


API_URL = os.getenv("API_URL", "http://opendata.trudvsem.ru/api/v1/vacancies/region/39")
MAX_VACANCIES = int(os.getenv("MAX_VACANCIES", "5"))
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", "10"))
SCHEDULE_INTERVAL_MINUTES = int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "60"))
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output/report.json")

# === LLM: читаем те же имена, что в .env ===
# Приоритет: PROXY_URL > LLM_BASE_URL > дефолт
LLM_BASE_URL = (
    os.getenv("PROXY_URL")
    or os.getenv("LLM_BASE_URL")
    or "http://localhost:8090/v1"
)
LLM_API_KEY = (
    os.getenv("GPT2GIGA_API_KEY")
    or os.getenv("LLM_API_KEY")
    or ""
)
LLM_MODEL = (
    os.getenv("GIGACHAT_MODEL")
    or os.getenv("LLM_MODEL")
    or "GigaChat-2-Max"
)


# Диагностика при импорте — сразу видно, куда пойдёт запрос
#print(f"[CONFIG] LLM_BASE_URL = {LLM_BASE_URL}")
#print(f"[CONFIG] LLM_MODEL    = {LLM_MODEL}")
#print(f"[CONFIG] LLM_API_KEY  = {'*' * 8 if LLM_API_KEY else 'НЕ ЗАДАН'} (len={len(LLM_API_KEY)})")
