import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")  # postgresql://user:pass@host:5432/db
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
