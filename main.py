from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="BizApps Incentives Agent")

# CORS: allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # *
    allow_methods=["*"],      # *
    allow_headers=["*"],      # *
    allow_credentials=False,  # must be False when allow_origins=["*"]
)

# include router
app.include_router(api_router, prefix="/api/v1")
