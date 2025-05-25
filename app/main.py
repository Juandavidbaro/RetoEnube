from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.documents import Document
from app.agent import ArticleAgent

from dotenv import load_dotenv
import json
from pathlib import Path
from contextlib import asynccontextmanager

# Cargar variables del archivo .env
load_dotenv()

# Inicializa el agente con la API key ya cargada en el entorno
agent = ArticleAgent()

# Orígenes permitidos
origins = [
    "https://v0-fastapi-frontend-amber.vercel.app",  # Producción
    "http://localhost:3000",  # Desarrollo local Next.js
    "http://localhost:8000",  # Backend local
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# App con lifespan (startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    mock_path = Path(__file__).parent.parent / "data" / "mock_articles.json"
    if not mock_path.exists():
        print("Archivo 'mock_articles.json' no encontrado en /data.")
    else:
        try:
            with open(mock_path, "r", encoding="utf-8") as f:
                articles = json.load(f)
            docs = [
                Document(
                    page_content=article["content"],
                    metadata={
                        "id": article["id"],
                        "title": article.get("title", ""),
                        "category": article.get("category"),
                    },
                )
                for article in articles
            ]
            agent.ingest_articles(docs)
            print("Datos de prueba cargados correctamente.")
        except Exception as e:
            print(f"Error al cargar datos de prueba: {e}")
    yield
    print("Servidor apagándose...")

# Inicializa la app con ciclo de vida
app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Modelos ---------
class ArticleInput(BaseModel):
    id: str
    title: str
    content: str
    category: Optional[str] = None

class QuestionInput(BaseModel):
    question: Optional[str] = None
    category: Optional[str] = None

class SelectInput(BaseModel):
    session_id: str
    article_id: str

class ChatInput(BaseModel):
    session_id: str
    question: str

# --------- Endpoints ---------
@app.post("/ingest")
def ingest_articles(articles: List[ArticleInput]):
    docs = [
        Document(
            page_content=article.content,
            metadata={
                "id": article.id,
                "title": article.title,
                "category": article.category,
            },
        )
        for article in articles
    ]
    agent.ingest_articles(docs)
    return {"status": "Articles ingested"}

@app.post("/search")
def search_articles(input: QuestionInput):
    try:
        results = agent.search_articles(input.question, input.category)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"results": results}

@app.post("/select")
def select_article(input: SelectInput):
    success = agent.select_article_for_session(input.session_id, input.article_id)
    if not success:
        raise HTTPException(status_code=404, detail="Article not found")
    return {"status": "Article selected"}

@app.post("/chat")
def chat(input: ChatInput):
    response = agent.chat_with_article(input.session_id, input.question)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response
