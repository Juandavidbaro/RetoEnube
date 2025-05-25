from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
import os
from typing import List, Optional
from qdrant_client.http.models import FieldCondition, MatchValue, Filter
from app.constants import PROMPT_TEMPLATE

# Configuración de entorno y constantes para Qdrant y OpenAI
QDRANT_COLLECTION = "articles"
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ArticleAgent:
    def __init__(self):
        # Inicializa el modelo de embeddings y el modelo conversacional (LLM)
        self.embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

        # Almacenes para sesiones de usuario y artículos seleccionados
        self.sessions = {}
        self.selected_articles = {}

        # Plantilla del sistema para construir el prompt que será enviado al LLM
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE),
            ("human", "Basado en el contexto anterior, responde la siguiente pregunta: {question}")
        ])

        # Composición de la cadena de ejecución con historial de mensajes por sesión
        self.runnable = RunnableWithMessageHistory(
            self.prompt | self.llm,
            self._get_session_history,  
            input_messages_key="question",
            history_messages_key="history"
        )

        # Cliente de Qdrant para manipular la base vectorial
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.db: Optional[Qdrant] = None

    def debug_print_payload(self):
        # Método de depuración para inspeccionar el payload del primer punto en Qdrant
        points, _ = self.qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1,
            with_payload=True
        )
        if points:
            print("Payload del primer punto:", points[0].payload)
        else:
            print("No hay puntos en la colección.")

    def ingest_articles(self, docs: List[Document]):
        # Recrea la colección de Qdrant y carga los artículos
        self.qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        # Extrae texto y metadatos de los documentos
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        # Ingresa los documentos a la base de datos vectorial
        self.db = Qdrant.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas,
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            collection_name=QDRANT_COLLECTION
        )

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        # Crea o recupera el historial de mensajes para una sesión dada
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    def search_articles(self, question: Optional[str] = None, category: Optional[str] = None):
        # Verifica si la base de datos está disponible
        if self.db is None:
            raise ValueError("La base de datos no está inicializada. Ingeste artículos primero.")

        # Caso 1: búsqueda por categoría sin pregunta → scroll manual
        if not question and category:
            points, _ = self.qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=100,
                with_payload=True
            )

            # Filtrado manual 
            filtered_points = [
                point for point in points
                if point.payload.get("metadata", {}).get("category", "").lower() == category.lower()
            ][:20]

            return [
                {
                    "id": point.payload.get("metadata", {}).get("id"),
                    "title": point.payload.get("metadata", {}).get("title"),
                    "category": point.payload.get("metadata", {}).get("category"),
                    "snippet": point.payload.get("page_content", "")[:200] + "...",
                    "score": 0.0  # No es una búsqueda semántica, no hay score real
                }
                for point in filtered_points
            ]

        # Caso 2: búsqueda semántica con o sin categoría
        query_text = question or ""
        filters = None
        if category:
            # Construcción de filtro por categoría para búsqueda semántica
            filters = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        # Búsqueda semántica con Qdrant
        docs_with_scores = self.db.similarity_search_with_score(query=query_text, k=5, filter=filters)

        return [
            {
                "id": doc.metadata.get("id"),
                "title": doc.metadata.get("title"),
                "category": doc.metadata.get("category"),
                "snippet": doc.page_content[:200] + "...",
                "score": score  # Score de similitud semántica
            }
            for doc, score in docs_with_scores
        ]

    def select_article_for_session(self, session_id: str, article_id: str) -> bool:
        # Selecciona un artículo por ID y lo asocia a una sesión
        docs = self.db.similarity_search("", k=100)  
        for doc in docs:
            if doc.metadata.get("id") == article_id:
                self.selected_articles[session_id] = doc
                return True
        return False  # No se encontró el artículo

    def chat_with_article(self, session_id: str, question: str):
        # Recupera el artículo seleccionado por la sesión
        doc = self.selected_articles.get(session_id)
        if not doc:
            return {"error": "No article selected for this session."}

        # Extrae el contexto del artículo y el historial de la sesión
        context = doc.page_content
        history = self._get_session_history(session_id)
        history.add_user_message(question)

        # Ejecuta la conversación utilizando el runnable con contexto e historial
        response = self.runnable.invoke(
            {"context": context, "question": question},
            config={"configurable": {"session_id": session_id}}
        )

        history.add_ai_message(response)
        return {"answer": response}
