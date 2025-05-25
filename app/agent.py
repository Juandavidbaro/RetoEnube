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

QDRANT_COLLECTION = "articles"
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ArticleAgent:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        self.sessions = {}
        self.selected_articles = {}
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE),
            ("human", "Basado en el contexto anterior, responde la siguiente pregunta: {question}")
        ])
        self.runnable = RunnableWithMessageHistory(
            self.prompt | self.llm,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.db: Optional[Qdrant] = None

    def debug_print_payload(self):
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
        self.qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        self.db = Qdrant.from_texts(
        texts=texts,
        embedding=self.embedding_model,
        metadatas=metadatas,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        collection_name=QDRANT_COLLECTION
    )


    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    def search_articles(self, question: Optional[str] = None, category: Optional[str] = None):
        if self.db is None:
            raise ValueError("La base de datos no está inicializada. Ingeste artículos primero.")

        # Si no hay pregunta pero sí categoría → usa scroll SIN filtro porque scroll no acepta filtro
        if not question and category:
            points, _ = self.qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=100,
                with_payload=True
            )

            print("Categorías en la colección:")
            for point in points:
                cat = point.payload.get("metadata", {}).get("category")
                print(cat)

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
                    "score": 0.0
                }
                for point in filtered_points
            ]


        # Si hay pregunta (con o sin categoría), usar búsqueda semántica con filtro (search acepta filtro)
        query_text = question or ""
        filters = None
        if category:
            filters = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category)
                    )
                ]
            )

        docs_with_scores = self.db.similarity_search_with_score(query=query_text, k=5, filter=filters)

        return [
            {
                "id": doc.metadata.get("id"),
                "title": doc.metadata.get("title"),
                "category": doc.metadata.get("category"),
                "snippet": doc.page_content[:200] + "...",
                "score": score
            }
            for doc, score in docs_with_scores
        ]


    def select_article_for_session(self, session_id: str, article_id: str) -> bool:
        docs = self.db.similarity_search("", k=100)
        for doc in docs:
            if doc.metadata.get("id") == article_id:
                self.selected_articles[session_id] = doc
                return True
        return False

    def chat_with_article(self, session_id: str, question: str):
        doc = self.selected_articles.get(session_id)
        if not doc:
            return {"error": "No article selected for this session."}

        context = doc.page_content
        history = self._get_session_history(session_id)
        history.add_user_message(question)

        response = self.runnable.invoke(
            {"context": context, "question": question},
            config={"configurable": {"session_id": session_id}}
        )

        history.add_ai_message(response)
        return {"answer": response}
