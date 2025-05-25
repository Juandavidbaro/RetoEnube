import json
import uuid
from langchain_core.documents import Document

def load_mock_articles(path: str) -> list[Document]:
    # Carga artículos simulados desde un archivo JSON y los convierte en objetos Document 
    # de LangChain, incluyendo su contenido y metadatos.

    # Abre y carga el archivo JSON
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # Transforma cada artículo en un objeto Document
    docs = [
        Document(
            # El contenido del documento incluye el título y el contenido del artículo
            page_content=f"{a['title']}\n\n{a['content']}",
            # Metadatos del documento, como ID único, título y categoría
            metadata={
                "id": str(uuid.uuid4()),  # Genera un ID único
                "title": a['title'],
                "category": a.get("category", "all")  # Usa "all" si no hay categoría
            }
        )
        for a in articles  # Itera sobre cada artículo en la lista
    ]

    return docs
