import json
import uuid
from langchain_core.documents import Document

def load_mock_articles(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    docs = [
        Document(
            page_content=f"{a['title']}\n\n{a['content']}",
            metadata={
                "id": str(uuid.uuid4()),
                "title": a['title'],
                "category": a.get("category", "General")
            }
        )
        for a in articles
    ]
    return docs
