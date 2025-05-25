from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    session_id: str

class ChatWithArticleRequest(BaseModel):
    session_id: str
    article_id: str
    question: str
