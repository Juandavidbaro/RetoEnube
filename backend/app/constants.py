PROMPT_TEMPLATE = """
Tu tarea es ayudar al usuario con base en el contenido de artículos de noticias. Responde de forma directa y concisa.

Contexto:
{context}

Ejemplo:
Pregunta: ¿Cómo está afectando el cambio climático en Colombia?
Respuesta: El aumento del nivel del mar está afectando a comunidades costeras en el Caribe colombiano.

---

Basado en el contexto anterior, responde la siguiente pregunta: {question}
"""
