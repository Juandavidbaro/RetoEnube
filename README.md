# Article Semantic Search App

Aplicación fullstack que permite la ingesta, almacenamiento y consulta de artículos mediante búsqueda semántica, utilizando Qdrant como base de datos vectorial y OpenAI para generar embeddings y respuestas conversacionales.

---

## Arquitectura del Proyecto

- **Backend:** FastAPI + LangChain + OpenAI + Qdrant
- **Base de datos vectorial:** Qdrant
- **Frontend:** [SPA desplegada en Vercel](https://v0-fastapi-frontend-amber.vercel.app)
- **Contenerización:** Docker + Docker Compose

---

## Instrucciones de Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/Juandavidbaro/RetoEnube.git
cd RetoEnube
```

### 2. Configurar variables de entorno

Crea un archivo .env dentro del directorio backend con el siguiente contenido:

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxx
```

### 3. Construir y ejecutar con Docker Compose

**Asegúrate de tener Docker Desktop abierto y corriendo antes de continuar.**

Desde la raíz del proyecto:

```bash
docker-compose up --build
```

Esto lanzará:

- Qdrant en localhost:6333
- Backend FastAPI en localhost:8000

## Endpoints del Backend

- POST /ingest: Cargar artículos (individual o lote).
- POST /search: Buscar artículos por pregunta o categoría.
- POST /select: Seleccionar un artículo para una sesión.
- POST /chat: Enviar una pregunta dentro de una sesión con artículo seleccionado.

## Acceso al Frontend

Una vez que el backend esté activo, puedes acceder al frontend en:

[Panel de busqueda de artículos](https://v0-fastapi-frontend-amber.vercel.app)

Desde allí podrás:

- Buscar artículos por texto o categoría
- Seleccionar un artículo
- Hacer preguntas sobre su contenido usando el chat con contexto

## Panel de Administración — /admin

[Panel de administración](https://v0-fastapi-frontend-amber.vercel.app/admin)

Puedes agregar manualmente artículos a la base de datos:

1. Completa los campos: ID (opcional), Título, Categoría, y Texto.

2. Haz clic en Agregar artículo. El artículo se añade a una lista pendiente.

3. Una vez agregados todos los deseados, haz clic en Subir todos.

4. Los artículos serán enviados al backend y almacenados en la base vectorial para futuras búsquedas semánticas.

