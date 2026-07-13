from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import admin, auth, chat, retrieval

app = FastAPI(title="NewsQA-RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(retrieval.router, prefix="/retrieval", tags=["retrieval"])


@app.get("/health")
def health():
    return {"status": "ok"}
