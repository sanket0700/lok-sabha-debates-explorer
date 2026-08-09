from fastapi import FastAPI, Request

from app.routers import insights, qa, search
from app.templates_env import templates

app = FastAPI(title="Lok Sabha Debates Explorer")

app.include_router(search.router)
app.include_router(qa.router)
app.include_router(insights.router)


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})
