from fastapi import FastAPI

app = FastAPI(title="RAGsentinel API")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "ragsentinel-api", "status": "ok"}