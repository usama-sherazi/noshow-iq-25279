from fastapi import FastAPI

app = FastAPI(title="NoShowIQ")


@app.get("/health")
def health():
    return {"status": "ok"}
