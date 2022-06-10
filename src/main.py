from fastapi import FastAPI
from dotenv import load_dotenv

from routers import vision

load_dotenv()

app = FastAPI()

app.include_router(vision.router)

@app.get("/")
async def root():
    return {"message": "🚀 works!"}
