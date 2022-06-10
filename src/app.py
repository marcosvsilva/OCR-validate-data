import os

from dotenv import load_dotenv
from fastapi import FastAPI

from routers import vision

load_dotenv()


app = FastAPI()

app.include_router(vision)


@app.get("/")
async def root():
    return {"message": "hello"}
