from fastapi import FastAPI
from app.search import search_products

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Search API running 🚀"}

@app.get("/search")
def search(query: str):
    return search_products(query)