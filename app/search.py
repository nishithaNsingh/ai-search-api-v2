import numpy as np
from sentence_transformers import SentenceTransformer
from app.model_loader import product_embeddings, products_df

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_products(query, top_k=10):

    query_embedding = model.encode([query], normalize_embeddings=True)

    similarities = np.dot(product_embeddings, query_embedding.T).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = products_df.iloc[top_indices]

    results = results[
        ["name", "main_category", "sub_category", "discount_price", "image", "link"]
    ]

    return results.fillna("").to_dict(orient="records")