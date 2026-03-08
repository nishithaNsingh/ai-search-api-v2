import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = np.load(os.path.join(BASE_DIR, "embeddings", "product_embeddings_v1.npy"))

products_df = pd.read_csv(os.path.join(BASE_DIR, "data", "cleaned_products.csv"))

# remove NaN values (important for JSON)
products_df = products_df.fillna("")


def search_products(query, top_k=10):

    query_embedding = model.encode([query], normalize_embeddings=True)

    similarities = np.dot(embeddings, query_embedding.T).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = products_df.iloc[top_indices]

    # display only required fields
    results = results[
        ["name", "category_source","discount_price"]
    ]

    results = results.replace([np.inf, -np.inf], "").fillna("")

    return results.to_dict(orient="records")