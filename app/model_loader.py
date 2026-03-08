import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

print("Loading model and data...")

model = SentenceTransformer("all-MiniLM-L6-v2")

product_embeddings = np.load("embeddings/product_embeddings_v1.npy")

products_df = pd.read_csv("data/cleaned_products.csv")

print("Model and data loaded.")