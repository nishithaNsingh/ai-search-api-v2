import numpy as np
import pandas as pd

print("Loading embeddings and dataset...")

product_embeddings = np.load("embeddings/product_embeddings_v1.npy")

products_df = pd.read_csv("data/cleaned_products.csv")

print("Data loaded successfully.")