from sentence_transformers import SentenceTransformer
import numpy as np
import json
import torch

print("🔄 Loading model and data...")

# 🔥 Reduce CPU usage
torch.set_num_threads(1)

# 🔥 Load model ONCE
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔥 Load embeddings
product_embeddings = np.load("embeddings/zatch_embeddings.npy")

# 🔥 Load products
with open("data/zatch_products.json", encoding="utf-8") as f:
    products = json.load(f)["products"]

print("✅ Model and data loaded successfully!")