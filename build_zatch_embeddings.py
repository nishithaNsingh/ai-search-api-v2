import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading Zatch data...")

# ✅ FIXED loading (no encoding issues)
with open("data/zatch_products.json", "rb") as f:
    data = json.loads(f.read().decode("utf-8", errors="ignore"))["products"]

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
products = []

print("Preparing product text...")

for product in data:

    # ✅ Variant info (color + size)
    variant_text = ""
    for v in product.get("variants", []):
        variant_text += f"{v.get('color','')} {v.get('size','')} "

    # ✅ Smart keywords (boost accuracy)
    keywords = ""
    sub = product.get("subCategory", "").lower()

    if "shirt" in sub:
        keywords += " shirt clothing men fashion casual formal"
    if "washing" in sub:
        keywords += " washing machine electronics appliance home"
    if "clock" in sub:
        keywords += " wall clock home decor time"
    if "pant" in sub or "trouser" in sub:
        keywords += " pants trousers men clothing fashion"
    if "decor" in sub:
        keywords += " home decor decoration interior"

    # ✅ Final searchable text
    text = f"""
    {product.get('name','')}
    {product.get('description','')}
    category {product.get('category','')}
    type {product.get('subCategory','')}
    {variant_text}
    {keywords}
    """

    text = text.lower()

    texts.append(text)
    products.append(product)

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ✅ Save embeddings
np.save("embeddings/zatch_embeddings.npy", embeddings)

# ✅ Save cleaned product data
with open("data/zatch_products_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False)

print("✅ Zatch embeddings created successfully!")