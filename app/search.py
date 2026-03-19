import numpy as np
from rapidfuzz import process
from app.model_loader import model, product_embeddings, products

# =========================
# 🔥 BUILD VOCABULARY
# =========================
vocabulary = set()

for p in products:
    name = p.get("name", "").lower().split()
    sub = p.get("subCategory", "").lower().split()

    vocabulary.update(name)
    vocabulary.update(sub)

vocabulary = list(vocabulary)


# =========================
# 🔥 SPELL CORRECTION
# =========================
def correct_query(query):
    words = query.lower().split()
    corrected = []

    for word in words:
        result = process.extractOne(word, vocabulary)

        if result:
            match, score, _ = result
            if score > 70:
                corrected.append(match)
            else:
                corrected.append(word)
        else:
            corrected.append(word)

    return " ".join(corrected)


# =========================
# 🔍 CATEGORY DETECTION
# =========================
def detect_category(query):
    query = query.lower()

    if any(w in query for w in ["shirt", "tshirt", "pant", "jeans", "clothing", "wear"]):
        return "clothing"

    elif any(w in query for w in ["washing", "machine", "refrigerator", "fridge"]):
        return "electronics"

    elif any(w in query for w in ["clock", "decor", "frame", "buddha", "home"]):
        return "home"

    return None


# =========================
# 🚀 MAIN SEARCH FUNCTION
# =========================
def search_products(query, top_k=5):

    # 🚨 Empty query
    if not query.strip():
        return {"success": False, "message": "Empty query"}

    # 🔥 Step 1: Spell correction
    query = correct_query(query)

    # 🔥 Step 2: Embedding
    query_embedding = model.encode([query], normalize_embeddings=True)

    similarities = np.dot(product_embeddings, query_embedding.T).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    # 🔥 Step 3: Category detection
    query_category = detect_category(query)

    for i in top_indices:
        product = products[i]
        score = similarities[i]

        name = product.get("name", "").lower()
        sub = product.get("subCategory", "").lower()

        # =========================
        # 🔥 CATEGORY FILTERING
        # =========================

        if query_category == "clothing":
            if not any(x in name + sub for x in ["shirt", "pant", "wear"]):
                continue

        elif query_category == "electronics":

            if "refrigerator" in query or "fridge" in query:
                if "refrigerator" not in sub:
                    continue

            elif "washing" in query or "machine" in query:
                if "washing" not in sub:
                    continue

        elif query_category == "home":
            if not any(x in sub for x in ["decor", "frame", "clock"]):
                continue

        # =========================

        results.append({
            "score": float(score),
            "product": product
        })

    # 🚨 No results
    if not results:
        return {"success": False, "message": "No products found"}

    # 🔥 Confidence check
    if results[0]["score"] < 0.35:
        return {"success": False, "message": "No products found"}

    # 🔥 Sort
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    final_results = [r["product"] for r in results]

    return {"success": True, "results": final_results}