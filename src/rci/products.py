"""Product categories from sentence embeddings (the data has no category column).

Idea: a sentence-embedding model (MiniLM, the same one as the ATS project) turns each product
description into a 384-number vector so that descriptions with similar meaning point in similar
directions ("RED RETROSPOT MUG" lands near "BLUE POLKADOT TEA CUP", far from "SILVER NECKLACE").
KMeans then groups nearby vectors, and each group is named by its most characteristic words.

This is OFFLINE work (needs fastembed). The app only reads the saved results.

Run:  python -m rci.products
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from rci import config, db

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_GRID = (8, 10, 12, 14, 16)

# Colour, size and filler words are removed before embedding so that clusters reflect WHAT the item
# is (mug, candle, necklace), not its colour or size.
NOISE_WORDS = set(
    """red blue green pink white black yellow orange purple silver gold golden ivory cream grey gray
    brown turquoise multicolour multicolor multi colour colours color assorted asstd ass mixed small
    medium large big mini jumbo set of pack the and in with for to a on by from design designs style
    """.split()
)


def clean_description(desc: str) -> str:
    tokens = [t for t in re.findall(r"[a-z]+", desc.lower()) if len(t) > 1 and t not in NOISE_WORDS]
    return " ".join(tokens) if tokens else desc.lower()


def embed(texts: list[str]) -> np.ndarray:
    """Unit-length MiniLM vectors, shape (n, 384). With unit vectors, dot product = cosine similarity."""
    from fastembed import TextEmbedding

    model = TextEmbedding(MODEL_NAME)
    E = np.asarray(list(model.embed(texts)), dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def cluster(E: np.ndarray, k_grid=K_GRID, seed: int = config.SEED):
    """KMeans for each k in the grid; keep the k with the best silhouette."""
    scores, fits = {}, {}
    for k in k_grid:
        km = KMeans(k, n_init=10, random_state=seed).fit(E)
        scores[k] = float(silhouette_score(E, km.labels_))
        fits[k] = km
    best = max(scores, key=scores.get)
    return fits[best], best, scores


def name_clusters(texts: list[str], labels: np.ndarray, n_words: int = 3) -> dict[int, str]:
    """Class-based TF-IDF: treat each cluster as ONE document, then ask which words are most
    characteristic of it compared with the other clusters. Plural/singular duplicates are merged."""
    k = int(labels.max()) + 1
    docs = [" ".join(t for t, lab in zip(texts, labels) if lab == c) for c in range(k)]
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, token_pattern=r"[a-z]{3,}")
    X = tfidf.fit_transform(docs)
    vocab = np.array(tfidf.get_feature_names_out())
    names: dict[int, str] = {}
    for c in range(k):
        chosen: list[str] = []
        for w in vocab[np.argsort(-X[c].toarray()[0])[:12]]:
            if any(w == o + "s" or o == w + "s" for o in chosen):
                continue
            chosen.append(w)
            if len(chosen) == n_words:
                break
        name = " / ".join(w.title() for w in chosen)
        names[c] = name if name not in names.values() else f"{name} ({c})"
    return names


def main() -> None:
    con = db.connect()
    products = con.execute("SELECT stock_code, description FROM products ORDER BY stock_code").df()
    texts = [clean_description(d) for d in products["description"]]
    print(f"Embedding {len(texts):,} product descriptions with {MODEL_NAME} ...")
    E = embed(texts)

    km, k, scores = cluster(E)
    labels = km.labels_
    names = name_clusters(texts, labels)
    print("silhouette by k:", {kk: round(s, 4) for kk, s in scores.items()}, "-> chosen", k)

    out = products[["stock_code"]].assign(
        category_id=labels.astype("int32"), category=[names[i] for i in labels]
    )
    out.to_parquet(config.PROCESSED_DIR / "product_categories.parquet", index=False, compression="zstd")
    np.save(config.PROCESSED_DIR / "product_embeddings.npy", E.astype(np.float16))

    # most typical products = nearest to the cluster centre (dot product on unit vectors)
    cats = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        centre = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
        nearest = idx[np.argsort(-(E[idx] @ centre))[:6]]
        cats.append({
            "id": c, "name": names[c], "n_products": int(len(idx)),
            "typical_products": products["description"].iloc[nearest].tolist(),
        })
    report = {
        "model": MODEL_NAME, "chosen_k": int(k),
        "silhouette_by_k": {str(kk): round(s, 4) for kk, s in scores.items()},
        "categories": sorted(cats, key=lambda x: -x["n_products"]),
        "notes": (
            "Silhouette is low (< 0.06) because product descriptions are short and overlap. It keeps "
            "rising with k, so the grid stops at 16 to keep categories readable. Categories are fuzzy "
            "groups, not an official catalogue."
        ),
    }
    (config.PROCESSED_DIR / "categories.json").write_text(json.dumps(report, indent=2))
    for c in report["categories"]:
        print(f"{c['n_products']:>5}  {c['name']:<38} {', '.join(p[:24] for p in c['typical_products'][:3])}")


if __name__ == "__main__":
    main()
