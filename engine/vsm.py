"""
Fitur 3 - Mesin Hitung VSM & Perangkingan
=========================================

Kueri diperlakukan sebagai dokumen kecil lalu diubah menjadi vektor TF-IDF:
    w(t,q) = tf(t,q) x idf(t)

Kemiripan kueri-dokumen dihitung dengan Cosine Similarity:

                      sum_t  w(t,q) x w(t,d)
    cos(q,d) = ----------------------------------
                 sqrt(sum w(t,q)^2) x sqrt(sum w(t,d)^2)

Perhitungan memakai inverted index: hanya dokumen yang memuat minimal satu
term kueri yang dihitung (dokumen lain pasti bernilai 0).
Dokumen diurutkan dari skor cosine tertinggi ke terendah.
"""

import math
import time
from collections import Counter

from .preprocessing import query_terms


def build_query_vector(index, terms):
    tf = Counter(terms)
    rows, vec = [], {}
    for t, f in tf.items():
        df = index.df.get(t, 0)
        idf = index.idf.get(t, 0.0)
        w = f * idf
        rows.append({"term": t, "tf": f, "df": df, "idf": idf, "w": w,
                     "in_index": df > 0})
        if w > 0:
            vec[t] = w
    norm = math.sqrt(sum(w * w for w in vec.values()))
    return vec, norm, rows


def search(index, query, category=None, year_from=None, year_to=None):
    t0 = time.perf_counter()
    terms, trace = query_terms(query, index.df)
    qvec, qnorm, qrows = build_query_vector(index, terms)

    # Akumulasi dot product lewat inverted index
    dot, contrib = {}, {}
    for t, wq in qvec.items():
        for d, tf in index.postings.get(t, []):
            wd = index.doc_vectors[d][t]
            dot[d] = dot.get(d, 0.0) + wq * wd
            contrib.setdefault(d, []).append(
                {"term": t, "tf": tf, "wq": wq, "wd": wd, "prod": wq * wd})

    results = []
    for d, num in dot.items():
        doc = index.docs[d]
        if category and doc.get("category") != category:
            continue
        y = _year(doc.get("year"))
        if year_from and (y is None or y < year_from):
            continue
        if year_to and (y is None or y > year_to):
            continue
        dn = index.doc_norms[d]
        score = num / (qnorm * dn) if qnorm and dn else 0.0
        if score <= 0:
            continue
        results.append({
            "idx": d, "doc": doc, "score": score, "dot": num, "doc_norm": dn,
            "contrib": sorted(contrib[d], key=lambda c: c["prod"], reverse=True),
            "matched": len(contrib[d]),
        })

    results.sort(key=lambda r: (-r["score"], r["idx"]))
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return {
        "query": query,
        "terms": terms,
        "trace": trace,
        "query_rows": qrows,
        "query_norm": qnorm,
        "results": results,
        "candidates": len(dot),
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


def similar_documents(index, doc_idx, k=5):
    """Dokumen lain yang paling mirip (cosine antar-dokumen)."""
    vec = index.doc_vectors[doc_idx]
    n1 = index.doc_norms[doc_idx]
    dot = {}
    for t, w in vec.items():
        if w == 0:
            continue
        for d, _ in index.postings[t]:
            if d != doc_idx:
                dot[d] = dot.get(d, 0.0) + w * index.doc_vectors[d][t]
    scored = [(d, v / (n1 * index.doc_norms[d]))
              for d, v in dot.items() if n1 and index.doc_norms[d]]
    scored.sort(key=lambda x: -x[1])
    return [{"idx": d, "doc": index.docs[d], "score": s} for d, s in scored[:k]]


def _year(v):
    try:
        return int(str(v)[:4])
    except (TypeError, ValueError):
        return None


# Kompatibilitas dengan kode lama
def ranking(index, query, top_k=10):
    return search(index, query)["results"][:top_k]
