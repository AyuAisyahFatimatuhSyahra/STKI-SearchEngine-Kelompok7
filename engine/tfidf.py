"""
Indexing & Pembobotan Kata (TF-IDF)
===================================

Untuk N dokumen:
    tf(t,d)  = frekuensi term t di dokumen d
    df(t)    = jumlah dokumen yang mengandung t
    idf(t)   = log10( N / df(t) )
    w(t,d)   = tf(t,d) x idf(t)
    |d|      = sqrt( sum_t w(t,d)^2 )         (panjang vektor dokumen)

Struktur yang dibangun:
    - inverted index : term -> [(doc_index, tf), ...]
    - vektor TF-IDF  : per dokumen {term: bobot}
Hasil indexing disimpan (cache) di folder cache/ agar aplikasi cepat dinyalakan
ulang; cache otomatis dibangun ulang bila isi dataset berubah.
"""

import hashlib
import math
import os
import pickle
import time
from collections import Counter

import pandas as pd

from .preprocessing import preprocess_tokens, tokenizing, cleaning, case_folding

INDEX_VERSION = 3


class SearchIndex:
    def __init__(self):
        self.docs = []            # metadata dokumen (dict)
        self.doc_terms = []       # Counter term per dokumen
        self.doc_lang = []
        self.postings = {}        # term -> list[(doc_idx, tf)]
        self.df = {}
        self.idf = {}
        self.doc_vectors = []     # list[dict term -> w]
        self.doc_norms = []
        self.N = 0
        self.stats = {}

    # ------------------------------------------------------------------
    def build(self, df_docs):
        t0 = time.time()
        self.docs = df_docs.to_dict("records")
        self.N = len(self.docs)

        raw_tokens = 0
        for d in self.docs:
            text = f"{d['title']}. {d['abstract']} {d.get('keyword', '')}"
            raw_tokens += len(tokenizing(cleaning(case_folding(text))))
            terms, lang = preprocess_tokens(text, d.get("language") or None)
            self.doc_terms.append(Counter(terms))
            self.doc_lang.append(lang)
            d["language"] = lang
            d["n_terms"] = len(terms)

        # --- Inverted index & document frequency --------------------------
        postings = {}
        for i, counts in enumerate(self.doc_terms):
            for term, tf in counts.items():
                postings.setdefault(term, []).append((i, tf))
        self.postings = postings
        self.df = {t: len(p) for t, p in postings.items()}

        # --- IDF -----------------------------------------------------------
        self.idf = {t: math.log10(self.N / df) for t, df in self.df.items()}

        # --- Bobot TF-IDF & panjang vektor ---------------------------------
        for counts in self.doc_terms:
            vec = {t: tf * self.idf[t] for t, tf in counts.items()}
            self.doc_vectors.append(vec)
            self.doc_norms.append(math.sqrt(sum(w * w for w in vec.values())))

        total_terms = sum(sum(c.values()) for c in self.doc_terms)
        nnz = sum(len(c) for c in self.doc_terms)
        V = len(self.df)
        self.stats = {
            "n_docs": self.N,
            "vocab_size": V,
            "raw_tokens": raw_tokens,
            "total_terms": total_terms,
            "avg_terms": total_terms / self.N if self.N else 0,
            "nnz": nnz,
            "sparsity": 1 - nnz / (self.N * V) if self.N and V else 0,
            "build_seconds": time.time() - t0,
            "lang_counts": dict(Counter(self.doc_lang)),
        }
        return self

    # ------------------------------------------------------------------
    def top_terms(self, doc_idx, k=10):
        vec = self.doc_vectors[doc_idx]
        items = sorted(vec.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{"term": t, "tf": self.doc_terms[doc_idx][t], "df": self.df[t],
                 "idf": self.idf[t], "w": w} for t, w in items]


# ----------------------------------------------------------------------
# Memuat dataset
# ----------------------------------------------------------------------
REQUIRED = ["title", "abstract"]
OPTIONAL = {"keyword": "", "category": "Lainnya", "year": "", "authors": "",
            "venue": "", "language": "", "doi": "", "url": "", "source_id": ""}


def load_dataset(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in REQUIRED:
        if c not in df.columns:
            raise ValueError(f"Kolom wajib '{c}' tidak ada di {path}")
    for c, default in OPTIONAL.items():
        if c not in df.columns:
            df[c] = default
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    df = df[(df["title"].str.strip() != "") & (df["abstract"].str.strip() != "")]
    df = df.reset_index(drop=True)
    df["id"] = df["id"].astype(str)
    df["year"] = df["year"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def _file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{h.hexdigest()}-v{INDEX_VERSION}"


def build_or_load_index(path, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    key = _file_hash(path)
    cache_file = os.path.join(cache_dir, "index.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                saved_key, index = pickle.load(f)
            if saved_key == key:
                index.stats["from_cache"] = True
                return index
        except Exception:
            pass
    index = SearchIndex().build(load_dataset(path))
    index.stats["from_cache"] = False
    with open(cache_file, "wb") as f:
        pickle.dump((key, index), f)
    return index
