"""
IoT-GamiSearch - Mesin Pencari Dokumen Penelitian IoT & Gamification
Metode: Preprocessing -> Indexing -> TF-IDF -> Vector Space Model (Cosine)

Jalankan:  python app.py   lalu buka http://127.0.0.1:5000
"""

import math
import os
import random
import re
import time
from collections import Counter
from datetime import datetime

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from markupsafe import Markup, escape

from engine import build_or_load_index, preprocess_trace, search, similar_documents
from engine.preprocessing import stem_word, TOKEN_RE
from engine.stopwords import STOPWORDS_EN, STOPWORDS_ID

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_MAIN = os.path.join(BASE, "dataset", "iot_gamification_dataset.csv")
DATASET_SAMPLE = os.path.join(BASE, "dataset", "sample_dataset.csv")
TARGET_DOCS = 1000
PER_PAGE = 10
POSTING_PREVIEW = 12  # jumlah posting (doc:tf) per term yang dikirim ke tabel kosakata

app = Flask(__name__)
STATE = {"index": None, "path": None, "loaded_at": None}
RECENT = []  # log riwayat kueri (in-memory): dict q, category, at, total, ms
HISTORY_MAX = 50


# ======================================================================
# LOAD / BUILD INDEX
# ======================================================================
def dataset_path():
    env = os.environ.get("DATASET_PATH")
    if env and os.path.exists(env):
        return env
    return DATASET_MAIN if os.path.exists(DATASET_MAIN) else DATASET_SAMPLE


def load_index(force=False):
    path = dataset_path()
    if force:
        cache = os.path.join(BASE, "cache", "index.pkl")
        if os.path.exists(cache):
            os.remove(cache)
    t0 = time.time()
    STATE["index"] = build_or_load_index(path, os.path.join(BASE, "cache"))
    STATE["path"] = path
    STATE["loaded_at"] = time.time() - t0
    idx = STATE["index"]
    src = "cache" if idx.stats.get("from_cache") else "dibangun"
    print(f"[IoT-GamiSearch] {idx.N} dokumen, {len(idx.df)} term ({src}, {STATE['loaded_at']:.2f}s)")


def IDX():
    if STATE["index"] is None:
        load_index()
    return STATE["index"]


@app.context_processor
def inject_globals():
    idx = IDX()
    return {
        "N_DOCS": idx.N,
        "VOCAB": len(idx.df),
        "TARGET_DOCS": TARGET_DOCS,
        "DATASET_FILE": os.path.basename(STATE["path"] or ""),
        "categories": sorted({d.get("category") or "Lainnya" for d in idx.docs}),
    }


# ======================================================================
# HELPER: highlight & snippet
# ======================================================================
def _word_matches(word, terms):
    w = word.lower()
    if len(w) < 2 or w in STOPWORDS_EN or w in STOPWORDS_ID:
        return False
    return (stem_word(w, "en") in terms) or (stem_word(w, "id") in terms) or (w in terms)


def highlight(text, terms, snippet_words=None):
    """Menandai kata yang stem-nya cocok dengan term kueri (aman dari XSS)."""
    text = str(text or "")
    terms = set(terms or [])
    words = re.split(r"(\s+)", text)
    if snippet_words:
        plain = [w for w in words if not w.isspace()]
        first = next((i for i, w in enumerate(plain)
                      if _word_matches(re.sub(r"[^A-Za-z]", "", w), terms)), 0)
        start = max(0, first - 12)
        chunk = plain[start:start + snippet_words]
        prefix = "… " if start > 0 else ""
        suffix = " …" if start + snippet_words < len(plain) else ""
        words = re.split(r"(\s+)", prefix + " ".join(chunk) + suffix)
    out = []
    for w in words:
        if w.isspace() or not w:
            out.append(w)
            continue
        m = re.match(r"^([^A-Za-z]*)([A-Za-z]+)(.*)$", w)
        if m and terms and _word_matches(m.group(2), terms):
            out.append(f"{escape(m.group(1))}<mark>{escape(m.group(2))}</mark>{escape(m.group(3))}")
        else:
            out.append(str(escape(w)))
    return Markup("".join(out))


app.jinja_env.globals["highlight"] = highlight


@app.template_filter("num")
def fmt_num(v, d=4):
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return v


@app.template_filter("thousands")
def fmt_thousands(v):
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return v


def paginate(total, page, per_page=PER_PAGE):
    pages = max(1, math.ceil(total / per_page))
    page = min(max(1, page), pages)
    lo = max(1, page - 2)
    hi = min(pages, lo + 4)
    lo = max(1, hi - 4)
    return {"page": page, "pages": pages, "total": total,
            "start": (page - 1) * per_page, "end": min(total, page * per_page),
            "window": list(range(lo, hi + 1))}


def _int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ======================================================================
# FITUR 1 - DASHBOARD PENCARIAN
# ======================================================================
SAMPLE_QUERIES = [
    "smart home gamification",
    "sensor iot health",
    "wearable motivation",
    "student leaderboard badge",
    "smart agriculture monitoring",
    "gamifikasi pembelajaran",
]


def log_search(q, category, total, ms):
    """Catat kueri ke log riwayat; kueri identik berturut-turut cukup diperbarui."""
    entry = {"q": q, "category": category or "", "total": total, "ms": ms,
             "at": datetime.now().strftime("%d %b %Y, %H:%M:%S")}
    if RECENT and RECENT[0]["q"].lower() == q.lower() and RECENT[0]["category"] == entry["category"]:
        RECENT[0] = entry
    else:
        RECENT.insert(0, entry)
        del RECENT[HISTORY_MAX:]


@app.route("/")
def index():
    idx = IDX()
    cat = Counter(d.get("category") or "Lainnya" for d in idx.docs)
    top_df = sorted(idx.df.items(), key=lambda x: -x[1])[:12]
    return render_template(
        "index.html", page="home", stats=idx.stats,
        cat_counts=cat.most_common(), top_df=top_df,
        samples=SAMPLE_QUERIES, recent=RECENT[:5],
    )


# ======================================================================
# FITUR 4 - HALAMAN HASIL PENCARIAN
# ======================================================================
@app.route("/search")
def search_page():
    q = (request.args.get("q") or "").strip()
    if not q:
        return redirect(url_for("index"))
    category = request.args.get("category") or None
    y1 = _int(request.args.get("year_from"))
    y2 = _int(request.args.get("year_to"))
    page = _int(request.args.get("page"), 1)

    idx = IDX()
    res = search(idx, q, category, y1, y2)
    pg = paginate(len(res["results"]), page)
    shown = res["results"][pg["start"]:pg["end"]]

    if pg["page"] == 1:
        log_search(q, category, len(res["results"]), res["elapsed_ms"])

    return render_template(
        "result.html", page="search", q=q, res=res, shown=shown, pg=pg,
        category=category or "", year_from=y1 or "", year_to=y2 or "",
        samples=SAMPLE_QUERIES, history=RECENT,
    )


@app.route("/history")
def history_page():
    return render_template(
        "result.html", page="history", q="", res=None, shown=[], pg=None,
        category="", year_from="", year_to="", samples=SAMPLE_QUERIES, history=RECENT,
    )


@app.route("/document/<int:i>")
def document(i):
    idx = IDX()
    if i < 0 or i >= idx.N:
        abort(404)
    doc = idx.docs[i]
    q = (request.args.get("q") or "").strip()
    detail = None
    terms = []
    if q:
        res = search(idx, q)
        terms = res["terms"]
        detail = next((r for r in res["results"] if r["idx"] == i), None)
        detail = {"res": res, "hit": detail}
    trace = preprocess_trace(f"{doc['title']}. {doc['abstract']}", doc.get("language"))
    return render_template(
        "document.html", page="search", doc=doc, i=i, q=q, terms=terms,
        detail=detail, top=idx.top_terms(i, 15), norm=idx.doc_norms[i],
        similar=similar_documents(idx, i, 5), trace=trace,
        n_unique=len(idx.doc_terms[i]),
    )


# ======================================================================
# FITUR 2 - PREPROCESSING
# ======================================================================
@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():
    idx = IDX()
    text = request.form.get("text") if request.method == "POST" else None
    src_doc = None
    if request.args.get("doc") is not None:
        i = _int(request.args.get("doc"), 0)
        if request.args.get("doc") == "random":
            i = random.randrange(idx.N)
        i = min(max(0, i), idx.N - 1)
        src_doc = (i, idx.docs[i])
        text = f"{idx.docs[i]['title']}. {idx.docs[i]['abstract']}"
    if not text:
        text = ("Penerapan Gamifikasi pada Sistem Monitoring IoT untuk Meningkatkan "
                "Motivasi Belajar Mahasiswa! Penelitian ini menggunakan sensor "
                "ESP32 dan leaderboard, poin, serta badge (2024).")
    trace = preprocess_trace(text)
    return render_template(
        "preprocessing.html", page="pre", text=text, t=trace, src_doc=src_doc,
        deltas=preprocess_deltas(trace),
        n_sw_id=len(STOPWORDS_ID), n_sw_en=len(STOPWORDS_EN), stats=idx.stats,
    )


def preprocess_deltas(t):
    """Perubahan data (delta) di setiap tahap preprocessing, untuk ditampilkan di UI."""
    orig, folded, cleaned = t["original"], t["case_folding"], t["cleaning"]
    words = cleaned.split()
    return {
        "case": {"chars": len(orig), "lowered": sum(1 for a, b in zip(orig, folded) if a != b)},
        "clean": {"before": len(folded), "after": len(cleaned), "removed": len(folded) - len(cleaned)},
        "token": {"tokens": len(t["tokens"]), "short": len(words) - len(t["tokens"])},
        "stop": {"before": len(t["tokens"]), "after": len(t["filtered"]),
                 "removed": len(t["stopwords_removed"])},
        "stem": {"terms": len(t["stems"]), "changed": len(t["stem_changed"]),
                 "unique_before": len(set(t["filtered"])), "unique_after": len(set(t["stems"]))},
    }


# ======================================================================
# FITUR 3 - MESIN HITUNG VSM / TF-IDF
# ======================================================================
@app.route("/engine")
def engine_page():
    idx = IDX()
    term_q = (request.args.get("term") or "").strip().lower()
    sort = request.args.get("sort", "df")
    page = _int(request.args.get("page"), 1)

    items = list(idx.df.items())
    if term_q:
        items = [x for x in items if term_q in x[0]]
    if sort == "idf":
        items.sort(key=lambda x: (-idx.idf[x[0]], x[0]))
    elif sort == "term":
        items.sort(key=lambda x: x[0])
    else:
        items.sort(key=lambda x: (-x[1], x[0]))
    pg = paginate(len(items), page, 20)
    vocab_rows = [{"term": t, "df": df, "idf": idx.idf[t],
                   "cf": sum(tf for _, tf in idx.postings[t]),
                   "postings": [(d, tf, idx.docs[d]["id"]) for d, tf in idx.postings[t][:6]]}
                  for t, df in items[pg["start"]:pg["end"]]]

    # Contoh matriks TF-IDF: 8 term ber-df tertinggi (yang idf > 0) x 6 dokumen pertama
    sample_terms = [t for t, _ in sorted(idx.df.items(), key=lambda x: -x[1])
                    if idx.idf[t] > 0][:8]
    sample_docs = list(range(min(6, idx.N)))
    matrix = [[idx.doc_vectors[d].get(t, 0.0) for t in sample_terms] for d in sample_docs]

    df_hist = Counter()
    for df in idx.df.values():
        b = "1" if df == 1 else "2-5" if df <= 5 else "6-20" if df <= 20 else \
            "21-100" if df <= 100 else ">100"
        df_hist[b] += 1
    hist = [(b, df_hist.get(b, 0)) for b in ["1", "2-5", "6-20", "21-100", ">100"]]

    # Seluruh kosakata untuk filter & paginasi sisi-klien (JS) di halaman /engine.
    # Format ringkas per term: [term, df, idf, cf, [[doc_idx, tf], ...maks. POSTING_PREVIEW]]
    vocab_all = [[t, df, round(idx.idf[t], 6), sum(tf for _, tf in idx.postings[t]),
                  [[d, tf] for d, tf in idx.postings[t][:POSTING_PREVIEW]]]
                 for t, df in idx.df.items()]
    doc_ids = [d["id"] for d in idx.docs]

    return render_template(
        "engine.html", page="engine", stats=idx.stats, vocab_rows=vocab_rows,
        pg=pg, term_q=term_q, sort=sort, sample_terms=sample_terms,
        sample_docs=[(d, idx.docs[d]) for d in sample_docs], matrix=matrix,
        hist=hist, max_idf=math.log10(idx.N) if idx.N else 0, norms=[idx.doc_norms[d] for d in sample_docs],
        vocab_all=vocab_all, doc_ids=doc_ids, posting_preview=POSTING_PREVIEW,
        idf_min=min(idx.idf.values(), default=0), idf_max=max(idx.idf.values(), default=0),
    )


# ======================================================================
# DATASET
# ======================================================================
@app.route("/dataset")
def dataset_page():
    idx = IDX()
    category = request.args.get("category") or ""
    lang = request.args.get("lang") or ""
    kw = (request.args.get("kw") or "").strip().lower()
    page = _int(request.args.get("page"), 1)

    rows = [(i, d) for i, d in enumerate(idx.docs)
            if (not category or d.get("category") == category)
            and (not lang or d.get("language") == lang)
            and (not kw or kw in d["title"].lower())]
    pg = paginate(len(rows), page, 15)

    years = Counter()
    for d in idx.docs:
        y = str(d.get("year") or "")[:4]
        if y.isdigit():
            years[int(y)] += 1
    year_rows = sorted(years.items())
    cat = Counter(d.get("category") or "Lainnya" for d in idx.docs).most_common()
    avg_words = sum(len(str(d["abstract"]).split()) for d in idx.docs) / max(1, idx.N)

    return render_template(
        "dataset.html", page="dataset", rows=rows[pg["start"]:pg["end"]], pg=pg,
        category=category, lang=lang, kw=kw, stats=idx.stats, year_rows=year_rows,
        cat_counts=cat, avg_words=avg_words, path=STATE["path"],
    )


@app.route("/reindex", methods=["POST"])
def reindex():
    load_index(force=True)
    return redirect(url_for("dataset_page"))


# ======================================================================
# API JSON
# ======================================================================
@app.route("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    k = min(_int(request.args.get("k"), 10) or 10, 100)
    res = search(IDX(), q)
    return jsonify({
        "query": q, "terms": res["terms"], "total": len(res["results"]),
        "elapsed_ms": round(res["elapsed_ms"], 3),
        "results": [{"rank": r["rank"], "id": r["doc"]["id"], "title": r["doc"]["title"],
                     "year": r["doc"].get("year"), "score": round(r["score"], 6)}
                    for r in res["results"][:k]],
    })


@app.errorhandler(404)
def not_found(_):
    return render_template("404.html", page=""), 404


# Muat indeks secara eager agar Vercel Serverless siap saat dipanggil
try:
    load_index()
except Exception as e:
    print(f"[Warning] Gagal memuat indeks awal: {e}")

# Handler WSGI untuk Vercel
app = app

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)