"""
Pengumpul Dataset IoT-GamiSearch
================================

Mengunduh 1.000 dokumen penelitian NYATA (judul + abstrak + metadata) bertopik
Internet of Things (IoT) & Gamification dari OpenAlex (https://openalex.org),
basis data publikasi ilmiah terbuka (gratis, tanpa login).

Cara pakai (dari folder project):

    python scripts/fetch_dataset.py
    python scripts/fetch_dataset.py --total 1000 --email emailanda@gmail.com

Hasil: dataset/iot_gamification_dataset.csv
Kolom : id, title, abstract, keyword, category, year, authors, venue,
        language, doi, url, source_id
"""

import argparse
import csv
import html
import os
import re
import sys
import time

import requests

API = "https://api.openalex.org/works"

# ---------------------------------------------------------------------------
# Kueri pengumpulan. Setiap kueri punya kategori & kuota. Kuota totalnya > 1000
# karena sebagian hasil akan terbuang (duplikat / abstrak terlalu pendek).
# ---------------------------------------------------------------------------
QUERIES = [
    # (kategori, kueri pencarian, filter tambahan, kuota)
    ("IoT-Gamification", "gamification internet of things", "", 260),
    ("IoT-Gamification", "gamified IoT smart", "", 120),
    ("IoT-Gamification", "gamifikasi internet of things", "", 60),
    ("Gamification", "gamification learning motivation", "", 200),
    ("Gamification", "gamification education students", "", 120),
    ("Gamification", "gamifikasi pembelajaran", "language:id", 120),
    ("IoT", "internet of things education learning", "", 160),
    ("IoT", "internet of things sensor monitoring", "", 140),
    ("IoT", "internet of things monitoring sensor", "language:id", 140),
    ("Smart System", "smart home smart city IoT energy", "", 100),
]

# Kueri cadangan bila jumlah dokumen masih kurang dari target
TOPUP_QUERIES = [
    ("IoT-Gamification", "gamification sensor"),
    ("Gamification", "gamification"),
    ("IoT", "internet of things"),
]

ALLOWED_LANG = {"en", "id"}
MIN_WORDS = 40          # abstrak minimal 40 kata
MAX_ABSTRACT_WORDS = 450


def rebuild_abstract(inv):
    """OpenAlex menyimpan abstrak sebagai inverted index {kata: [posisi,...]}."""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    text = " ".join(pos[i] for i in sorted(pos))
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)          # buang tag HTML/JATS
    text = re.sub(r"^\s*(abstract|abstrak)[\s:.-]*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean(s):
    s = html.unescape(str(s or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch_query(session, query, extra_filter, quota, email, api_key):
    flt = "has_abstract:true"
    if extra_filter:
        flt += "," + extra_filter
    params = {
        "search": query,
        "filter": flt,
        "per_page": 200,
        "cursor": "*",
        "select": "id,doi,display_name,publication_year,abstract_inverted_index,"
                  "authorships,primary_location,language,keywords",
    }
    if email:
        params["mailto"] = email
    if api_key:
        params["api_key"] = api_key

    got = []
    while len(got) < quota and params["cursor"]:
        for attempt in range(4):
            try:
                r = session.get(API, params=params, timeout=60)
                if r.status_code == 400 and "select" in params:
                    # fallback bila parameter select tidak didukung
                    params.pop("select")
                    continue
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"   ! gagal ({e}), mencoba lagi...")
                time.sleep(2 * (attempt + 1))
        else:
            print("   ! menyerah untuk kueri ini")
            break

        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        got.extend(results)
        params["cursor"] = data.get("meta", {}).get("next_cursor")
        time.sleep(0.25)  # sopan terhadap API
    return got[: quota * 2]  # ambil lebih, karena sebagian akan disaring


def to_row(w, category):
    title = clean(w.get("display_name") or w.get("title"))
    abstract = rebuild_abstract(w.get("abstract_inverted_index"))
    lang = (w.get("language") or "").lower()
    words = abstract.split()
    if not title or len(words) < MIN_WORDS or lang not in ALLOWED_LANG:
        return None
    if len(words) > MAX_ABSTRACT_WORDS:
        abstract = " ".join(words[:MAX_ABSTRACT_WORDS]) + " ..."

    authors = [clean(a.get("author", {}).get("display_name"))
               for a in (w.get("authorships") or [])]
    authors = [a for a in authors if a]
    authors_s = ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else "")

    loc = w.get("primary_location") or {}
    src = (loc.get("source") or {}) if isinstance(loc, dict) else {}
    venue = clean(src.get("display_name"))
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    url = w.get("doi") or (loc.get("landing_page_url") if isinstance(loc, dict) else "") or w.get("id")
    kws = [clean(k.get("display_name")) for k in (w.get("keywords") or [])]
    return {
        "title": title,
        "abstract": abstract,
        "keyword": "; ".join([k for k in kws if k][:6]).lower(),
        "category": category,
        "year": w.get("publication_year") or "",
        "authors": authors_s,
        "venue": venue,
        "language": lang,
        "doi": doi,
        "url": url,
        "source_id": (w.get("id") or "").replace("https://openalex.org/", ""),
    }


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--email", default=os.environ.get("OPENALEX_EMAIL", ""))
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY", ""))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset", "iot_gamification_dataset.csv"))
    args = ap.parse_args()

    s = requests.Session()
    s.headers["User-Agent"] = "IoT-GamiSearch/1.0 (student project)"

    rows, seen_ids, seen_titles = [], set(), set()
    per_cat = {}
    for category, query, extra, quota in QUERIES:
        print(f"-> [{category}] '{query}' {extra}")
        works = fetch_query(s, query, extra, quota, args.email, args.api_key)
        added = 0
        for w in works:
            if added >= quota:
                break
            row = to_row(w, category)
            if not row:
                continue
            nt = norm_title(row["title"])
            if row["source_id"] in seen_ids or nt in seen_titles:
                continue
            seen_ids.add(row["source_id"])
            seen_titles.add(nt)
            rows.append(row)
            added += 1
        per_cat[category] = per_cat.get(category, 0) + added
        print(f"   +{added} dokumen (total {len(rows)})")

    # Tahap tambahan: bila masih kurang dari target, ambil dari kueri umum
    for category, query in TOPUP_QUERIES:
        if len(rows) >= args.total:
            break
        need = args.total - len(rows)
        print(f"-> [top-up {category}] '{query}' (kurang {need})")
        for w in fetch_query(s, query, "", need * 2, args.email, args.api_key):
            if len(rows) >= args.total:
                break
            row = to_row(w, category)
            if not row:
                continue
            nt = norm_title(row["title"])
            if row["source_id"] in seen_ids or nt in seen_titles:
                continue
            seen_ids.add(row["source_id"])
            seen_titles.add(nt)
            rows.append(row)
            per_cat[category] = per_cat.get(category, 0) + 1
        print(f"   total {len(rows)}")

    if len(rows) < args.total:
        print(f"\n! Hanya terkumpul {len(rows)} dokumen (< {args.total}). "
              "Tambah kuota/kueri di QUERIES lalu jalankan lagi.")
    rows = rows[: args.total]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["id", "title", "abstract", "keyword", "category", "year", "authors",
              "venue", "language", "doi", "url", "source_id"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for i, r in enumerate(rows, 1):
            wr.writerow({"id": i, **r})

    print(f"\nSelesai: {len(rows)} dokumen -> {args.out}")
    langs = {}
    for r in rows:
        langs[r["language"]] = langs.get(r["language"], 0) + 1
    print("Bahasa  :", langs)
    print("Kategori:", per_cat)
    return 0 if len(rows) >= args.total else 1


if __name__ == "__main__":
    sys.exit(main())
