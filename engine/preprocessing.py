"""
Fitur 2 - Sistem Pra-proses Teks (Preprocessing Pipeline)
=========================================================

Tahapan:
  1. Case folding    : semua huruf menjadi huruf kecil
  2. Cleaning        : buang angka, tanda baca, simbol, URL
  3. Tokenizing      : memecah teks menjadi token (kata)
  4. Stopword removal: membuang kata umum (ID + EN)
  5. Stemming        : mengubah kata ke bentuk dasar
                       - Bahasa Indonesia -> Sastrawi (Nazief-Adriani)
                       - Bahasa Inggris   -> Porter Stemmer (NLTK)

Dataset bersifat dwibahasa (abstrak Indonesia & Inggris), sehingga bahasa
setiap dokumen dideteksi otomatis berdasarkan jumlah stopword yang muncul.
"""

import re
import warnings
from functools import lru_cache

warnings.filterwarnings("ignore", category=DeprecationWarning, module="Sastrawi")

from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from .stopwords import STOPWORDS, STOPWORDS_EN, STOPWORDS_ID

_sastrawi = StemmerFactory().create_stemmer()
_porter = PorterStemmer()

TOKEN_RE = re.compile(r"[a-z]+")
MIN_TOKEN_LEN = 2


# ---------------------------------------------------------------------------
# Tahap-tahap
# ---------------------------------------------------------------------------
def case_folding(text):
    return str(text).lower()


def cleaning(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"[^a-z\s]", " ", text)                 # angka & simbol
    return re.sub(r"\s+", " ", text).strip()


def tokenizing(text):
    return [t for t in TOKEN_RE.findall(text) if len(t) >= MIN_TOKEN_LEN]


def stopword_removal(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def detect_language(tokens):
    """'id' atau 'en' berdasar jumlah stopword masing-masing bahasa."""
    id_hits = sum(1 for t in tokens if t in STOPWORDS_ID)
    en_hits = sum(1 for t in tokens if t in STOPWORDS_EN)
    if id_hits == 0 and en_hits == 0:
        return None
    return "id" if id_hits > en_hits else "en"


@lru_cache(maxsize=200_000)
def stem_word(word, lang):
    if lang == "id":
        return _sastrawi.stem(word) or word
    return _porter.stem(word)


def stemming(tokens, lang):
    return [stem_word(t, lang or "en") for t in tokens]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def preprocess_tokens(text, lang=None):
    """Mengembalikan (list term hasil stemming, bahasa)."""
    tokens = tokenizing(cleaning(case_folding(text)))
    lang = lang or detect_language(tokens) or "en"
    return stemming(stopword_removal(tokens), lang), lang


def preprocess(text, lang=None):
    """Versi string (kompatibel dengan kode lama)."""
    return " ".join(preprocess_tokens(text, lang)[0])


def preprocess_trace(text, lang=None):
    """Menjalankan pipeline sambil mencatat hasil setiap tahap (untuk UI)."""
    folded = case_folding(text)
    cleaned = cleaning(folded)
    tokens = tokenizing(cleaned)
    detected = detect_language(tokens)
    use_lang = lang or detected or "en"
    removed = [t for t in tokens if t in STOPWORDS]
    filtered = stopword_removal(tokens)
    stems = stemming(filtered, use_lang)
    pairs = [(w, s) for w, s in zip(filtered, stems)]
    changed = [(w, s) for w, s in pairs if w != s]
    return {
        "original": text,
        "case_folding": folded,
        "cleaning": cleaned,
        "tokens": tokens,
        "stopwords_removed": removed,
        "filtered": filtered,
        "stems": stems,
        "stem_pairs": pairs,
        "stem_changed": changed,
        "language": use_lang,
        "language_detected": detected,
        "stemmer": "Sastrawi (Nazief-Adriani)" if use_lang == "id" else "Porter Stemmer (NLTK)",
    }


def query_terms(query, vocabulary):
    """
    Pra-proses kueri. Kueri biasanya pendek sehingga bahasanya sulit dideteksi;
    karena itu untuk setiap token dicoba stem Porter (EN) dan Sastrawi (ID),
    lalu dipilih bentuk yang ada di kosakata indeks (df terbesar).
    Mengembalikan (list term, trace).
    """
    trace = preprocess_trace(query)
    terms, mapping = [], []
    for tok in trace["filtered"]:
        cands = []
        for lang in ("en", "id"):
            s = stem_word(tok, lang)
            if s not in cands:
                cands.append(s)
        if tok not in cands:
            cands.append(tok)
        in_vocab = [c for c in cands if c in vocabulary]
        if in_vocab:
            chosen = max(in_vocab, key=lambda c: vocabulary[c])
        else:
            chosen = cands[0]
        terms.append(chosen)
        mapping.append({"token": tok, "candidates": cands, "term": chosen,
                        "found": chosen in vocabulary})
    trace["stems"] = terms
    trace["query_mapping"] = mapping
    trace["stemmer"] = "Porter (EN) / Sastrawi (ID) - dipilih yang ada di indeks"
    return terms, trace
