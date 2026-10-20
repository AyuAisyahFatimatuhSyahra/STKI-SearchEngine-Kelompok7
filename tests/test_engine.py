"""
Uji kebenaran perhitungan TF-IDF & VSM pada korpus kecil yang bisa dihitung
manual. Jalankan:  python -m unittest discover tests -v
"""
import math
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import SearchIndex, search, preprocess_trace  # noqa: E402

DOCS = pd.DataFrame([
    {"id": "1", "title": "sensor iot", "abstract": "sensor sensor monitoring", "keyword": "", "category": "IoT", "language": "en"},
    {"id": "2", "title": "game badge", "abstract": "badge leaderboard game", "keyword": "", "category": "Gamification", "language": "en"},
    {"id": "3", "title": "iot game", "abstract": "monitoring game", "keyword": "", "category": "IoT-Gamification", "language": "en"},
])


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx = SearchIndex().build(DOCS.copy())

    def test_df_idf(self):
        idx = self.idx
        self.assertEqual(idx.N, 3)
        self.assertEqual(idx.df["sensor"], 1)
        self.assertEqual(idx.df["game"], 2)
        self.assertAlmostEqual(idx.idf["sensor"], math.log10(3 / 1))
        self.assertAlmostEqual(idx.idf["game"], math.log10(3 / 2))

    def test_tf_weight(self):
        # D1 = "sensor iot sensor sensor monitor" -> tf(sensor)=3
        idx = self.idx
        self.assertEqual(idx.doc_terms[0]["sensor"], 3)
        self.assertAlmostEqual(idx.doc_vectors[0]["sensor"], 3 * math.log10(3))

    def test_cosine_manual(self):
        idx = self.idx
        res = search(idx, "sensor monitoring")
        # Vektor kueri
        wq = {"sensor": math.log10(3), "monitor": math.log10(1.5)}
        qn = math.sqrt(sum(v * v for v in wq.values()))
        # Cosine manual untuk D1
        d = idx.doc_vectors[0]
        dn = math.sqrt(sum(v * v for v in d.values()))
        expected = sum(wq[t] * d.get(t, 0) for t in wq) / (qn * dn)
        top = res["results"][0]
        self.assertEqual(top["idx"], 0)
        self.assertAlmostEqual(top["score"], expected, places=10)
        # Ranking menurun & dokumen tanpa term kueri tidak muncul
        scores = [r["score"] for r in res["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertNotIn(1, [r["idx"] for r in res["results"]])

    def test_unknown_query(self):
        res = search(self.idx, "zzzz qqqq")
        self.assertEqual(res["results"], [])

    def test_preprocessing_id(self):
        t = preprocess_trace("Penerapan gamifikasi untuk meningkatkan pembelajaran di sekolah")
        self.assertEqual(t["language"], "id")
        self.assertNotIn("untuk", t["filtered"])
        self.assertIn("ajar", t["stems"])        # pembelajaran -> ajar (Sastrawi)

    def test_preprocessing_en(self):
        t = preprocess_trace("The students are learning with gamified sensors, 2024!")
        self.assertEqual(t["language"], "en")
        self.assertNotIn("the", t["filtered"])
        self.assertIn("sensor", t["stems"])
        self.assertIn("learn", t["stems"])

    def test_against_numpy(self):
        """Bandingkan dengan perhitungan matriks numpy penuh."""
        import numpy as np
        idx = self.idx
        vocab = sorted(idx.df)
        M = np.array([[idx.doc_terms[i].get(t, 0) for t in vocab] for i in range(idx.N)], float)
        idf = np.log10(idx.N / (M > 0).sum(0))
        W = M * idf
        q = np.array([1.0 if t in ("game", "badg") else 0.0 for t in vocab]) * idf
        cos = W @ q / (np.linalg.norm(W, axis=1) * np.linalg.norm(q))
        res = {r["idx"]: r["score"] for r in search(idx, "game badge")["results"]}
        for i in range(idx.N):
            self.assertAlmostEqual(res.get(i, 0.0), cos[i], places=10)


if __name__ == "__main__":
    unittest.main()
