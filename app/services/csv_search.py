from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from services.store import extract_keywords_with_vocab, normalize_text, tokenize


@dataclass
class CsvSearchResult:
    row: Dict[str, str]
    score: int
    matched: List[str]
    matched_count: int
    expected: int


class CsvSearchStore:
    def __init__(
        self,
        path: str,
        headers: List[str],
        category_map: Dict[str, List[str]],
        logger=None,
    ) -> None:
        self.path = path
        self.headers = headers
        self.category_map = category_map
        self.logger = logger

    def ensure_file(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers, delimiter=";")
                writer.writeheader()

    def read_rows(self) -> List[Dict[str, str]]:
        self.ensure_file()
        rows: List[Dict[str, str]] = []
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                rows.append({k: (v or "").strip() for k, v in row.items()})
        return rows

    def append_row(self, row: Dict[str, str]) -> None:
        self.ensure_file()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers, delimiter=";")
            writer.writerow({k: row.get(k, "") for k in self.headers})

    @staticmethod
    def _pairs_for_value(value: str) -> List[Tuple[str, str]]:
        if not value:
            return []
        raw, _, pairs = extract_keywords_with_vocab(value, None)
        return pairs if pairs else [(v, normalize_text(v)) for v in raw]

    def build_pairs_by_cat(
        self,
        query_by_cat: Dict[str, str],
    ) -> Tuple[Dict[str, List[Tuple[str, str]]], List[Tuple[str, str]]]:
        pairs_by_cat: Dict[str, List[Tuple[str, str]]] = {}
        keywords_display: List[Tuple[str, str]] = []
        for cat, value in query_by_cat.items():
            if not value:
                pairs_by_cat[cat] = []
                continue
            pairs = self._pairs_for_value(value)
            pairs_by_cat[cat] = pairs
            keywords_display.extend([(cat, p[0]) for p in pairs])
        return pairs_by_cat, keywords_display

    def score_row(
        self,
        row: Dict[str, str],
        pairs_by_cat: Dict[str, List[Tuple[str, str]]],
    ) -> CsvSearchResult | None:
        matched: List[str] = []
        score = 0
        expected = 0

        for cat, pairs in pairs_by_cat.items():
            if not pairs:
                continue
            expected += len(pairs)
            fields = self.category_map.get(cat, [])
            text = " ".join(row.get(field, "") for field in fields if row.get(field))
            if not text:
                continue
            norm_text = normalize_text(text)
            counter = Counter(tokenize(norm_text))
            for display, kw in pairs:
                kw_norm = normalize_text(kw)
                hit = False
                if " " in kw_norm:
                    if kw_norm in norm_text:
                        hit = True
                        score += len(tokenize(kw_norm)) or 1
                else:
                    count = counter.get(kw_norm, 0)
                    if count > 0:
                        hit = True
                        score += count
                if hit:
                    matched.append(display)

        matched_count = len(matched)
        if matched_count == 0:
            return None
        return CsvSearchResult(row=row, score=score, matched=matched, matched_count=matched_count, expected=expected)

    def search(
        self,
        query_by_cat: Dict[str, str],
        limit: int = 20,
    ) -> Tuple[List[CsvSearchResult], List[Tuple[str, str]]]:
        pairs_by_cat, keywords_display = self.build_pairs_by_cat(query_by_cat)
        if not any(pairs_by_cat.values()):
            return [], []

        results: List[CsvSearchResult] = []
        for row in self.read_rows():
            scored = self.score_row(row, pairs_by_cat)
            if scored:
                results.append(scored)

        results.sort(
            key=lambda r: (
                -(r.matched_count / max(r.expected, 1)),
                -r.matched_count,
                -r.score,
            )
        )
        return results[:limit], keywords_display
