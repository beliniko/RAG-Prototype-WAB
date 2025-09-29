import os
import csv
import re
import argparse
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Wir nutzen die bestehenden Bausteine aus deinem rag_min.py
from rag_min import (
    load_text_docs_from_folder,
    load_pdf_docs_from_folder,
    build_corpus,
    embed_texts,
    build_faiss_ip_index,
    retrieve_top_k,
    build_messages,
    call_lmstudio_chat,
    DATA_DIR,
    EMBED_MODEL_NAME,
    LMSTUDIO_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
)

# ------------------------------ Utils ------------------------------

def sentence_split(text: str) -> List[str]:
    """Sehr einfache Satzaufteilung.
    Rückgabe: Liste der Sätze (ohne Leerstrings)."""

    # Split an . ? ! ; und Zeilenumbrüchen, behalte nur sinnvolle Stücke
    parts = re.split(r'[\.!?;\n]+', text)
    return [s.strip() for s in parts if s.strip()]

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kosinus-Ähnlichkeit zwischen zwei Mengen normalisierter Vektoren."""
    # A: (m, d), B: (n, d). Beide sollen bereits L2-normalisiert sein.
    return A @ B.T

def parse_gold_items(gold_str: str) -> Set[Tuple[str, Optional[int]]]:
    """
    Erwartetes Format pro Datensatz (Beispiele):
      "calculator.pdf#1;calculator.pdf#2"
      "calculator.pdf"
    '#' trennt Dateiname und 1-basierte Seite. Ohne '#': jede Seite der Datei zählt als relevant.
    """
    items = set()
    if not gold_str:
        return items
    for token in re.split(r'[;,]', gold_str):
        t = token.strip()
        if not t:
            continue
        if "#" in t:
            fname, page = t.split("#", 1)
            try:
                page_num = int(page)
            except ValueError:
                page_num = None
            items.add((os.path.basename(fname).lower(), page_num))
        else:
            items.add((os.path.basename(t).lower(), None))
    return items

def hit_key(meta: Dict[str, any]) -> Tuple[str, Optional[int]]:
    """
    Erzeugt den Vergleichsschlüssel für einen Treffer:
    (dateiname_lower, 1-basierte_Seite_oder_None)
    """
    src = os.path.basename(meta.get("source", "unknown")).lower()
    page = meta.get("page", None)
    if isinstance(page, int):
        page = page + 1  # in rag_min ist page 0-basiert gespeichert
    else:
        page = None
    return (src, page)

# ------------------------------ Metriken ------------------------------

def recall_at_k(gold_keys: Set[Tuple[str, Optional[int]]],
                hit_keys: List[Tuple[str, Optional[int]]]) -> float:
    """
    Recall = (# relevante in Top-k) / (# relevante insgesamt)
    Wenn Gold-Items ohne Seitenangabe vorkommen, matchen sie jede Seite der Datei.
    """
    if not gold_keys:
        return 0.0
    found = 0
    for g in gold_keys:
        g_file, g_page = g
        if g_page is None:
            # gilt als Treffer, wenn irgendein Hit dieselbe Datei hat
            if any(hf == g_file for (hf, hp) in hit_keys):
                found += 1
        else:
            if g in hit_keys:
                found += 1
    return found / max(1, len(gold_keys))

def faithfulness_score(answer: str,
                       context_texts: List[str],
                       embedder: SentenceTransformer,
                       threshold: float = 0.70,
                       max_ctx_sentences: int = 800) -> Tuple[float, int, int]:
    """
    Faithfulness ≈ (# Antwort-Sätze, die semantisch von irgendeinem Kontext-Satz gestützt sind)
                  / (# Antwort-Sätze insgesamt)

    Heuristik:
      - Splitte die Antwort in Sätze.
      - Splitte alle Kontext-Passagen in Sätze.
      - Embedde Antwort-Sätze und Kontext-Sätze (normalisiert).
      - Ein Antwort-Satz gilt als 'gestützt', wenn die max. Cosinus-Ähnlichkeit >= threshold.
    """
    ans_sents = sentence_split(answer)
    if not ans_sents:
        return 0.0, 0, 0

    ctx_sents = []
    for t in context_texts:
        ctx_sents.extend(sentence_split(t))

    if not ctx_sents:
        # Kein Kontext => keine Stützung möglich
        return 0.0, 0, len(ans_sents)

    # Begrenzen (Speed)
    if len(ctx_sents) > max_ctx_sentences:
        ctx_sents = ctx_sents[:max_ctx_sentences]

    # Embeddings (normalisiert)
    ans_vecs = embedder.encode(ans_sents, normalize_embeddings=True)
    ctx_vecs = embedder.encode(ctx_sents, normalize_embeddings=True)

    S = cosine_sim_matrix(ans_vecs, ctx_vecs)
    supported = 0
    for i in range(S.shape[0]):
        if np.max(S[i, :]) >= threshold:
            supported += 1

    return supported / len(ans_sents), supported, len(ans_sents)

# ------------------------------ Hauptablauf ------------------------------

def load_gold_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append(row)
    return rows

def main():
    parser = argparse.ArgumentParser(description="Evaluate Recall@k (Retriever) und Faithfulness (Antwort) für rag_min.py")
    parser.add_argument("--dataset", required=True, help="Pfad zur CSV mit Spalten: question,gold[,answer]")
    parser.add_argument("--k", type=int, default=int(os.environ.get("RAG_EVAL_K", "5")), help="k für Recall@k und Retrieval")
    parser.add_argument("--faith-threshold", type=float, default=0.70, help="Schwellwert für Faithfulness (Cosine-Ähnlichkeit)")
    args = parser.parse_args()

    # 1) Gold-Set laden
    ds = load_gold_csv(args.dataset)
    if not ds:
        print("Leere oder ungültige Dataset-Datei.")
        return

    # 2) Index bauen (gleicher Weg wie in rag_min)
    print("[Eval] Baue Index aus DATA_DIR:", DATA_DIR)
    docs = []
    docs.extend(load_text_docs_from_folder(DATA_DIR))
    docs.extend(load_pdf_docs_from_folder(DATA_DIR))
    texts, metas = build_corpus(docs)

    print("[Eval] Lade Embedding-Modell:", EMBED_MODEL_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embs = embed_texts(embedder, texts)
    index = build_faiss_ip_index(embs)

    # 3) Evaluation
    recalls = []
    faiths = []
    total_supported = 0
    total_ans_sents = 0

    for i, row in enumerate(ds, 1):
        q = (row.get("question") or "").strip()
        gold_str = (row.get("gold") or "").strip()
        gold_keys = parse_gold_items(gold_str)

        if not q:
            print(f"[{i}] Zeile ohne Frage – übersprungen.")
            continue

        # Retrieval
        hits = retrieve_top_k(q, embedder, index, texts, metas, args.k)
        hit_keys = [hit_key(h["meta"]) for h in hits]
        rec = recall_at_k(gold_keys, hit_keys)
        recalls.append(rec)

        # Antwort (evtl. generieren)
        ans = (row.get("answer") or "").strip()
        if not ans and args.generate:
            msgs = build_messages(q, hits)
            try:
                ans = call_lmstudio_chat(LMSTUDIO_MODEL, msgs, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            except Exception as e:
                print(f"[{i}] LM-Fehler: {e}")
                ans = ""

        # Faithfulness nur berechnen, wenn eine Antwort vorliegt
        if ans:
            ctx_texts = [h["text"] for h in hits]
            f_score, supported, ans_count = faithfulness_score(
                ans, ctx_texts, embedder, threshold=args.faith_threshold
            )
            faiths.append(f_score)
            total_supported += supported
            total_ans_sents += ans_count

        # Kurzes per-Item-Log
        print(f"[{i}] Recall@{args.k}={rec:.2f}" + (f", Faithfulness={faiths[-1]:.2f}" if ans else ", Faithfulness=–"))

    # 4) Aggregation
    rec_avg = sum(recalls)/len(recalls) if recalls else 0.0
    faith_avg = sum(faiths)/len(faiths) if faiths else 0.0
    print("\n=== Gesamtergebnisse ===")
    print(f"Durchschnitt Recall@{args.k}: {rec_avg:.3f}  (n={len(recalls)})")
    if faiths:
        print(f"Durchschnitt Faithfulness: {faith_avg:.3f}  (n={len(faiths)})")
        if total_ans_sents:
            print(f"Unterstützte Sätze: {total_supported}/{total_ans_sents} "
                  f"({(total_supported/total_ans_sents):.3f})")
    else:
        print("Faithfulness: keine Antworten bewertet (spalte 'answer' leer und --generate nicht gesetzt).")

if __name__ == "__main__":
    main()