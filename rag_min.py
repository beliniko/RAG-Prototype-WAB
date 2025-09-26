import os
import glob
from typing import List, Dict, Any, Tuple

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

from dotenv import load_dotenv
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== LM Studio Diagnose ==========
def debug_lmstudio_connection() -> bool:
    """
    Prüft, ob der LM Studio Local Server erreichbar ist und welches Modell er anbietet.
    """
    print(f"[LM Studio] Endpoint: {LMSTUDIO_BASE_URL}")
    url = f"{LMSTUDIO_BASE_URL}/models"
    headers = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        print(f"[LM Studio] GET /models → Status {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        ids = [m.get("id") for m in data.get("data", [])]
        print("[LM Studio] Models verfügbar:", ids if ids else "(keine)")
        if LMSTUDIO_MODEL not in ids:
            print(f"[LM Studio] WARN: Modell '{LMSTUDIO_MODEL}' wird nicht vom Server bereitgestellt.")
            print("→ Wähle das Modell in LM Studio aus und starte den Local Server,")
            print("  oder setze RAG_LLM_MODEL auf einen der obigen IDs.")
        return True
    except requests.exceptions.ConnectionError as e:
        print(f"[LM Studio] Verbindung fehlgeschlagen: {e}")
        print("→ Starte in LM Studio den 'Local Server' (Playground › Local Server › Start).")
        print("→ Prüfe den Port/URL und setze ggf. OPENAI_BASE_URL, z. B. http://localhost:1234/v1")
        return False
    except Exception as e:
        print(f"[LM Studio] /models Fehler: {e}")
        return False

# ============== CONFIG ==================
DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")     # Ordner mit PDFs/TXT/MD
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# Chunking: Zeichen-basiert (für MVP simpel & robust)
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "1200"))   # ~1–2 Absätze
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))

# Embedding-Modell
EMBED_MODEL_NAME = os.environ.get(
    "RAG_EMBED_MODEL",
    "intfloat/multilingual-e5-base"   # Multilingual Modell, dass Deutsche Texte verarbeiten kann
)

# LM Studio (OpenAI-kompatibel)
LMSTUDIO_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY  = os.environ.get("OPENAI_API_KEY", "lm-studio")  # Dummy-Key ist ok
LMSTUDIO_MODEL    = os.environ.get("RAG_LLM_MODEL", "gpt-oss-20b")  # exakt wie in LM Studio angezeigt

TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.0"))
MAX_TOKENS  = int(os.environ.get("RAG_MAX_TOKENS", "700"))
# ========================================

# ---------- Text Splitter (LangChain) ----------
# Nutzt den LangChain-RecursiveCharacterTextSplitter für robuste Abschnitte
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)


# ---------- Hilfsfunktionen: Laden & Aufbereiten ----------

def load_text_docs_from_folder(data_dir: str) -> List[Dict[str, Any]]:
    """
    Lädt .txt und .md Dateien als einzelne Dokumente.
    Rückgabe: Liste von {"source": Pfad, "text": Text, "page": None}
    """
    docs: List[Dict[str, Any]] = []
    patterns = ["**/*.txt", "**/*.md"]
    for pat in patterns:
        for path in glob.glob(os.path.join(data_dir, pat), recursive=True):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                docs.append({"source": path, "text": txt, "page": None})
            except Exception:
                pass
    return docs


def load_pdf_docs_from_folder(data_dir: str) -> List[Dict[str, Any]]:
    """
    Lädt PDFs seitenweise mit Seitenangaben als Metadaten haben.
    Rückgabe: Liste von {"source": Pfad, "text": Text_der_Seite, "page": int}
    """
    docs: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True):
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({"source": path, "text": text, "page": i})
        except Exception:
            pass
    return docs


def build_corpus(docs: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Dokumente in Chunks aufteilen und Metadaten (Quelle, Seite, Chunk-Id) mitführen.
    Rückgabe:
      texts: Liste von Chunk-Texten
      metas: Liste gleichlanger Metadaten-Dicts
    """
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for d in docs:
        src, page = d["source"], d["page"]
        parts = SPLITTER.split_text(d["text"])
        for j, ch in enumerate(parts):
            texts.append(ch)
            metas.append({"source": src, "page": page, "chunk_id": j})
    return texts, metas


# ---------- Embeddings & Index ----------

def embed_texts(model: SentenceTransformer, items: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Erzeugt normalisierte Embeddings (Cosine-Ready) und gibt float32-Matrix zurück.
    """
    vecs = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        embs = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
        vecs.append(embs)
    return np.vstack(vecs).astype("float32")


def build_faiss_ip_index(embs: np.ndarray) -> faiss.Index:
    """
    FAISS Index für Inner Product (mit normalisierten Vektoren = Cosinus-Ähnlichkeit).
    """
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


def retrieve_top_k(
        query: str,
        embedder: SentenceTransformer,
        index: faiss.Index,
        texts: List[str],
        metas: List[Dict[str, Any]],
        k: int
) -> List[Dict[str, Any]]:
    """
    Query einbetten, im Index suchen und Top-k Treffer zurückgeben.
    """
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = index.search(q, k)
    out: List[Dict[str, Any]] = []
    for rank, (idx, sc) in enumerate(zip(ids[0], scores[0]), start=1):
        item = {
            "rank": rank,
            "score": float(sc),
            "text": texts[idx],
            "meta": metas[idx]
        }
        out.append(item)
    return out


# ---------- Prompting & LLM ----------

SYSTEM_PROMPT = (
    "Du bist ein Assistenzsystem für Unternehmensdokumente.\n"
    "Beantworte die Frage NUR anhand des bereitgestellten Kontexts.\n"
    "Wenn die Antwort nicht eindeutig im Kontext steht, antworte: 'Ich weiß es nicht.'\n"
    "Gib, wenn möglich, kurze Quellenhinweise (Dateiname und Seite)."
)

def build_messages(user_question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Baut einen einfachen Prompt mit Kontextblöcken (Top-k Treffer).
    """
    ctx_blocks = []
    for h in hits:
        src = os.path.basename(h["meta"].get("source", "unbekannt"))
        page = h["meta"].get("page")
        page_str = f", S.{page + 1}" if isinstance(page, int) else ""
        header = f"[{h['rank']} | {src}{page_str}]"
        ctx_blocks.append(f"{header}\n{h['text']}")
    context = "\n\n".join(ctx_blocks)

    user = (
        f"Frage: {user_question}\n\n"
        f"Kontext:\n{context}\n\n"
        f"Antwort:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user}
    ]


def call_lmstudio_chat(
        model: str, messages: List[Dict[str, str]],
        temperature: float = 0.0, max_tokens: int = 700
) -> str:
    """
    Ruft den lokalen LM Studio Server (OpenAI-kompatibel) via HTTP auf,
    ohne das openai-Python-SDK zu verwenden (um httpx-Probleme zu umgehen).
    """
    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    if not resp.ok:
        print("[LM Studio] Fehlerantwort:", resp.text[:500])
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------- Main ----------

def main():
    load_dotenv()  # optional: .env einlesen

    # Vor dem Start: LM Studio erreichen?
    if not debug_lmstudio_connection():
        return

    # 1) Dokumente laden
    print(f"[1/5] Lade Dokumente aus: {DATA_DIR}")
    docs = []
    docs.extend(load_text_docs_from_folder(DATA_DIR))
    docs.extend(load_pdf_docs_from_folder(DATA_DIR))
    if not docs:
        print("Keine Dokumente gefunden. Lege PDFs/TXT/MD unter ./data/ ab.")
        return

    # 2) Chunks & Metadaten
    print("[2/5] Erzeuge Chunks…")
    texts, metas = build_corpus(docs)
    print(f"  → {len(texts)} Chunks")

    # 3) Embedding & Index
    print("[3/5] Lade Embedding-Modell & erstelle Vektorindex…")
    emb_model = SentenceTransformer(EMBED_MODEL_NAME)
    embs = embed_texts(emb_model, texts)
    index = build_faiss_ip_index(embs)
    print("  → Index fertig")

    # 4) Interaktiver Loop
    print("[4/5] LLM vorbereitet:", LMSTUDIO_MODEL)
    print("[5/5] Bereit. Tippe eine Frage (oder 'exit').")

    while True:
        q = input("\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        hits = retrieve_top_k(q, emb_model, index, texts, metas, TOP_K)
        msgs = build_messages(q, hits)
        try:
            answer = call_lmstudio_chat(
                LMSTUDIO_MODEL, msgs,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS
            )
        except Exception as e:
            print("Fehler beim LLM-Aufruf:", e)
            continue

        print("\nAntwort:\n", answer)
        print("\nQuellen (Top-k):")
        seen = set()
        for h in hits:
            src = os.path.basename(h["meta"].get("source", "unbekannt"))
            page = h["meta"].get("page")
            key = (src, page)
            if key in seen:
                continue
            seen.add(key)
            page_str = f", S.{page + 1}" if isinstance(page, int) else ""
            print(f"  - {src}{page_str}")

if __name__ == "__main__":
    main()