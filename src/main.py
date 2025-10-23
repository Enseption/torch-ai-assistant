import os, sys, json, argparse, glob
from typing import List, Dict, Any, Tuple
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
META_PATH = os.path.join(DATA_DIR, "meta.json")

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    out = []
    for p in reader.pages:
        out.append(p.extract_text() or "")
    return "\\n".join(out)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, max_tokens: int = 300, overlap: int = 60) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    step = max_tokens - overlap
    while i < len(words):
        ch = " ".join(words[i:i+max_tokens])
        if ch.strip():
            chunks.append(ch)
        i += step
    return chunks

def load_and_chunk(path: str) -> List[Tuple[str, int, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md"]:
        text = read_text(path)
    elif ext == ".pdf":
        text = read_pdf(path)
    else:
        return []
    chunks = chunk_text(text)
    did = os.path.basename(path)
    return [(did, i, ch) for i, ch in enumerate(chunks) if ch.strip()]

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict[str, Any]] = []

    def add(self, embs: np.ndarray, metas: List[Dict[str, Any]]):
        faiss.normalize_L2(embs)
        self.index.add(embs.astype(np.float32))
        self.meta.extend(metas)

    def search(self, q_emb: np.ndarray, k: int = 5):
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb.astype(np.float32), k)
        res = []
        for sc, ids in zip(scores, idxs):
            row = []
            for s, i in zip(sc, ids):
                if i == -1: continue
                row.append({"score": float(s), "metadata": self.meta[i]})
            res.append(row)
        return res

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dim: int):
        vs = cls(dim)
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            vs.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                vs.meta = json.load(f)
        return vs

def embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

def semantic_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    vs = VectorStore.load(EMB_DIM)
    if len(vs.meta) == 0:
        print("Index is empty. Run 'ingest' first.", file=sys.stderr); sys.exit(1)
    q_emb = embed_texts(EMB_MODEL, [query])
    return vs.search(q_emb, k=k)[0]

def call_ollama(prompt: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "mistral")
    try:
        import ollama
        client = ollama.Client(host=host)
        resp = client.chat(model=model, messages=[
            {"role":"system","content":"Be concise; use the provided context; cite with [doc#chunk]; note ambiguities."},
            {"role":"user","content": prompt},
        ])
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"[ollama-error] {e}"

def make_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    ctx_lines = [f"[{h['metadata']['doc_id']}#{h['metadata']['chunk_id']}] {h['metadata']['text']}" for h in hits]
    ctx = "\\n\\n".join(ctx_lines)
    return f"""Answer the QUESTION using the CONTEXT. If ambiguous, explain briefly and propose follow-ups.

QUESTION:
{question}

CONTEXT:
{ctx}

Return a concise answer grounded in the context. Cite chunks like [doc#chunk].
"""

def to_json(question: str, hits: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
    citations = [{
        "doc_id": h["metadata"]["doc_id"],
        "chunk_id": h["metadata"]["chunk_id"],
        "score": float(h["score"]),
        "text": h["metadata"]["text"][:240]
    } for h in hits]
    disambig = any(w in question.lower() for w in [" or ", " vs ", "versus", "difference", "compare", "ambiguous"])
    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "disambiguation": {
            "needed": disambig,
            "notes": "May need scope/definitions; see follow_ups." if disambig else ""
        },
        "follow_ups": (["Clarify ask."] if disambig else [])
    }

def cmd_ingest(folder: str):
    files = [p for p in glob.glob(os.path.join(folder, "*")) if os.path.isfile(p)]
    if not files:
        print("No files found.", file=sys.stderr); sys.exit(1)
    chunks, metas = [], []
    for p in files:
        if os.path.splitext(p)[1].lower() not in [".txt", ".md", ".pdf"]:
            continue
        for did, cid, text in load_and_chunk(p):
            chunks.append(text)
            metas.append({"doc_id": did, "chunk_id": cid, "text": text})
    if not chunks:
        print("No supported documents.", file=sys.stderr); sys.exit(1)
    embs = embed_texts(EMB_MODEL, chunks)
    vs = VectorStore.load(EMB_DIM)
    vs.add(embs, metas); vs.save()
    print(f"Ingested {len(chunks)} chunks from {len(files)} files.")

def cmd_search(query: str, k: int):
    hits = semantic_search(query, k=k)
    print(json.dumps(hits, ensure_ascii=False, indent=2))

def cmd_ask(question: str, k: int, as_json: bool):
    hits = semantic_search(question, k=k)
    prompt = make_prompt(question, hits)
    answer = call_ollama(prompt)
    out = to_json(question, hits, answer)
    if as_json: print(json.dumps(out, ensure_ascii=False, indent=2))
    else: print(out["answer"])

def cmd_summarize(k: int, as_json: bool):
    seed = "Summarize the key concepts across this corpus."
    hits = semantic_search("key concepts overview", k=k)
    prompt = make_prompt(seed, hits)
    answer = call_ollama(prompt)
    out = to_json(seed, hits, answer)
    if as_json: print(json.dumps(out, ensure_ascii=False, indent=2))
    else: print(out["answer"])

def main():
    ap = argparse.ArgumentParser(description="Basic RAG (Ollama + Mistral)")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("ingest"); p.add_argument("folder")
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("--k", type=int, default=5)
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("--k", type=int, default=5); p.add_argument("--json", action="store_true")
    p = sub.add_parser("summarize"); p.add_argument("--k", type=int, default=12); p.add_argument("--json", action="store_true")

    args = ap.parse_args()
    if args.cmd == "ingest": cmd_ingest(args.folder)
    elif args.cmd == "search": cmd_search(args.query, args.k)
    elif args.cmd == "ask": cmd_ask(args.question, args.k, args.json)
    elif args.cmd == "summarize": cmd_summarize(args.k, args.json)
    else: ap.print_help()

if __name__ == "__main__":
    main()
