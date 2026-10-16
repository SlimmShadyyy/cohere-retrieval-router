import os
import json
import time
import hashlib
import numpy as np
import cohere
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found! Make sure it is saved inside your .env file.")

co = cohere.ClientV2(api_key=api_key)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _hash_key(prefix: str, payload: str) -> str:
    """Creates a unique filename based on the exact query/documents passed in."""
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.json")

def _embed_batch_with_retry(batch: list[str], model: str, input_type: str) -> list[list[float]]:
    """Calls co.embed with automatic backoff if the 100k token/min trial limit is reached."""
    for attempt in range(8):
        try:
            res = co.embed(
                texts=batch,
                model=model,
                input_type=input_type,
                embedding_types=["float"]
            )
            return res.embeddings.float
        except Exception as e:
            wait_sec = 15 * (attempt + 1)
            print(f"\n[Token Rate Limit] Pausing {wait_sec}s for minute window to reset...")
            time.sleep(wait_sec)
    raise RuntimeError("Failed to embed batch after multiple retries.")

def get_doc_embeddings(texts: list[str], model="embed-v4.0") -> np.ndarray:
    """Embeds all corpus documents in safe batches of 40 and saves the matrix to disk."""
    payload_hash = hashlib.sha256("||".join(texts).encode("utf-8")).hexdigest()[:10]
    cache_path = os.path.join(CACHE_DIR, f"corpus_emb_{len(texts)}_{payload_hash}.npy")
    
    if os.path.exists(cache_path):
        print(f"[CACHE HIT] Loaded {len(texts)} document embeddings from disk.")
        return np.load(cache_path)

    print(f"[API CALL] Embedding {len(texts)} documents with Cohere ({model}) in batches of 40...")
    all_embeddings = []
    batch_size = 40
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  -> Embedding docs {i+1} to {min(i+batch_size, len(texts))}...")
        vecs = _embed_batch_with_retry(batch, model=model, input_type="search_document")
        all_embeddings.extend(vecs)
        time.sleep(2.0)  # Gentle pacing between batches

    emb_matrix = np.array(all_embeddings, dtype=np.float32)
    np.save(cache_path, emb_matrix)
    print(f"[SAVED] Cached {len(texts)} document embeddings to {cache_path}")
    return emb_matrix

def get_all_query_embeddings(queries: list[str], model="embed-v4.0") -> np.ndarray:
    """Batch-embeds search queries with automatic rate-limit retry."""
    payload_hash = hashlib.sha256("||".join(queries).encode("utf-8")).hexdigest()[:10]
    cache_path = os.path.join(CACHE_DIR, f"queries_emb_{len(queries)}_{payload_hash}.npy")
    
    if os.path.exists(cache_path):
        print(f"[CACHE HIT] Loaded {len(queries)} query embeddings from disk.")
        return np.load(cache_path)

    print(f"[API CALL] Batch-embedding {len(queries)} queries with Cohere ({model})...")
    all_embeddings = []
    batch_size = 50
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        vecs = _embed_batch_with_retry(batch, model=model, input_type="search_query")
        all_embeddings.extend(vecs)

    emb_matrix = np.array(all_embeddings, dtype=np.float32)
    np.save(cache_path, emb_matrix)
    return emb_matrix

def cached_rerank(query: str, docs: list[str], top_n=10, model="rerank-v3.5") -> list[dict]:
    """Calls Cohere Rerank and caches the returned indices & relevance scores."""
    payload = json.dumps({"q": query, "docs": docs, "model": model, "top_n": top_n})
    cache_file = _hash_key("rerank", payload)
    
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Respect free-tier rate limit (10 calls/min on Rerank) with automatic retry
    for attempt in range(8):
        try:
            res = co.rerank(model=model, query=query, documents=docs, top_n=top_n)
            out = [{"index": r.index, "score": r.relevance_score} for r in res.results]
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(out, f)
            time.sleep(6.2)  # Smoothly pace to ~9.5 calls/min so we rarely trigger 429s
            return out
        except Exception as e:
            print(f"\n[Rerank Rate Limit] Waiting 10s... ({e})")
            time.sleep(10.0)
    raise RuntimeError("Failed to call Cohere Rerank after 8 retries.")