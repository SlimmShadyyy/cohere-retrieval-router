import re
import numpy as np
from rank_bm25 import BM25Okapi
from cache_manager import get_doc_embeddings, get_all_query_embeddings, cached_rerank

STOPWORDS = {"what", "does", "why", "my", "to", "because", "of", "which", "in", "the", "a", "an", "is", "for", "and", "or", "when"}

def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_.\-]+", text.lower())
    return [w for w in words if w not in STOPWORDS]

corpus = [
    {"id": "doc_1", "text": "Error code ERR_SSL_VERSION_OR_CIPHER_MISMATCH occurs when the client and server do not support a common SSL protocol version."},
    {"id": "doc_2", "text": "When a browser refuses a secure connection due to outdated encryption handshakes, update your TLS settings."},
    # Negation Pair: doc_3 excludes Node.js, doc_4 repeats 'backend frameworks' + 'Node.js'
    {"id": "doc_3", "text": "Our architecture uses Python and Go in production. We migrated away from Node.js due to memory overhead."},
    {"id": "doc_4", "text": "Our architecture uses backend frameworks like Node.js in production, running Node.js microservices at scale."},
    # Version ID Pair: identical topic, only the patch number differs!
    {"id": "doc_5", "text": "Patch v4.12.9-rc2 fixes the memory leak in the WebSocket connection pool when idle timeout is low."},
    {"id": "doc_6_trap", "text": "Patch v9.01.0-beta fixes the memory leak in the WebSocket connection pool when idle timeout is low while improving patch stability."}
]

doc_ids = [d["id"] for d in corpus]
doc_texts = [d["text"] for d in corpus]

bm25 = BM25Okapi([tokenize(t) for t in doc_texts])
doc_embs = get_doc_embeddings(doc_texts)

test_queries = [
    {
        "type": "1. Zero-Keyword Paraphrase (BM25 fails because zero shared words!)",
        "query": "Why did my webpage fail to load because of old security certificates?",
        "gold": "doc_2"
    },
    {
        "type": "2. Strict Negation (Embed & BM25 get tricked by 'Node.js' trap in doc_4, Rerank reads 'migrated away')",
        "query": "Which architecture runs in production EXCLUDING Node.js?",
        "gold": "doc_3"
    }
]

query_embs = get_all_query_embeddings([q["query"] for q in test_queries])

for i, q_obj in enumerate(test_queries):
    q = q_obj["query"]
    
    bm25_scores = bm25.get_scores(tokenize(q))
    bm25_top1 = doc_ids[np.argmax(bm25_scores)]
    
    emb_scores = np.dot(doc_embs, query_embs[i])
    emb_top1 = doc_ids[np.argmax(emb_scores)]
    
    rerank_res = cached_rerank(q, doc_texts, top_n=3)
    rerank_top1 = doc_ids[rerank_res[0]["index"]]
    
    print(f"\n--- {q_obj['type']} ---")
    print(f"Query: {q} (Correct Answer: {q_obj['gold']})")
    print(f"  BM25 Picked   : {bm25_top1:10s} -> {'[PASS]' if bm25_top1 == q_obj['gold'] else '[FAIL]'}")
    print(f"  Embed Picked  : {emb_top1:10s} -> {'[PASS]' if emb_top1 == q_obj['gold'] else '[FAIL]'}")
    print(f"  Rerank Picked : {rerank_top1:10s} -> {'[PASS]' if rerank_top1 == q_obj['gold'] else '[FAIL]'}")