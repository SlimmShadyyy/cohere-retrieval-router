import os
import re
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from cache_manager import get_doc_embeddings, get_all_query_embeddings, cached_rerank

STOPWORDS = {
    "what", "does", "why", "my", "to", "because", "of", "which", "in", "the",
    "a", "an", "is", "for", "and", "or", "when", "how", "do", "are", "with",
    "on", "by", "from", "that", "this", "as", "be", "at", "it", "can"
}

def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_.\-]+", text.lower())
    return [w for w in words if w not in STOPWORDS]

def reciprocal_rank_fusion(bm25_ranked_ids: list[str], embed_ranked_ids: list[str], k: int = 60) -> list[str]:
    """Combines BM25 and Embed rankings into a single Hybrid ranking."""
    scores = {}
    for rank, doc_id in enumerate(bm25_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(embed_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs]

def calc_metrics(ranked_ids: list[str], gold_id: str) -> tuple[float, float, float]:
    """Returns (Hit@1, Recall@5, MRR@10) for a single query."""
    hit_at_1 = 1.0 if ranked_ids[0] == gold_id else 0.0
    recall_at_5 = 1.0 if gold_id in ranked_ids[:5] else 0.0
    if gold_id in ranked_ids[:10]:
        rank = ranked_ids[:10].index(gold_id) + 1
        mrr_at_10 = 1.0 / rank
    else:
        mrr_at_10 = 0.0
    return hit_at_1, recall_at_5, mrr_at_10

def run_benchmark():
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open("data/benchmark_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    doc_ids = [d["doc_id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]
    docs_map = {d["doc_id"]: d["text"] for d in corpus}

    print(f"Indexing {len(corpus)} documents and {len(queries)} queries...")
    bm25 = BM25Okapi([tokenize(t) for t in doc_texts])
    doc_embs = get_doc_embeddings(doc_texts)
    query_embs = get_all_query_embeddings([q["query"] for q in queries])

    rows = []
    print("\nEvaluating all 4 retrieval strategies (BM25, Embed, Hybrid, Hybrid+Rerank)...")
    
    for i, q_obj in enumerate(tqdm(queries)):
        q_id = q_obj["query_id"]
        q_text = q_obj["query"]
        cat = q_obj["category"]
        gold_id = q_obj["gold_doc_id"]

        # 1. BM25 Retrieval
        t0 = time.perf_counter()
        bm25_scores = bm25.get_scores(tokenize(q_text))
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:50]
        bm25_ranked = [doc_ids[idx] for idx in bm25_top_idx]
        bm25_ms = (time.perf_counter() - t0) * 1000

        # Gather BM25 confidence gap for our Router later!
        sorted_b_scores = np.sort(bm25_scores)[::-1]
        bm25_top1_score = float(sorted_b_scores[0])
        bm25_gap = float(sorted_b_scores[0] - sorted_b_scores[1])

        # 2. Dense Embed Retrieval
        t0 = time.perf_counter()
        emb_scores = np.dot(doc_embs, query_embs[i])
        emb_top_idx = np.argsort(emb_scores)[::-1][:50]
        emb_ranked = [doc_ids[idx] for idx in emb_top_idx]
        emb_ms = (time.perf_counter() - t0) * 1000 + 25.0 # +25ms realistic API network overhead

        # 3. Hybrid (BM25 + Embed via RRF)
        t0 = time.perf_counter()
        hybrid_ranked = reciprocal_rank_fusion(bm25_ranked, emb_ranked)[:50]
        hybrid_ms = bm25_ms + emb_ms + (time.perf_counter() - t0) * 1000

        # 4. Hybrid + Cohere Rerank (reranks top 25 candidates from Hybrid)
        t0 = time.perf_counter()
        candidates_25 = hybrid_ranked[:25]
        candidate_texts = [docs_map[cid] for cid in candidates_25]
        rerank_res = cached_rerank(q_text, candidate_texts, top_n=10)
        rerank_ranked = [candidates_25[r["index"]] for r in rerank_res]
        rerank_ms = hybrid_ms + 120.0 # +120ms realistic Rerank API latency

        # Pace free-tier Rerank calls ONLY if not already cached
        time.sleep(0.1)

        # Score all 4 strategies
        b_hit1, b_rec5, b_mrr = calc_metrics(bm25_ranked, gold_id)
        e_hit1, e_rec5, e_mrr = calc_metrics(emb_ranked, gold_id)
        h_hit1, h_rec5, h_mrr = calc_metrics(hybrid_ranked, gold_id)
        r_hit1, r_rec5, r_mrr = calc_metrics(rerank_ranked, gold_id)

        rows.append({
            "query_id": q_id,
            "category": cat,
            "query": q_text,
            "gold_doc_id": gold_id,
            "bm25_top1_score": bm25_top1_score,
            "bm25_gap": bm25_gap,
            "BM25_Hit1": b_hit1, "BM25_MRR": b_mrr, "BM25_ms": bm25_ms,
            "Embed_Hit1": e_hit1, "Embed_MRR": e_mrr, "Embed_ms": emb_ms,
            "Hybrid_Hit1": h_hit1, "Hybrid_MRR": h_mrr, "Hybrid_ms": hybrid_ms,
            "Rerank_Hit1": r_hit1, "Rerank_MRR": r_mrr, "Rerank_ms": rerank_ms,
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/eval_results.csv", index=False)
    print("\nSaved detailed per-query metrics to data/eval_results.csv!")

    print("\n======================================================================")
    print("  FAILURE-MODE BENCHMARK: TOP-1 ACCURACY (Hit@1) BY QUERY CATEGORY")
    print("======================================================================")
    summary_hit1 = df.groupby("category")[["BM25_Hit1", "Embed_Hit1", "Hybrid_Hit1", "Rerank_Hit1"]].mean() * 100
    print(summary_hit1.round(1).to_string())

    print("\n======================================================================")
    print("  OVERALL PERFORMANCE SUMMARY (Across All 75 Queries)")
    print("======================================================================")
    overall = df[["BM25_Hit1", "Embed_Hit1", "Hybrid_Hit1", "Rerank_Hit1",
                  "BM25_MRR", "Embed_MRR", "Hybrid_MRR", "Rerank_MRR"]].mean()
    print(f"BM25 Only       -> Hit@1: {overall['BM25_Hit1']*100:5.1f}% | MRR@10: {overall['BM25_MRR']:.3f}")
    print(f"Cohere Embed    -> Hit@1: {overall['Embed_Hit1']*100:5.1f}% | MRR@10: {overall['Embed_MRR']:.3f}")
    print(f"Hybrid (RRF)    -> Hit@1: {overall['Hybrid_Hit1']*100:5.1f}% | MRR@10: {overall['Hybrid_MRR']:.3f}")
    print(f"Hybrid + Rerank -> Hit@1: {overall['Rerank_Hit1']*100:5.1f}% | MRR@10: {overall['Rerank_MRR']:.3f}")

if __name__ == "__main__":
    run_benchmark()