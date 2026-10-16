import os
import json
import random
import time
import cohere
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CORPUS_PATH = os.path.join(DATA_DIR, "corpus.json")
QUERIES_PATH = os.path.join(DATA_DIR, "benchmark_queries.json")

def build_corpus(num_docs=200):
    """Downloads real scientific abstracts from mteb/scifact and engineers hard-negative traps."""
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        print(f"[LOADED] {len(corpus)} documents from {CORPUS_PATH}")
        return corpus

    print("Downloading SciFact corpus from HuggingFace (mteb/scifact)...")
    ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    
    random.seed(42)
    sampled = ds.shuffle(seed=42).select(range(num_docs))
    
    corpus = []
    for i, row in enumerate(sampled):
        doc_id = f"doc_{i:03d}"
        title = row["title"]
        body = row["text"]
        
        # Bucket 1 (Docs 0-29): Exact Alphanumeric IDs + Twin Distractor with Wrong ID
        if i < 30:
            code_id = f"ERR-{4050 + i * 7}-B"
            version = f"v{i % 4 + 2}.{i % 9 + 1}.{i % 7}"
            text = f"[System {code_id} | Firmware {version}] {title}. {body}"
            meta_tag = {"code_id": code_id, "version": version}
            
            # Plant a twin distractor with the SAME semantic topic but DIFFERENT ID/Version!
            trap_code = f"ERR-{9010 + i * 3}-Z"
            trap_ver = f"v9.{i % 5}.0-legacy"
            trap_text = f"[System {trap_code} | Firmware {trap_ver}] {title}. {body}"
            corpus.append({
                "doc_id": f"trap_num_{i:03d}",
                "title": f"{title} ({trap_ver})",
                "text": trap_text,
                "meta": {}
            })
            
        # Bucket 3 (Docs 60-89): Strict Negation + High-Keyword Trap Doc
        elif 60 <= i < 90:
            text = (
                f"{title}. {body} IMPORTANT CLINICAL CONSTRAINT: This pathway operates strictly "
                f"WITHOUT beta-adrenergic receptor activation and must be excluded in beta-agonist protocols."
            )
            meta_tag = {"excluded_term": "beta-adrenergic receptor activation"}
            
            trap_text = (
                f"{title} via Beta-Adrenergic Receptor Activation. {body} "
                f"Specifically, beta-adrenergic receptor activation is the primary required driver "
                f"and relies heavily on active beta-adrenergic receptor signaling."
            )
            corpus.append({
                "doc_id": f"trap_neg_{i:03d}",
                "title": f"{title} (Beta-Adrenergic Active)",
                "text": trap_text,
                "meta": {}
            })
            
        # Bucket 5 (Docs 120-149): Needle in a Long Document (bury needle inside 3 unrelated docs)
        elif 120 <= i < 150:
            filler_1 = sampled[(i + 35) % num_docs]["text"]
            filler_2 = sampled[(i + 70) % num_docs]["text"]
            filler_3 = sampled[(i + 105) % num_docs]["text"]
            text = f"{filler_1} {filler_2} SUB-SECTION FINDING: {title}. {body} {filler_3}"
            meta_tag = {"needle": f"{title}. {body}"}
            
        else:
            text = f"{title}. {body}"
            meta_tag = {}

        corpus.append({
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "meta": meta_tag
        })

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)
    print(f"[SAVED] {len(corpus)} total documents (including hard-negative traps) to {CORPUS_PATH}")
    return corpus


PROMPT_TEMPLATES = {
    "numeric_id": (
        "Write a short search query (7-12 words) asking specifically about System {code_id} and Firmware {version} "
        "regarding the general topic of this passage. You MUST include '{code_id}' and '{version}' verbatim in the query. "
        "Return ONLY the query string.\n\nPassage: {text}"
    ),
    "semantic": (
        "Write a natural everyday question (10-16 words) asking about the core effect described in this passage, "
        "using ONLY plain English synonyms so that ZERO scientific/technical nouns or numbers from the passage appear in your question. "
        "Return ONLY the query string.\n\nPassage: {text}"
    ),
    "negation": (
        "Write a search query asking for the mechanism in this passage, and explicitly end the query with "
        "'WITHOUT beta-adrenergic receptor activation (not beta-adrenergic)'. "
        "Return ONLY the query string.\n\nPassage: {text}"
    ),
    "complex": (
        "Write a detailed multi-condition question (22-32 words) asking how two specific factors in this passage "
        "interact to produce the final outcome. Paraphrase moderately. "
        "Return ONLY the query string.\n\nPassage: {text}"
    ),
    "long_doc": (
        "Write a focused question (10-15 words) asking about this specific finding (paraphrase slightly so it is not copy-pasted): "
        "{needle}\nReturn ONLY the query string."
    )
}

def generate_benchmark_queries(corpus, queries_per_bucket=15):
    """Generates 75 balanced queries (15 per bucket) and saves after every query."""
    real_docs = [d for d in corpus if d["doc_id"].startswith("doc_")]

    existing_queries = []
    if os.path.exists(QUERIES_PATH):
        with open(QUERIES_PATH, "r", encoding="utf-8") as f:
            existing_queries = json.load(f)
        if len(existing_queries) >= queries_per_bucket * 5:
            print(f"[LOADED] All {len(existing_queries)} benchmark queries already exist in {QUERIES_PATH}")
            return existing_queries

    done_keys = {(q["category"], q["gold_doc_id"]) for q in existing_queries}

    buckets = [
        ("numeric_id", real_docs[0 : queries_per_bucket]),
        ("semantic",   real_docs[30 : 30 + queries_per_bucket]),
        ("negation",   real_docs[60 : 60 + queries_per_bucket]),
        ("complex",    real_docs[90 : 90 + queries_per_bucket]),
        ("long_doc",   real_docs[120 : 120 + queries_per_bucket])
    ]

    q_counter = len(existing_queries)

    for category, docs_slice in buckets:
        print(f"\nGenerating '{category}' queries...")
        for doc in tqdm(docs_slice):
            if (category, doc["doc_id"]) in done_keys:
                continue

            prompt = PROMPT_TEMPLATES[category].format(
                text=doc["text"],
                code_id=doc["meta"].get("code_id", ""),
                version=doc["meta"].get("version", ""),
                needle=doc["meta"].get("needle", doc["text"])
            )

            query_str = None
            for attempt in range(5):
                try:
                    res = co.chat(
                        model="command-a-03-2025",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    query_str = res.message.content[0].text.strip().strip('"').strip("'")
                    break
                except Exception as e:
                    print(f"Rate limit pause (6.5s)... {e}")
                    time.sleep(6.5)

            if query_str:
                existing_queries.append({
                    "query_id": f"q_{q_counter:03d}",
                    "category": category,
                    "query": query_str,
                    "gold_doc_id": doc["doc_id"]
                })
                q_counter += 1
                with open(QUERIES_PATH, "w", encoding="utf-8") as f:
                    json.dump(existing_queries, f, indent=2)

            time.sleep(3.2) # Respect 20 calls/min Trial limit

    print(f"\n[DONE] Saved {len(existing_queries)} total benchmark queries to {QUERIES_PATH}")
    return existing_queries

if __name__ == "__main__":
    corpus = build_corpus(num_docs=200)
    generate_benchmark_queries(corpus, queries_per_bucket=15)