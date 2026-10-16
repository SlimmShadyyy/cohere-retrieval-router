import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# 1. Load the evaluation results from evaluate.py
df = pd.read_csv("data/eval_results.csv")

# 2. Extract fast, zero-API-cost query features for routing
def extract_features(row):
    q = str(row["query"])
    q_lower = q.lower()
    
    # Feature 1: Regex check for alphanumeric codes, version numbers (v1.2), or hyphenated IDs (ERR-4050)
    has_code_or_ver = 1.0 if re.search(r"(err-\d+|v\d+\.\d+|\b\w+-\d+-\w+\b|\d+\.\d+)", q_lower) else 0.0
    
    # Feature 2: Explicit negation / exclusion keywords
    has_negation = 1.0 if re.search(r"\b(without|excluding|not|except|exclude|lacks|never)\b", q_lower) else 0.0
    
    # Feature 3: Query length (word count)
    word_count = float(len(q.split()))
    
    # Feature 4 & 5: BM25 lexical confidence (runs locally in 0.5ms before any API call!)
    bm25_top1 = float(row["bm25_top1_score"])
    bm25_gap = float(row["bm25_gap"])
    
    return pd.Series([has_code_or_ver, has_negation, word_count, bm25_top1, bm25_gap],
                     index=["has_code", "has_negation", "word_count", "bm25_top1", "bm25_gap"])

X = df.apply(extract_features, axis=1)

# 3. Build Router A: Interpretable Heuristic Router (Production Rule Engine)
def heuristic_router(row, feat):
    # Rule 1: If query has negation -> MUST use Hybrid + Rerank (Cross-Encoder needed)
    if feat["has_negation"] == 1.0:
        return "Rerank", row["Rerank_Hit1"], row["Rerank_MRR"], row["Rerank_ms"], 1
    
    # Rule 2: If query has exact ID/Version OR high BM25 confidence gap -> Fast BM25 / Hybrid
    if feat["has_code"] == 1.0 or feat["bm25_gap"] > 6.5:
        return "BM25", row["BM25_Hit1"], row["BM25_MRR"], row["BM25_ms"], 0
        
    # Rule 3: If BM25 top score is weak (likely semantic paraphrase with zero keyword overlap) -> Dense Embed
    if feat["bm25_top1"] < 12.0:
        return "Embed", row["Embed_Hit1"], row["Embed_MRR"], row["Embed_ms"], 0
        
    # Rule 4: Otherwise -> Hybrid RRF (or Rerank if margin is super tight)
    if feat["bm25_gap"] < 1.5:
        return "Rerank", row["Rerank_Hit1"], row["Rerank_MRR"], row["Rerank_ms"], 1
    return "Hybrid", row["Hybrid_Hit1"], row["Hybrid_MRR"], row["Hybrid_ms"], 0

h_routes = [heuristic_router(df.iloc[i], X.iloc[i]) for i in range(len(df))]
df["H_Route"] = [r[0] for r in h_routes]
df["H_Hit1"] = [r[1] for r in h_routes]
df["H_MRR"] = [r[2] for r in h_routes]
df["H_ms"] = [r[3] for r in h_routes]
df["H_rerank_calls"] = [r[4] for r in h_routes]

# 4. Build Router B: Learned ML Router (5-Fold Cross-Validated Logistic Regression)
# Create target label: Which cheapest strategy already achieves the best accuracy on this query?
def get_oracle_class(row):
    if row["BM25_Hit1"] == 1.0:
        return 0  # BM25 (Free & 0.5ms)
    elif row["Embed_Hit1"] == 1.0:
        return 1  # Embed (Cheap & 25ms)
    else:
        return 2  # Rerank (Expensive & 145ms)

y_oracle = df.apply(get_oracle_class, axis=1)
clf = LogisticRegression(max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ml_preds = cross_val_predict(clf, X, y_oracle, cv=cv)

ml_hit1, ml_mrr, ml_ms, ml_rerank_calls = [], [], [], []
for i, pred in enumerate(ml_preds):
    row = df.iloc[i]
    if pred == 0:
        ml_hit1.append(row["BM25_Hit1"]); ml_mrr.append(row["BM25_MRR"]); ml_ms.append(row["BM25_ms"]); ml_rerank_calls.append(0)
    elif pred == 1:
        ml_hit1.append(row["Embed_Hit1"]); ml_mrr.append(row["Embed_MRR"]); ml_ms.append(row["Embed_ms"]); ml_rerank_calls.append(0)
    else:
        ml_hit1.append(row["Rerank_Hit1"]); ml_mrr.append(row["Rerank_MRR"]); ml_ms.append(row["Rerank_ms"]); ml_rerank_calls.append(1)

df["ML_Hit1"] = ml_hit1
df["ML_MRR"] = ml_mrr
df["ML_ms"] = ml_ms
df["ML_rerank_calls"] = ml_rerank_calls

# 5. Print Final Benchmark + Router Comparison Table
print("\n======================================================================================")
print("  FINAL SYSTEM COMPARISON: ACCURACY vs. LATENCY vs. RERANK API COST (75 Queries)")
print("======================================================================================")
summary_rows = [
    ("1. BM25 Only",        df["BM25_Hit1"].mean()*100,   df["BM25_MRR"].mean(),   df["BM25_ms"].mean(),   0.0),
    ("2. Cohere Embed",     df["Embed_Hit1"].mean()*100,  df["Embed_MRR"].mean(),  df["Embed_ms"].mean(),  0.0),
    ("3. Hybrid (RRF)",     df["Hybrid_Hit1"].mean()*100, df["Hybrid_MRR"].mean(), df["Hybrid_ms"].mean(), 0.0),
    ("4. Hybrid + Rerank",  df["Rerank_Hit1"].mean()*100, df["Rerank_MRR"].mean(), df["Rerank_ms"].mean(), 100.0),
    ("5. Heuristic Router", df["H_Hit1"].mean()*100,      df["H_MRR"].mean(),      df["H_ms"].mean(),      df["H_rerank_calls"].mean()*100),
    ("6. Learned ML Router",df["ML_Hit1"].mean()*100,     df["ML_MRR"].mean(),     df["ML_ms"].mean(),     df["ML_rerank_calls"].mean()*100),
]

print(f"{'Strategy':<22} | {'Hit@1 (%)':<10} | {'MRR@10':<8} | {'Avg Latency':<12} | {'Rerank Calls Used'}")
print("-" * 86)
for name, hit1, mrr, ms, r_pct in summary_rows:
    print(f"{name:<22} | {hit1:>8.1f}%  | {mrr:>7.3f}  | {ms:>8.1f} ms  | {r_pct:>6.1f}%")

print("\n[Heuristic Router Breakdown by Category]")
print(pd.crosstab(df["category"], df["H_Route"]))

# 6. Generate Publication-Quality Plot 1: Failure Modes Bar Chart
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
cat_means = df.groupby("category")[["BM25_Hit1", "Embed_Hit1", "Hybrid_Hit1", "Rerank_Hit1", "H_Hit1"]].mean() * 100
cat_means.columns = ["BM25", "Cohere Embed", "Hybrid (RRF)", "Hybrid + Rerank", "Adaptive Router (Ours)"]
cat_means.plot(kind="bar", ax=ax, width=0.82, colormap="viridis")
ax.set_title("Retrieval Top-1 Accuracy (Hit@1) Across Failure Modes", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
ax.set_xlabel("Query Failure Bucket", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
ax.set_ylim(0, 112)
ax.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("failure_modes_chart.png")
print("\n[SAVED PLOT] failure_modes_chart.png")

# 7. Generate Plot 2: Accuracy vs. Latency Pareto Frontier
fig2, ax2 = plt.subplots(figsize=(8.5, 5), dpi=150)
for name, hit1, mrr, ms, r_pct in summary_rows:
    color = "#d95f02" if "Router" in name else "#1b9e77" if "Rerank" in name else "#7570b3"
    marker = "*" if "Router" in name else "o"
    size = 180 if "Router" in name else 100
    ax2.scatter(ms, hit1, s=size, color=color, marker=marker, zorder=5)
    ax2.annotate(f"  {name} ({r_pct:.0f}% Rerank)", (ms, hit1), fontsize=10, fontweight="bold" if "Router" in name else "normal", va="center")

ax2.set_title("Accuracy vs. Latency Trade-off (The Router Pareto Frontier)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Average Retrieval Latency (ms) [Lower is Better]", fontsize=11)
ax2.set_ylabel("Overall Top-1 Accuracy (%) [Higher is Better]", fontsize=11)
ax2.set_xlim(-10, 195)
ax2.set_ylim(55, 102)
plt.tight_layout()
plt.savefig("pareto_frontier.png")
print("[SAVED PLOT] pareto_frontier.png")