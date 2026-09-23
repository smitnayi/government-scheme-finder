"""
audit_laptop_nlp.py — Deep NLP audit of the "college student laptop" query.
Generates:
  1. results/laptop_query_nlp_audit.csv
  2. results/laptop_nlp_audit_plot.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import search_engine

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load search engine
search_engine.load_index()

# 1. Evaluate "college student laptop" under Semantic and Keyword search
query = "college student laptop"
k = 10

sem_results = search_engine.semantic_search(query, k=k)
kw_results = search_engine.keyword_search(query, k=k)

audit_records = []

import re

# Helper to categorize demographic target with word boundaries
def classify_target(name: str, desc: str, tags: str) -> str:
    combined = f"{name} {desc} {tags}".lower()
    if re.search(r'\b(labour|laborer|laborers|worker|bocw|industrial worker)\b', combined):
        return "Labour / Worker Family"
    elif re.search(r'\b(visually impaired|disability|disabled|handicapped|speech impaired)\b', combined):
        return "Disability / PwD"
    elif re.search(r'\b(teacher|teachers)\b', combined):
        return "Teachers"
    elif re.search(r'\b(media|journalist|media persons)\b', combined):
        return "Media Persons"
    elif re.search(r'\b(merit|meritorious|topper|toppers|first division)\b', combined):
        return "Merit / Academic Toppers"
    elif re.search(r'\b(sc|st|scheduled caste|scheduled tribe)\b', combined):
        return "SC / ST Students"
    elif re.search(r'\b(college|higher education|university|class xi|class 11|post-matric)\b', combined):
        return "General / State Students"
    else:
        return "General Welfare"

# Helper to check if scheme actually provides a laptop/computer
def provides_laptop(name: str, desc: str, tags: str) -> bool:
    combined = f"{name} {desc} {tags}".lower()
    return "laptop" in combined or "computer" in combined or "tablet" in combined

# Process Semantic results
for rank, (_, row) in enumerate(sem_results.iterrows(), 1):
    target = classify_target(str(row['scheme_name']), str(row['description']), str(row['tags']))
    has_laptop = provides_laptop(str(row['scheme_name']), str(row['description']), str(row['tags']))
    
    nlp_reason = ""
    user_perception = ""
    if "labour" in target.lower():
        nlp_reason = "Model matches 'laptop' + 'enrolled in professional courses' (college student equivalent). Mean pooling blends the labour demographic."
        user_perception = "Surprising ('Out of the box'): Scheme legitimately provides college laptops, but requires parents to be registered construction/industrial laborers."
    elif has_laptop:
        nlp_reason = "Strong semantic alignment with student laptop assistance and educational technology support."
        user_perception = f"Relevant laptop scheme targeting {target}."
    else:
        nlp_reason = "General educational support without direct hardware provision."
        user_perception = "Not a direct laptop scheme."
        
    audit_records.append({
        "Search_Method": "Semantic (MiniLM)",
        "Rank": rank,
        "Scheme_ID": row['scheme_id'],
        "Scheme_Name": row['scheme_name'],
        "Score": round(row['score'], 3),
        "Provides_Laptop": "Yes" if has_laptop else "No",
        "Target_Demographic": target,
        "NLP_Matching_Mechanism": nlp_reason,
        "Citizen_Perception_Analysis": user_perception
    })

# Process Keyword results
for rank, (_, row) in enumerate(kw_results.iterrows(), 1):
    target = classify_target(str(row['scheme_name']), str(row['description']), str(row['tags']))
    has_laptop = provides_laptop(str(row['scheme_name']), str(row['description']), str(row['tags']))
    
    if has_laptop:
        nlp_reason = "Exact keyword overlap on 'laptop', 'student', and 'college'."
        user_perception = "Accurate keyword match."
    else:
        nlp_reason = "Lexical match on generic words 'college' and 'student'. Zero semantic understanding of laptop intent."
        user_perception = "Irrelevant noise: Hostels/Scholarships that mention 'college student' but provide no computers."
        
    audit_records.append({
        "Search_Method": "Keyword (Token Overlap)",
        "Rank": rank,
        "Scheme_ID": row['scheme_id'],
        "Scheme_Name": row['scheme_name'],
        "Score": round(row['score'], 3),
        "Provides_Laptop": "Yes" if has_laptop else "No",
        "Target_Demographic": target,
        "NLP_Matching_Mechanism": nlp_reason,
        "Citizen_Perception_Analysis": user_perception
    })

audit_df = pd.DataFrame(audit_records)
csv_path = os.path.join(RESULTS_DIR, "laptop_query_nlp_audit.csv")
audit_df.to_csv(csv_path, index=False)
print(f"[OK] Saved audit CSV to {csv_path}")

# 2. Visualizations
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

# Panel 1: Intent Precision (Does the top 10 actually provide a laptop?)
ax1 = fig.add_subplot(gs[0, 0])
sem_laptop_count = sum(1 for r in audit_records if r['Search_Method'] == 'Semantic (MiniLM)' and r['Provides_Laptop'] == 'Yes')
kw_laptop_count = sum(1 for r in audit_records if r['Search_Method'] == 'Keyword (Token Overlap)' and r['Provides_Laptop'] == 'Yes')

methods = ['Semantic Search', 'Keyword Search']
counts = [sem_laptop_count * 10, kw_laptop_count * 10]  # Percentage of top 10
colors = ['#2E7D32', '#C62828']

bars = ax1.bar(methods, counts, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Top-10 Precision for Laptop Intent (%)', fontsize=11, fontweight='bold')
ax1.set_title('A. Benefit Intent Precision\n("Does retrieved scheme actually give a laptop?")', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 105)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{int(yval)}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Panel 2: Target Demographic Distribution of Semantic Laptop Results
ax2 = fig.add_subplot(gs[0, 1])
sem_demographics = [r['Target_Demographic'] for r in audit_records if r['Search_Method'] == 'Semantic (MiniLM)' and r['Provides_Laptop'] == 'Yes']
demo_counts = pd.Series(sem_demographics).value_counts()

palette = ['#1976D2', '#F57C00', '#7B1FA2', '#388E3C', '#78909C']
ax2.pie(demo_counts.values, labels=demo_counts.index, autopct='%1.0f%%', startangle=140, colors=palette[:len(demo_counts)],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax2.set_title('B. Demographic Distribution of Retrieved Laptop Schemes\n(Why "Labour" Surfaces in Vector Space)', fontsize=12, fontweight='bold')

# Panel 3: Semantic Score vs Demographic Conditioning across Query Variations
ax3 = fig.add_subplot(gs[1, :])
query_variations = [
    "college student laptop",
    "free laptop for college students general category",
    "laptop financial assistance for children of laborers",
    "laptop for visually impaired disabled student"
]

variant_scores = []
target_schemes = [
    ("cysky", "Chhattisgarh Yuva Suchna Kranti (General Higher Ed)"),
    ("lpas-glwb", "Gujarat Labour Welfare Board Laptop Scheme"),
    ("sflvisgpg", "Free Laptops for Visually Impaired Students"),
    ("lsmsscst", "Meritorious SC/ST Laptop Scheme")
]

scores_matrix = []
for q in query_variations:
    res = search_engine.semantic_search(q, k=50)
    score_map = dict(zip(res['scheme_id'], res['score']))
    row_scores = [score_map.get(sid, 0.0) for sid, _ in target_schemes]
    scores_matrix.append(row_scores)

scores_matrix = np.array(scores_matrix)  # shape: (4 queries, 4 schemes)

x = np.arange(len(query_variations))
width = 0.2
labels = [label for _, label in target_schemes]
scheme_colors = ['#1E88E5', '#E53935', '#8E24AA', '#FB8C00']

for idx in range(len(target_schemes)):
    ax3.bar(x + idx*width - 1.5*width, scores_matrix[:, idx], width, label=labels[idx], color=scheme_colors[idx], edgecolor='black', linewidth=0.8)

ax3.set_xticks(x)
ax3.set_xticklabels([
    'Generic:\n"college student laptop"',
    'General Category:\n"free laptop for college students\ngeneral category"',
    'Labour-Specific:\n"laptop financial assistance\nfor children of laborers"',
    'Disability-Specific:\n"laptop for visually impaired\ndisabled student"'
], fontsize=10)
ax3.set_ylabel('Cosine Similarity Score', fontsize=11, fontweight='bold')
ax3.set_title('C. Semantic Disentanglement: How Query Specificity Modulates Demographic Retrieval', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 0.9)
ax3.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

plot_path = os.path.join(RESULTS_DIR, "laptop_nlp_audit_plot.png")
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
print(f"[OK] Saved audit plot to {plot_path}")
plt.close()
