import pandas as pd
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from difflib import SequenceMatcher
import re

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_section_number(text):
    match = re.match(r"^\s*(\d+[a-zA-Z]?(\.\d+)*)(\s|$)", str(text))
    return match.group(1) if match else None

def clean_section(text):
    return re.sub(r"[^\w\d\. ]", "", str(text)).strip()

def get_top_sections(sources_df, msg_id, top_k=3):
    subset = sources_df[sources_df['project_message_id'] == msg_id].copy()
    subset = subset.sort_values(by='distance', ascending=False)
    top_sections = subset.head(top_k)['section'].tolist()
    return [extract_section_number(clean_section(s)) for s in top_sections if extract_section_number(clean_section(s))]

def run_eval(chats_file, sources_file, gt_file, ai_col, gt_col, out_csv):
    chats_df = pd.read_csv(chats_file)
    sources_df = pd.read_csv(sources_file)
    gt_df = pd.read_csv(gt_file)

    results = []
    for _, row in gt_df.iterrows():
        query = row['query']
        ai_resp = row[ai_col]
        gt_resp = row[gt_col]

        match = chats_df[chats_df['query'].str.strip() == query.strip()]
        if match.empty:
            results.append({"query": query, "message_id": None, "top_sections": None, "section_match": "NO_MATCH", "cosine_similarity": None})
            continue

        msg_id = match.iloc[0]['id']
        top_sections = get_top_sections(sources_df, msg_id)

        # SECTION MATCHING
        found_sections = [sec for sec in top_sections if sec and sec in ai_resp]
        section_match = (
            "ALL_MATCHED" if len(found_sections) == len(top_sections)
            else "PARTIAL_MATCHED" if found_sections else "NO_MATCH"
        )

        # COSINE SIMILARITY
        if pd.isna(ai_resp) or pd.isna(gt_resp):
            cos_sim = None
        else:
            emb1 = model.encode([ai_resp], convert_to_tensor=True).cpu()
            emb2 = model.encode([gt_resp], convert_to_tensor=True).cpu()
            cos_sim = float(cosine_similarity(emb1, emb2)[0][0])

        results.append({
            "query": query,
            "message_id": msg_id,
            "top_sections": top_sections,
            "section_match": section_match,
            "cosine_similarity": cos_sim
        })

    pd.DataFrame(results).to_csv(out_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chats_file", required=True)
    parser.add_argument("--sources_file", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--ai_column", required=True)
    parser.add_argument("--gt_column", required=True)
    parser.add_argument("--out_csv", default="eval_output.csv")
    args = parser.parse_args()

    run_eval(
        chats_file=args.chats_file,
        sources_file=args.sources_file,
        gt_file=args.gt_file,
        ai_col=args.ai_column,
        gt_col=args.gt_column,
        out_csv=args.out_csv
    )
