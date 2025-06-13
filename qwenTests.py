import json
import sys
import time
import re
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score

SYS_PROMPT = """\
You are an API, not a chatbot.
You will be given a user prompt that has been sent to a chat bot who is tasked with answering questions about construction contracts and displaying the information in the user's desired format.
Task: For each user prompt, decide if it is a QUERY that asks to locate or retrieve one or more contract clauses (return true) OR an instruction/formatting/summarizing/translating request that is NOT a retrieval query (return false).

Output format (exactly one line, valid JSON, no extra text, no markdown fences):
{"value": true}
{"value": false}
"""
keywordSys = """\
You are a helpful assistant. You will be given a user prompt that will be used as a search probe to retrieve information from contracts related to construction. 
Task: for each prompt, return a valid JSON that extracts 5 keywords from the prompt that will be the most helpful to search for during the retrieval process. 
If there are less than 5 keywords in the prompt, generate the remaining keyword(s) as synonyms or possible related words that could be helpful during the retrieval process.
Output format(exactly one line, valid JSON, no extra text, no markdown fences):
{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","clause"]}

"""
subquerySys = """\
You are a helpful assistant. You will be given a user prompt that will be used as a search probe to retrieve information from contracts related to construction. 
Task: for each prompt, return a valid JSON that extracts 5 keywords from the prompt that will be the most helpful to search for during the retrieval process. 
If there are less than 5 keywords in the prompt, generate the remaining keyword(s) as synonyms or possible related words that could be helpful during the retrieval process.
Output format(exactly one line, valid JSON, no extra text, no markdown fences):
{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","clause"]}

"""



FEWSHOT = [
    {"role": "user", "content": "Is there a cap of liability?"},
    {"role": "assistant", "content": '{"value": true}'},
    {"role": "user", "content": "Summarize the agreement in two lines."},
    {"role": "assistant", "content": '{"value": false}'},
]
fewshotKeyword = [
    {"role": "user", "content": "What are the liquidated damages?"},
    {"role": "assistant", "content": '{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","clause"]}'},

]
fewshotSubquery = [
    {"role": "user", "content": "What are the liquidated damages?"},
    {"role": "assistant", "content": '{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","clause"]}'},

]

def call_llama(model: str, messages: list) -> str:
    resp = ollama.chat(
        model=model, messages=messages, options={"temperature": 0}, think=False
    )["message"]["content"]
    return resp.strip()


def parse(raw: str) -> int:
    m = re.search(r"\{[^{}]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if obj.get("value") is True:
                return 1
            if obj.get("value") is False:
                return 0
        except json.JSONDecodeError:
            pass
    token = raw.lower().split()[0]
    if token in {"true", "yes"}:
        return 1
    if token in {"false", "no"}:
        return 0
    return -1


def classify_and_check(model: str, prompt: str):
    # primary classification
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        *FEWSHOT,
        {"role": "user", "content": prompt},
    ]
    primary = call_llama(model, messages)

    messagesKeyword= [
        {"role": "system", "content": keywordSys},
        *fewshotKeyword,
        {"role": "user", "content": prompt},
    ]
    keyword = ""
    if parse(primary) == 1:
        keyword = call_llama(model,messagesKeyword)

    messagesSubQuery= [
        {"role": "system", "content": subquerySys},
        *fewshotSubquery,
        {"role": "user", "content": prompt},
    ]
    subquery = call_llama(model,messagesSubQuery)
    return parse(primary), primary, keyword, subquery


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 script.py <model_name> <prompts.json>")
        sys.exit(1)

    model_name, path = argv
    with open(path) as fh:
        rows = json.load(fh)

    truth, model = [], []
    prompts, ground_truths = [], []
    primary_outputs, keywords = [], []

    t0 = time.time()
    for row in rows:
        prompt = row["prompt"]
        gt = 'true' if row["value"] else 'false'

        parsed, primary_raw, keyword, subquery = classify_and_check(model_name, prompt)
        if keyword:
            keywords.append(keyword)
        prompts.append(prompt)
        ground_truths.append(gt)
        primary_outputs.append(primary_raw)
        truth.append(1 if row["value"] else 0)
        model.append(parsed)

    elapsed = time.time() - t0

    # Filter out invalid predictions
    valid_indices = [i for i, p in enumerate(model) if p in (0, 1)]
    filt_truth = [truth[i] for i in valid_indices]
    filt_model = [model[i] for i in valid_indices]

    if filt_truth:
        acc = accuracy_score(filt_truth, filt_model)
        prec = precision_score(filt_truth, filt_model, zero_division=0)
        rec = recall_score(filt_truth, filt_model, zero_division=0)

        # Markdown table 1: model metrics
        print(f"\n| Model | Accuracy | Precision | Recall | Total Time (s) | Avg Time per Prompt (s) |")
        print("|---|---|---|---|---|---|")
        print(f"| {model_name} | {acc:.2f} | {prec:.2f} | {rec:.2f} | {elapsed:.2f} | {elapsed/len(filt_truth):.2f} |")

        # Markdown table 2: per-prompt results
        # print(f"\n| Prompt | Ground Truth | Primary Output |")
        # print("|---|---|---|")
        # for i in valid_indices:
        #     p = prompts[i].replace('|', '\\|')
        #     print(f"| {p} | {ground_truths[i]} | {primary_outputs[i]} |")
        # Keyword Display
        for kw in keywords:
            print(f"\n Keyword extractor: {kw}")
        for quer in subquery:
            print(f"\n Subqueries: {quer}")
    else:
        print("No valid predictions to score.")

if __name__ == "__main__":
    main(sys.argv[1:])
