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
If there are less than 5 keywords in the prompt, generate the remaining keyword(s) based on the fact that 
key words you are generating will be used to create subqueries that will be used as search probes for the retrieval process.
Output format(exactly one line, valid JSON, no extra text, no markdown fences):
{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","calculation"]}

"""
subquerySys = """\
You are a helpful assistant. You will be given a user prompt that will be used as a search probe to retrieve information from contracts related to construction. You will also be given 5 associated keywords. 
The keywords will either be directly extracted from the user prompt, or chosen by using the user prompt to generate words related to the query.All keywords will be helpful for effective retrieval of information.
Task: For each user prompt and associated keywords, generate subqueries that break the original query into steps, so that we can use the subqueries as search probes.
Each subquery must contain at least 1 keyword. You should not be repetitive or break down the query into too many steps. The goal is to improve accuracy and efficiency.  
Your subqueries will be given to a retrieval model, so they should communicate the information in the best way for a retrieval model to receive and execute.
Output format(no extra text, no markdown fences):
1. 'subquery 1' \n
2. 'subquery 2'\n
3. 'subquery 3'\n
... and so on until the subqueries cover the entire query. The fewshot you are given lists 4 subqueries but you should provide the correct amount of subqueries to cover the entire query, which may be more or less than 4.

"""



FEWSHOT = [
    {"role": "user", "content": "Is there a cap of liability?"},
    {"role": "assistant", "content": '{"value": true}'},
    {"role": "user", "content": "Summarize the agreement in two lines."},
    {"role": "assistant", "content": '{"value": false}'},
]
fewshotKeyword = [
    {"role": "user", "content": "What are the liquidated damages?"},
    {"role": "assistant", "content": '{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","calculation"]}'},
]
fewshotSubquery = [
    {"role": "user", "content": '{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","calculation"]}'},
    {"role": "assistant", "content": "1. 'Find the clause defining liquidated damages'\n2. 'Locate any penalty provisions related to damages'\n3. 'Extract compensation terms specified for damages'\n4. 'Extract numerical examples illustrating liquidated damages calculation'"}
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
    primary = call_llama("qwen3:4b", messages)

    messagesKeyword= [
        {"role": "system", "content": keywordSys},
        *fewshotKeyword,
        {"role": "user", "content": prompt},
    ]
    keyword = ""
    subquery = ""
    if parse(primary) == 1:
        keyword = call_llama(model,messagesKeyword)
        messagesSubQuery= [
            {"role": "system", "content": subquerySys},
            *fewshotSubquery,
            {"role": "user", "content": keyword},
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
    primary_outputs, keywords, subqueries = [], [], []

    t0 = time.time()
    for row in rows:
        prompt = row["prompt"]
        gt = 'true' if row["value"] else 'false'

        parsed, primary_raw, keyword, subquery = classify_and_check(model_name, prompt)
        if keyword:
            keywords.append(keyword)
            subqueries.append(subquery)
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
        for j in range(len(keywords)):
            print(f"\n Keyword extractor: {keywords[j]}")
            print(f"\n Subqueries: {subqueries[j]}")
    else:
        print("No valid predictions to score.")

if __name__ == "__main__":
    main(sys.argv[1:])
