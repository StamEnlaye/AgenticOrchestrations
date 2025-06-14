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

CHECKER_SYS = """\
You are a verification assistant. Your job is to check whether the previous assistant correctly performed the following task:

Task: You will be given a user prompt that has been sent to a chat bot who is tasked with answering questions about construction contracts and displaying the information in the user's desired format.
For each user prompt, decide if it is a QUERY that asks to locate or retrieve one or more contract clauses (return true)  
OR an instruction/formatting/summarizing/translating request that is NOT a retrieval query (return false).

Now you will be given:
  1) the original user prompt  
  2) the assistant's output  

Validate two things:
  • Logical correctness: that “value” matches the right answer for this prompt  
  • Formatting correctness: exactly one line of valid JSON, a single key "value" with a lowercase boolean, no extra keys, no markdown fences, no commentary

If both are OK, return the output unchanged.  
If the format is wrong but the label is correct, strip away extra characters.  
If the label is wrong, correct it to the proper boolean.  

Return exactly one line and nothing else.
"""

FEWSHOT = [
    {"role": "user", "content": "Is there a cap of liability?"},
    {"role": "assistant", "content": '{"value": true}'},
    {"role": "user", "content": "Summarize the agreement in two lines."},
    {"role": "assistant", "content": '{"value": false}'},
]

def checkerPrompt(prompt: str, previous_output: str) -> str:
    return (
        f"Original user prompt:\n\"{prompt}\"\n\n"
        f"Assistant output:\n{previous_output}\n\n"
        f"Please apply the rules from the system message."
    )


def call_llama(model: str, messages: list) -> str:
    resp = ollama.chat(
        model=model, messages=messages, options={"temperature": 0}, think=False
    )["message"]["content"]
    return resp.strip()


def parse(raw: str) -> int:
    true_pattern = r"\{\s*\"value\"\s*:\s*true\s*\}"  # matches exactly one true literal
    false_pattern = r"\{\s*\"value\"\s*:\s*false\s*\}"  # matches exactly one false literal

    trues = re.findall(true_pattern, raw, flags=re.IGNORECASE)
    falses = re.findall(false_pattern, raw, flags=re.IGNORECASE)
    if len(trues) == 1 and len(falses) == 0:
        return 1
    if len(falses) == 1 and len(trues) == 0:
        return 0

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
    # verification
    checker_msgs = [
        {"role": "system",  "content": CHECKER_SYS},
        {"role": "user",    "content": checkerPrompt(prompt, primary)},
    ]
    verified = call_llama(model, checker_msgs)

    return parse(verified), primary, verified


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 script.py <model_name> <prompts.json>")
        sys.exit(1)

    model_name, path = argv
    with open(path) as fh:
        rows = json.load(fh)

    truth, model = [], []
    prompts, ground_truths = [], []
    primary_outputs, verified_outputs = [], []

    t0 = time.time()
    for row in rows:
        prompt = row["prompt"]
        gt = 'true' if row["value"] else 'false'

        parsed, primary_raw, verified_raw = classify_and_check(model_name, prompt)

        prompts.append(prompt)
        ground_truths.append(gt)
        primary_outputs.append(primary_raw)
        verified_outputs.append(verified_raw)

        truth.append(1 if row["value"] else 0)
        model.append(parsed)

    elapsed = time.time() - t0

    # Filter out invalid predictions
    valid_indices = [i for i, p in enumerate(model) if p in (0, 1)]
    filt_truth = [truth[i] for i in valid_indices]
    filt_preds = [model[i] for i in valid_indices]

    if filt_truth:
        acc = accuracy_score(filt_truth, filt_preds)
        prec = precision_score(filt_truth, filt_preds, zero_division=0)
        rec = recall_score(filt_truth, filt_preds, zero_division=0)

        # Markdown table 1: model metrics
        print(f"\n| Model | Accuracy | Precision | Recall | Total Time (s) | Avg Time per Prompt (s) |")
        print("|---|---|---|---|---|---|")
        print(f"| {model_name} | {acc:.2f} | {prec:.2f} | {rec:.2f} | {elapsed:.2f} | {elapsed/len(filt_truth):.2f} |")

        # Markdown table 2: per-prompt results
        print(f"\n| Prompt | Ground Truth | Primary Output | Verification Output |")
        print("|---|---|---|---|")
        for i in valid_indices:
            p = prompts[i].replace('|', '\\|')
            print(f"| {p} | {ground_truths[i]} | {primary_outputs[i]} | {verified_outputs[i]} |")
    else:
        print("No valid predictions to score.")

if __name__ == "__main__":
    main(sys.argv[1:])
