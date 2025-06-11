import json
import sys
import time
import re
import argparse
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os


message = (
    # --- task -------------------------------------------------------------- #
    "You are an API, not a chatbot.\n"
    "Task: For each user prompt decide if it is a QUERY that asks to locate or "
    "retrieve one or more contract clauses (return true) OR an instruction / "
    "formatting / summarising / translating request that is *not* a retrieval "
    "query (return false).\n\n"
    # --- output rules ------------------------------------------------------ #
    'Return **exactly one line** of JSON with a single key named "value" and a '
    "lower-case boolean.  Only allowed forms:\n"
    '  {"value": true}\n'
    '  {"value": false}'
    "No other keys. No arrays. No markdown fences, numbers, tables, or text."
)


FEWSHOT = [
    {"role": "user", "content": "Is there a cap of liability?"},
    {"role": "assistant", "content": '{"value": true}'},
]


def parse(raw: str) -> int:
    modelAns = raw
    modelAns = modelAns.strip()
    if modelAns.startswith("```"):
        modelAns = modelAns.strip("` \n")
    m = re.search(r"\{[^{}]*\}", modelAns, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "value" in obj:
                return 1 if obj["value"] else 0
        except json.JSONDecodeError:
            pass
    modelAns = modelAns.lower().split()[0]
    if modelAns in {"true", "yes"}:
        return 1
    elif modelAns in {"false", "no"}:
        return 0
    else:
        print( f"Skipped row: {raw}")
        return -1


def getAns(model: str, prompt: str) -> int:
    messages = [
        {"role": "system", "content": message},
        *FEWSHOT,
        {"role": "user", "content": prompt},
    ]
    resp = ollama.chat(model=model, messages=messages, options={"temperature": 0}, think=False)[
        "message"
    ]["content"].strip()

    val = parse(resp)
    if val == -1:
        print("Parsing unsuccessful. Trying with temperature 1")
        resp = ollama.chat(model=model, messages=messages, options={"temperature": 1})[
            "message"
        ]["content"].strip()
        return parse(resp)

    return val


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 script.py <model_name> <prompts.json>")
        sys.exit(1)

    model_name, json_path = argv
    with open(json_path, "r") as fh:
        rows = json.load(fh)

    truth, model, skipped = [], [], 0
    t0 = time.time()

    for row in rows:
        if not isinstance(row.get("value"), bool):
            skipped += 1
            continue

        trueVal = 1 if row["value"] else 0

        modelAns = getAns(model_name, row["prompt"])

        if modelAns == -1:
            skipped += 1
            continue

        truth.append(trueVal)
        model.append(modelAns)

    elapsed = time.time() - t0

    if truth:
        acc = accuracy_score(truth, model)
        prec = precision_score(truth, model, zero_division=0)
        rec = recall_score(truth, model, zero_division=0)

        print(f"\nModel: {model_name}")
        print(f"Accuracy : {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall   : {rec:.2f}")
        print(f"Total time          : {elapsed:.2f} s")
        print(f"Average time/prompt : {elapsed/len(truth):.2f} s")
        if skipped:
            print(f"Skipped rows        : {skipped}")
    else:
        print("No valid predictions to score.")


if __name__ == "__main__":
    main(sys.argv[1:])
